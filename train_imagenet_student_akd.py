"""
ImageNet knowledge distillation: AKD_L1 / AKD / SoftAKD

Teacher : ResNet-34  (pretrained on ImageNet via PyTorch model zoo)
Student : ResNet-18  (trained from scratch)

Distill modes
-------------
  AKD_L1   — batch-to-batch KL only (L1 / PKT), no anchors needed
  AKD      — full anchor-based KD: L1 + L2 (batch-anchor) + L3 (anchor-batch)
  SoftAKD  — AKD with adaptive per-sample sigma blending

Usage
-----
  python train_imagenet_student_akd.py --distill AKD_L1 --data_root ./data/imagenet --gids 0,1,2,3
  python train_imagenet_student_akd.py --distill AKD     --data_root ./data/imagenet --gids 0,1,2,3
  python train_imagenet_student_akd.py --distill SoftAKD --lambda_soft 0.3 --data_root ./data/imagenet --gids 0,1,2,3
"""
from __future__ import print_function

import os
import sys
import time
import argparse
import random
from collections import defaultdict
from math import cos, pi

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

cudnn.benchmark = True

# ── Models ────────────────────────────────────────────────────────────────────
from models.resnet_imagenet import resnet18, resnet34

# ── Distillation losses ───────────────────────────────────────────────────────
from distiller_zoo.AKD     import akd_loss, AnchorNet
from distiller_zoo.SoftAKD import soft_akd_loss, DisagreementScaler
from distiller_zoo.KD      import DistillKL

# ── Utilities ─────────────────────────────────────────────────────────────────
from helper.util import AverageMeter, accuracy


# ──────────────────────────────────────────────────────────────────────────────
# Learning rate schedule
# ──────────────────────────────────────────────────────────────────────────────

def adjust_learning_rate(optimizer, epoch, opt):
    if opt.lr_decay == 'step':
        steps = sum(epoch > np.asarray(opt.lr_decay_epochs))
        lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
    else:  # cos
        lr = opt.learning_rate * (1 + cos(pi * epoch / opt.epochs)) / 2
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ──────────────────────────────────────────────────────────────────────────────
# ImageNet data
# ──────────────────────────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_TRAIN_TF = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
_VAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])
_ANCHOR_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(_MEAN, _STD),
])


def get_imagenet_loaders(data_root, batch_size, num_workers):
    class _InstanceFolder(datasets.ImageFolder):
        def __getitem__(self, idx):
            img, lbl = super().__getitem__(idx)
            return img, lbl, idx

    train_loader = DataLoader(
        _InstanceFolder(os.path.join(data_root, 'train'), transform=_TRAIN_TF),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        datasets.ImageFolder(os.path.join(data_root, 'val'), transform=_VAL_TF),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers // 2, pin_memory=True,
    )
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# Anchor set construction  (used only for AKD / SoftAKD)
# ──────────────────────────────────────────────────────────────────────────────

def build_anchor_set(teacher, data_root, num_classes, max_scan_per_class,
                     batch_size, num_workers, cache_path=None):
    """Select one representative anchor image per class.

    Scans at most `max_scan_per_class` images per class, picks the most
    central one (highest average cosine similarity to class-mates).
    Result is cached so recomputation is only needed once.

    Returns
    -------
    anchor_images : FloatTensor [num_classes, 3, 224, 224]
    anchor_labels : LongTensor  [num_classes]
    """
    if cache_path and os.path.isfile(cache_path):
        print(f'[Anchor] Loading from cache: {cache_path}')
        saved = torch.load(cache_path, map_location='cpu')
        return saved['images'], saved['labels']

    print(f'[Anchor] Scanning up to {max_scan_per_class} images/class …')
    dataset = datasets.ImageFolder(os.path.join(data_root, 'train'),
                                   transform=_ANCHOR_TF)

    class_indices = defaultdict(list)
    for idx, (_, lbl) in enumerate(dataset.imgs):
        class_indices[lbl].append(idx)

    sampled_idx = []
    for cls in range(num_classes):
        pool = class_indices[cls]
        sampled_idx.extend(random.sample(pool, min(max_scan_per_class, len(pool))))

    loader = DataLoader(Subset(dataset, sampled_idx), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)

    teacher.eval()
    class_images   = defaultdict(list)
    class_features = defaultdict(list)

    with torch.no_grad():
        for imgs, labels in loader:
            # f5 = feat_list[-1]: already the 512-dim pooled vector
            feats = teacher(imgs.cuda(), is_feat=True)[0][-1].cpu()
            for i in range(imgs.size(0)):
                lbl = labels[i].item()
                class_images[lbl].append(imgs[i])
                class_features[lbl].append(feats[i])

    anchors = []
    for cls in range(num_classes):
        imgs_cls  = class_images[cls]
        feats_cls = class_features[cls]
        if not imgs_cls:
            anchors.append(dataset[class_indices[cls][0]][0])
            continue
        feat_mat = torch.stack(feats_cls).numpy()
        sim_mat  = sklearn_cosine(feat_mat)
        best     = int(np.argmax(sim_mat.sum(axis=1)))
        anchors.append(imgs_cls[best])

    anchor_images = torch.stack(anchors)
    anchor_labels = torch.arange(num_classes, dtype=torch.long)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save({'images': anchor_images, 'labels': anchor_labels}, cache_path)
        print(f'[Anchor] Cached to {cache_path}')

    return anchor_images, anchor_labels


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def _l1_loss(s_pooled, t_pooled, eps=1e-7):
    """Batch-to-batch KL divergence (L1 term of AKD), using sum to match akd_loss."""
    s = F.normalize(s_pooled, p=2, dim=1)
    t = F.normalize(t_pooled, p=2, dim=1)
    s_sim = (torch.mm(s, s.T) + 1) / 2
    t_sim = (torch.mm(t, t.T) + 1) / 2
    s_sim = s_sim / s_sim.sum(dim=1, keepdim=True)
    t_sim = t_sim / t_sim.sum(dim=1, keepdim=True)
    return torch.sum(t_sim * torch.log((t_sim + eps) / (s_sim + eps)))


def train_epoch(epoch, train_loader, model_t, model_s,
                optimizer, criterion_cls, criterion_div,
                opt,
                # anchor arguments — only used for AKD / SoftAKD
                anchor_images_gpu=None, anchor_labels_gpu=None,
                anchor_net=None, a_feat_t=None,
                optimizer_anchor=None, scaler=None):

    model_s.train()
    model_t.eval()
    if anchor_net is not None:
        anchor_net.train()

    losses     = AverageMeter()
    losses_ce  = AverageMeter()
    losses_kd  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        images  = data[0].cuda(non_blocking=True)
        targets = data[1].cuda(non_blocking=True)

        feat_s, logit_s = model_s(images, is_feat=True)
        s_pooled = feat_s[-1]   # [B, 512]
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            t_pooled = feat_t[-1]   # [B, 512]

        # ── CE + optional KL divergence ───────────────────────────────────────
        loss_ce  = criterion_cls(logit_s, targets)
        loss_div = (criterion_div(logit_s, logit_t)
                    if opt.alpha is not None else torch.tensor(0.0).cuda())

        # ── Distillation loss ─────────────────────────────────────────────────
        if opt.distill == 'AKD_L1':
            # L1 only: batch-to-batch KL, sum — consistent with full akd_loss
            loss_kd = _l1_loss(s_pooled, t_pooled)

        else:  # AKD or SoftAKD — requires anchors
            attended = anchor_net(anchor_images_gpu)          # [A, 3, 224, 224]
            feat_s_anc, _ = model_s(attended, is_feat=True)
            a_feat_s = feat_s_anc[-1]                        # [A, 512]

            if opt.distill == 'AKD':
                loss_kd = akd_loss(t_pooled, s_pooled, a_feat_t, a_feat_s, opt)
            else:  # SoftAKD
                loss_kd, _ = soft_akd_loss(
                    t_pooled, s_pooled,
                    a_feat_t, a_feat_s,
                    targets, anchor_labels_gpu,
                    sigma=None,
                    lambda_soft=opt.lambda_soft,
                    opt=opt,
                    scaler=scaler,
                )

        alpha = opt.alpha if opt.alpha is not None else 0.0
        loss  = opt.gamma * loss_ce + alpha * loss_div + opt.l_KD * loss_kd

        acc1, acc5 = accuracy(logit_s, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        losses_ce.update(loss_ce.item(), images.size(0))
        losses_kd.update(loss_kd.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        optimizer.zero_grad()
        if optimizer_anchor is not None:
            optimizer_anchor.zero_grad()
        loss.backward()
        optimizer.step()
        if optimizer_anchor is not None:
            optimizer_anchor.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % opt.print_freq == 0:
            print(f'Epoch [{epoch}][{idx}/{len(train_loader)}]  '
                  f'Time {batch_time.avg:.2f}  '
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                  f'CE {losses_ce.avg:.4f}  KD {losses_kd.avg:.4f}  '
                  f'Acc@1 {top1.val:.2f} ({top1.avg:.2f})  '
                  f'Acc@5 {top5.val:.2f} ({top5.avg:.2f})')
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def validate(val_loader, model, criterion):
    model.eval()
    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs, targets = imgs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            logits = model(imgs)
            loss   = criterion(logits, targets)
            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0].item(), imgs.size(0))
            top5.update(acc5[0].item(), imgs.size(0))

    print(f' * Acc@1 {top1.avg:.3f}  Acc@5 {top5.avg:.3f}')
    return top1.avg, top5.avg, losses.avg


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_option():
    p = argparse.ArgumentParser('ImageNet AKD_L1 / AKD / SoftAKD distillation')

    p.add_argument('--distill', type=str, default='AKD_L1',
                   choices=['AKD_L1', 'AKD', 'SoftAKD'])

    # Data / hardware
    p.add_argument('--data_root',   type=str, default='./data/imagenet')
    p.add_argument('--gids',        type=str, default='0,1,2,3')
    p.add_argument('--batch_size',  type=int, default=256)
    p.add_argument('--num_workers', type=int, default=16)

    # Training schedule
    p.add_argument('--epochs',          type=int,   default=100)
    p.add_argument('--learning_rate',   type=float, default=0.1)
    p.add_argument('--lr_decay',        type=str,   default='step', choices=['step', 'cos'])
    p.add_argument('--lr_decay_epochs', type=str,   default='30,60,90')
    p.add_argument('--lr_decay_rate',   type=float, default=0.1)
    p.add_argument('--weight_decay',    type=float, default=1e-4)
    p.add_argument('--momentum',        type=float, default=0.9)

    # Loss weights
    p.add_argument('--gamma', type=float, default=1.0,  help='CE loss weight')
    p.add_argument('--alpha', type=float, default=None, help='KD loss weight (None = off)')
    p.add_argument('--kd_T', type=float,  default=4.0,  help='temperature for KD term')
    p.add_argument('--l_KD', type=float,  default=1.0,
                   help='overall distillation loss weight (replaces l_AKD for all modes)')
    # AKD-specific (only used when distill=AKD or SoftAKD)
    p.add_argument('--l_1', type=float, default=1.0, help='batch-batch KL weight in AKD loss')
    p.add_argument('--l_2', type=float, default=0.1, help='anchor KL split weight in AKD loss')

    # SoftAKD-specific
    p.add_argument('--lambda_soft', type=float, default=0.3)
    p.add_argument('--lambda_d0',   type=float, default=0.3)
    p.add_argument('--sigma_temp',  type=float, default=1.0)
    p.add_argument('--lock_epoch',  type=int,   default=60)

    # Anchor settings (only used when distill=AKD or SoftAKD)
    p.add_argument('--num_classes',        type=int, default=1000)
    p.add_argument('--max_scan_per_class', type=int, default=20)
    p.add_argument('--anchor_cache', type=str, default=None,
                   help='path to save/load anchor tensor')

    # Logging / saving
    p.add_argument('--print_freq',  type=int, default=200)
    p.add_argument('--save_freq',   type=int, default=10)
    p.add_argument('--trial',       type=str, default='1')
    p.add_argument('--resume_path', type=str, default=None)

    opt = p.parse_args()

    opt.lr_decay_epochs = [int(e) for e in opt.lr_decay_epochs.split(',')]

    tag = (f'T_resnet34_S_resnet18_{opt.distill}'
           f'_lKD{opt.l_KD}_g{opt.gamma}'
           + (f'_lsoft{opt.lambda_soft}' if opt.distill == 'SoftAKD' else '')
           + f'_trial{opt.trial}')
    opt.save_folder = os.path.join('save', 'student_model', tag)
    opt.tb_folder   = os.path.join('save', 'student_tensorboards', tag)
    os.makedirs(opt.save_folder, exist_ok=True)
    os.makedirs(opt.tb_folder,   exist_ok=True)

    if opt.anchor_cache is None:
        opt.anchor_cache = os.path.join('save', 'anchor_set_imagenet_resnet34.pth')

    return opt


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    best_acc = 0.0
    opt = parse_option()
    print(opt)

    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    gids   = [int(g) for g in opt.gids.split(',')]

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader = get_imagenet_loaders(
        opt.data_root, opt.batch_size, opt.num_workers
    )

    # ── Models ────────────────────────────────────────────────────────────────
    print('Loading teacher (ResNet-34, ImageNet pretrained) …')
    model_t = resnet34(pretrained=True, num_classes=1000)
    model_t = nn.DataParallel(model_t, device_ids=gids).cuda().eval()

    model_s = resnet18(pretrained=False, num_classes=1000)
    model_s = nn.DataParallel(model_s, device_ids=gids).cuda()

    if opt.resume_path:
        ckpt = torch.load(opt.resume_path, map_location='cpu')
        model_s.load_state_dict(ckpt['model'])
        print(f"Resumed from epoch {ckpt['epoch']}  best_acc={ckpt.get('best_acc', 0):.2f}")

    # ── Anchor setup (AKD / SoftAKD only) ────────────────────────────────────
    anchor_images_gpu = anchor_labels_gpu = anchor_net = a_feat_t = optimizer_anchor = None

    if opt.distill in ('AKD', 'SoftAKD'):
        anchor_images, anchor_labels = build_anchor_set(
            model_t,
            data_root=opt.data_root,
            num_classes=opt.num_classes,
            max_scan_per_class=opt.max_scan_per_class,
            batch_size=opt.batch_size,
            num_workers=opt.num_workers,
            cache_path=opt.anchor_cache,
        )
        anchor_images_gpu = anchor_images.cuda()
        anchor_labels_gpu = anchor_labels.cuda()

        with torch.no_grad():
            a_feat_t = model_t(anchor_images_gpu, is_feat=True)[0][-1].detach()
        print(f'[Anchor] teacher features: {a_feat_t.shape}')

        anchor_net = AnchorNet(
            anchor_batch_size=opt.num_classes, DIM=224, weight_size=14
        ).cuda()
        optimizer_anchor = optim.SGD(anchor_net.parameters(), lr=1e-3, weight_decay=1e-5)

    # ── Criteria ──────────────────────────────────────────────────────────────
    criterion_cls = nn.CrossEntropyLoss().cuda()
    criterion_div = DistillKL(opt.kd_T).cuda()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = optim.SGD(
        model_s.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )

    # ── DisagreementScaler (SoftAKD only) ─────────────────────────────────────
    scaler = None
    if opt.distill == 'SoftAKD':
        scaler = DisagreementScaler(
            lock_epoch=opt.lock_epoch,
            lock_std_only=True,
            save_folder=opt.save_folder,
        )

    # ── Teacher baseline ──────────────────────────────────────────────────────
    print('Validating teacher …')
    validate(val_loader, model_t, criterion_cls)

    # ── Training loop ─────────────────────────────────────────────────────────
    time_start = time.time()
    for epoch in range(1, opt.epochs + 1):

        lr = adjust_learning_rate(optimizer, epoch, opt)
        print(f'Epoch {epoch}  lr={lr:.5f}')

        train_acc1, train_acc5, train_loss = train_epoch(
            epoch, train_loader, model_t, model_s,
            optimizer, criterion_cls, criterion_div,
            opt,
            anchor_images_gpu=anchor_images_gpu,
            anchor_labels_gpu=anchor_labels_gpu,
            anchor_net=anchor_net,
            a_feat_t=a_feat_t,
            optimizer_anchor=optimizer_anchor,
            scaler=scaler,
        )

        if scaler is not None:
            scaler.step_epoch(epoch)

        print(f'Epoch {epoch:3d}  train Acc@1 {train_acc1:.3f}  '
              f'Acc@5 {train_acc5:.3f}  Loss {train_loss:.4f}')

        test_acc1, test_acc5, test_loss = validate(val_loader, model_s, criterion_cls)

        logger.log_value('train_acc1', train_acc1, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('test_acc1',  test_acc1,  epoch)
        logger.log_value('test_acc5',  test_acc5,  epoch)
        logger.log_value('test_loss',  test_loss,  epoch)

        if test_acc1 > best_acc:
            best_acc = test_acc1
            torch.save(
                {'epoch': epoch, 'model': model_s.state_dict(), 'best_acc': best_acc},
                os.path.join(opt.save_folder, 'resnet18_best.pth'),
            )
            print(f'  → new best: {best_acc:.3f}')

        if epoch % opt.save_freq == 0:
            torch.save(
                {'epoch': epoch, 'model': model_s.state_dict(), 'accuracy': test_acc1},
                os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth'),
            )

    elapsed = (time.time() - time_start) / 3600.0
    print(f'Best accuracy: {best_acc:.3f}   Total time: {elapsed:.2f} h')

    torch.save({'opt': opt, 'model': model_s.state_dict()},
               os.path.join(opt.save_folder, 'resnet18_last.pth'))


if __name__ == '__main__':
    main()
