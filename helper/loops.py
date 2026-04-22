from __future__ import print_function, division

import sys
import time
import heapq
import torch
import torch.nn.functional as F
from .util import AverageMeter, accuracy, reduce_tensor

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader) if opt.dali is None else (train_loader._size + opt.batch_size - 1) // opt.batch_size

    end = time.time()
    for idx, batch_data in enumerate(train_loader):
        images, labels = batch_data
        
        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # ===================forward=====================
        output = model(images)
        loss = criterion(output, labels)
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(output, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()
            
    return top1.avg, top5.avg, losses.avg

def _norm_beta1(epoch, dataset):
    """Dynamic beta1 schedule for NORM distillation."""
    if dataset == 'cifar100':
        if epoch < 10:   return 0.1
        if epoch < 20:   return 0.5
        if epoch < 180:  return 1.0
        if epoch < 210:  return 0.5
        return 0.1
    else:  # imagenet
        if epoch < 5:    return 0.1
        if epoch < 15:   return 0.5
        if epoch < 60:   return 0.9
        if epoch < 80:   return 0.5
        return 0.1


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt, g=None, g_opt=None):
    """one epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            images, labels, index, contrast_idx = data
        else:
            images, labels = data
        
        if opt.distill == 'semckd' and images.shape[0] < opt.batch_size:
            continue

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if opt.distill in ['crd']:
                index = index.cuda()
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        cls_t = model_t.module.get_feat_modules()[-1] if opt.multiprocessing_distributed else model_t.get_feat_modules()[-1]
        
        # cls + kl div
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)
        
        # other kd loss
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s, f_t = module_list[1](feat_s[opt.hint_layer], feat_t[opt.hint_layer])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'attention':
            # include 1, exclude -1.
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill == 'semckd':
            s_value, f_target, weight = module_list[1](feat_s[1:-1], feat_t[1:-1])
            loss_kd = criterion_kd(s_value, f_target, weight)                                                 
        elif opt.distill == 'srrl':
            trans_feat_s, pred_feat_s = module_list[1](feat_s[-1], cls_t)
            loss_kd = criterion_kd(trans_feat_s, feat_t[-1]) + criterion_kd(pred_feat_s, logit_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill in ('ttm', 'wttm'):
            loss_kd = criterion_kd(logit_s, logit_t)
        elif opt.distill == 'norm':
            # Logit normalization (overrides loss_cls / loss_div computed above)
            n_cls = logit_s.size(1)
            logit_s_n = F.layer_norm(logit_s, (n_cls,), eps=1e-7) * opt.ceta
            logit_t_n = F.layer_norm(logit_t, (n_cls,), eps=1e-7) * opt.ceta
            loss_cls = criterion_cls(logit_s_n, labels)
            loss_div = criterion_div(logit_s_n, logit_t_n)
            # Feature MSE loss with channel expansion
            norm_connector = module_list[1]
            f_s_raw = feat_s[-2]
            f_t_raw = feat_t[-2]
            pool_size = f_t_raw.shape[2] // f_s_raw.shape[2]
            if pool_size > 1:
                f_t_raw = F.max_pool2d(f_t_raw, pool_size, pool_size)
            f_s = norm_connector(f_s_raw)
            f_t = f_t_raw.repeat(1, opt.co_sponge, 1, 1)
            beta1 = _norm_beta1(epoch, opt.dataset)
            loss_kd = beta1 * F.mse_loss(f_s, f_t.detach()) * opt.co_sponge
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_kd
        losses.update(loss.item(), images.size(0))

        # ===================Metrics=====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print info
        if idx % opt.print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                        'GPU {3}\t' \
                        'Time: {batch_time.avg:.3f}\t' \
                        'Loss {loss.avg:.4f}\t' \
                        'Acc@1 {top1.avg:.3f}\t' \
                        'Acc@5 {top5.avg:.3f}'.format(
                epoch, idx, n_batch, opt.gpu, loss=losses, top1=top1, top5=top5,
                batch_time=batch_time)
            print(print_str)
            sys.stdout.flush()

    return top1.avg, top5.avg, losses.avg


DIST_SNAPSHOT_EPOCHS   = {30, 150, 180, 240}
HARDEST_SNAPSHOT_EPOCHS = {30, 150, 180, 240}


CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm',
]


def _unnormalize_cifar(img_tensor):
    """Undo CIFAR-100 normalization for display. [C,H,W] float → [H,W,C] numpy in [0,1]."""
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
    std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)
    return (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def save_hardest_samples(epoch, hardest_top, hardest_bot, save_folder):
    """Save a 2x5 grid: top row = 5 highest L1 disagreement, bottom row = 5 lowest.

    Args:
        epoch:       Current epoch number
        hardest_top: min-heap of (d,  counter, img_cpu [C,H,W], label) — 5 highest d
        hardest_bot: min-heap of (-d, counter, img_cpu [C,H,W], label) — 5 lowest d
        save_folder: Directory where the PNG is saved
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Sort: highest d first for top, lowest d first for bot
    top_sorted = sorted(hardest_top, key=lambda x: -x[0])
    bot_sorted = sorted(hardest_bot, key=lambda x:  x[0])  # ascending -d = ascending d...
    # bot stored as (-d,...): sort by x[0] ascending = most negative first = largest d first
    # we want lowest d first, so sort descending by x[0] (least negative = smallest d first)
    bot_sorted = sorted(hardest_bot, key=lambda x: -x[0])

    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for col in range(5):
        for row, (samples, row_label, sign) in enumerate([
            (top_sorted, 'Hardest',  1),
            (bot_sorted, 'Easiest', -1),
        ]):
            ax = axes[row, col]
            if col >= len(samples):
                ax.axis('off')
                continue
            stored_d, _, img_tensor, label = samples[col]
            dis_val = stored_d * sign  # undo negation for easiest row
            img_np  = _unnormalize_cifar(img_tensor)
            ax.imshow(img_np)
            ax.set_title(f'{CIFAR100_CLASSES[label]}\nL1={dis_val:.3f}', fontsize=8)
            ax.axis('off')
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9)

    fig.suptitle(f'Hardest vs Easiest Samples (L1 disagreement) — Epoch {epoch}', fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(save_folder, f'hardest_samples_epoch{epoch}.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[SoftAKD] Saved hardest/easiest samples → {out_path}')


def train_distill_akd(anchor_set, anchor_net, epoch, train_loader, module_list,
                      criterion_list, optimizer, optimizer_anchor, opt, a_feat_t,
                      sigma=None, anchor_labels=None,
                      gcn=None, optimizer_gcn=None, snapshot_store=None,
                      scaler=None, dist_snapshot_store=None,
                      typicality_logger=None,
                      typ_t_precomputed=None):
    """One epoch of Anchor-based Knowledge Distillation (AKD / Soft AKD).

    AKD introduces learnable anchor images whose spatial attention is optimised
    jointly with the student.  Three KL-divergence terms compare
    teacher/student similarity distributions w.r.t. the batch and the anchors.

    Extra args compared to train_distill:
        anchor_set      – tensor of anchor images (n_anchors, C, H, W), on GPU
        anchor_net      – AnchorNet that applies learnable attention to anchors
        optimizer_anchor – separate SGD optimiser for anchor_net
        a_feat_t        – precomputed teacher pooled features for anchor images
        sigma           – (soft_akd only) class similarity matrix [C, C], on GPU
        anchor_labels   – (soft_akd only) class label per anchor [A], on GPU
        gcn             – (soft_akd + GCN) GCN module for softening sigma
        optimizer_gcn   – (soft_akd + GCN) Adam optimiser for GCN
    """
    from distiller_zoo.AKD import akd_loss
    from distiller_zoo.SoftAKD import soft_akd_loss
    import torch.nn.functional as F

    use_soft = (sigma is not None and anchor_labels is not None) or \
               getattr(opt, 'sigma_mode', 'precomputed') == 'student'
    use_gcn = gcn is not None and optimizer_gcn is not None
    is_rw = getattr(opt, 'lambda_mode', 'rw') in ('rw', 'cw')
    collect_tb  = use_soft and is_rw           # every epoch: AverageMeters for tensorboard
    collect_dis = collect_tb and (epoch % 5 == 0)  # every 5 epochs: full tensor lists for print
    collect_dist_snap  = (dist_snapshot_store is not None and epoch in DIST_SNAPSHOT_EPOCHS)
    collect_hardest    = use_soft and is_rw and epoch in HARDEST_SNAPSHOT_EPOCHS

    if collect_tb:
        dis_raw_meters = {k: AverageMeter() for k in ('L1', 'L2', 'L3')}
        dis_val_meters = {k: AverageMeter() for k in ('L1', 'L2', 'L3')}

    if collect_dis:
        dis_accum = {k: [] for k in ('raw_L1', 'raw_L2', 'raw_L3',
                                     'gate_L1', 'gate_L2', 'gate_L3')}

    if collect_dist_snap:
        dist_snap_accum = {'L1': [], 'L2': [], 'L3': []}

    if collect_hardest:
        hardest_top = []   # min-heap (d, counter, img, label)  — tracks 5 highest d
        hardest_bot = []   # min-heap (-d, counter, img, label) — tracks 5 lowest d
        _hctr = 0          # unique counter to break ties without comparing tensors

    if use_gcn:
        from distiller_zoo.SoftAKD import soften_sigma_with_gcn
        from distiller_zoo.SoftAKD2 import monitor_gcn

    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]

    model_s = module_list[0]
    model_t = module_list[-1]

    n_cls = {
        'cifar100': 100,
        'imagenet': 1000,
        'cifar100_oscr': 50,
    }.get(opt.dataset, 100)

    batch_time = AverageMeter()
    losses = AverageMeter()
    loss_G_meter = AverageMeter()
    loss_akd_meter = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    n_batch = len(train_loader)
    end = time.time()

    optimizer.zero_grad()
    optimizer_anchor.zero_grad()

    for idx, data in enumerate(train_loader):
        images, labels = data[0], data[1]

        if opt.gpu is not None:
            images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
        if torch.cuda.is_available():
            labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

        # =================== forward =====================
        # Anchor path: apply learnable attention and forward through student
        running_anchor_set = anchor_net(anchor_set)
        a_feat_s_list, _ = model_s(running_anchor_set, is_feat=True)
        a_feat_s = a_feat_s_list[-1]  # pooled anchor features from student

        # Batch path
        feat_s, logit_s = model_s(images, is_feat=True)
        with torch.no_grad():
            feat_t, logit_t = model_t(images, is_feat=True)
            feat_t = [f.detach() for f in feat_t]

        feat_student = feat_s[-1]   # student pooled features
        feat_teacher = feat_t[-1]   # teacher pooled features

        # Layer-normalise logits (part of AKD methodology)
        logit_s = F.layer_norm(logit_s, torch.Size((logit_s.size(1),)), None, None, 1e-7) * opt.ceta
        logit_t = F.layer_norm(logit_t, torch.Size((logit_t.size(1),)), None, None, 1e-7) * opt.ceta

        # Losses
        loss_cls = criterion_cls(logit_s, labels)
        if logit_s.size(1) != logit_t.size(1):
            # Teacher/student have different output dims (e.g. OSCR: 100 vs 50).
            # KL div is undefined; requires opt.div == 0.
            loss_div = torch.tensor(0.0, device=logit_s.device)
        else:
            loss_div = criterion_div(logit_s, logit_t)

        loss_G = None
        is_student_mode = getattr(opt, 'sigma_mode', 'precomputed') == 'student'
        if use_soft and use_gcn and not is_student_mode:
            # Soften sigma using GCN
            sigma_soft, loss_G = soften_sigma_with_gcn(
                sigma, feat_teacher, feat_student,
                a_feat_t, a_feat_s,
                labels, anchor_labels, gcn, alpha=opt.alpha_soft,
                sigma_s_mode=getattr(opt, 'sigma_s_mode', 'ab')
            )
            loss_akd, dis_stats = soft_akd_loss(feat_teacher.detach(), feat_student,
                                     a_feat_t.detach(), a_feat_s,
                                     labels, anchor_labels, sigma_soft,
                                     opt.lambda_soft, opt, return_stats=collect_tb,
                                     scaler=scaler,
                                     typicality_logger=typicality_logger,
                                     batch_idx=idx, epoch=epoch,
                                     typ_t_precomputed=typ_t_precomputed)
        elif use_soft:
            sigma_in = None if is_student_mode else sigma
            loss_akd, dis_stats = soft_akd_loss(feat_teacher.detach(), feat_student,
                                     a_feat_t.detach(), a_feat_s,
                                     labels, anchor_labels, sigma_in,
                                     opt.lambda_soft, opt, return_stats=collect_tb,
                                     scaler=scaler,
                                     typicality_logger=typicality_logger,
                                     batch_idx=idx, epoch=epoch,
                                     typ_t_precomputed=typ_t_precomputed)
        else:
            loss_akd = akd_loss(feat_teacher.detach(), feat_student,
                                a_feat_t.detach(), a_feat_s, opt)
            dis_stats = None

        if collect_tb and dis_stats is not None:
            for k in ('L1', 'L2', 'L3'):
                raw_t = dis_stats[f'raw_{k}']
                val_t = dis_stats[f'gate_{k}']
                dis_raw_meters[k].update(raw_t.mean().item(), raw_t.numel())
                dis_val_meters[k].update(val_t.mean().item(), val_t.numel())

        if collect_dis and dis_stats is not None:
            for k in dis_accum:
                dis_accum[k].append(dis_stats[k].cpu())

        if collect_dist_snap and dis_stats is not None:
            for k in ('L1', 'L2', 'L3'):
                dist_snap_accum[k].append(dis_stats[f'raw_{k}'].cpu())

        if collect_hardest and dis_stats is not None:
            batch_dis  = dis_stats['raw_L1'].cpu()   # [B]
            batch_imgs = images.cpu()                 # [B, C, H, W]
            batch_lbls = labels.cpu()                 # [B]
            for i in range(batch_dis.size(0)):
                d   = batch_dis[i].item()
                img = batch_imgs[i]
                lbl = batch_lbls[i].item()
                _hctr += 1
                # top-5 highest: min-heap, replace smallest when new d is larger
                if len(hardest_top) < 5:
                    heapq.heappush(hardest_top, (d, _hctr, img, lbl))
                elif d > hardest_top[0][0]:
                    heapq.heapreplace(hardest_top, (d, _hctr, img, lbl))
                # bottom-5 lowest: max-heap via negation, replace largest when new d is smaller
                if len(hardest_bot) < 5:
                    heapq.heappush(hardest_bot, (-d, _hctr, img, lbl))
                elif -d > hardest_bot[0][0]:
                    heapq.heapreplace(hardest_bot, (-d, _hctr, img, lbl))

        loss = opt.cls * loss_cls + opt.div * loss_div + opt.beta * loss_akd
        losses.update(loss.item(), images.size(0))
        loss_akd_meter.update(loss_akd.item(), images.size(0))
        if loss_G is not None:
            loss_G_meter.update(loss_G.item(), images.size(0))

        # =================== Metrics =====================
        metrics = accuracy(logit_s, labels, topk=(1, 5))
        top1.update(metrics[0].item(), images.size(0))
        top5.update(metrics[1].item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # =================== backward =====================
        t_bwd0 = time.time()
        # Student + anchor loss
        optimizer.zero_grad()
        optimizer_anchor.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_anchor.step()
        t_bwd1 = time.time()

        # GCN loss (separate optimization)
        if loss_G is not None:
            optimizer_gcn.zero_grad()
            loss_G.backward()
            optimizer_gcn.step()
        t_bwd2 = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'GPU {3}\t'
                  'Time: {batch_time.avg:.3f}\t'
                  'Loss {loss.avg:.4f}\t'
                  'Acc@1 {top1.avg:.3f}\t'
                  'Acc@5 {top5.avg:.3f}'.format(
                   epoch, idx, n_batch, opt.gpu, batch_time=batch_time,
                   loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    if scaler is not None:
        scaler.step_epoch(epoch)

    # Hardest samples snapshot
    if collect_hardest and hardest_top:
        save_hardest_samples(epoch, hardest_top, hardest_bot, opt.save_folder)

    # Distance distribution snapshot
    if collect_dist_snap and any(len(v) > 0 for v in dist_snap_accum.values()):
        dist_snapshot_store[epoch] = {k: torch.cat(dist_snap_accum[k]).numpy()
                                      for k in ('L1', 'L2', 'L3')}

    # GCN monitoring at end of epoch
    if use_gcn:
        monitor_gcn(gcn, epoch, n_cls, sigma=sigma,
                    alpha=opt.alpha_soft, snapshot_store=snapshot_store)

    # Disagreement stats every 5 epochs
    if collect_dis and any(len(v) > 0 for v in dis_accum.values()):
        from distiller_zoo.SoftAKD2 import print_disagreement_stats
        print_disagreement_stats(dis_accum, epoch, opt, scaler=scaler)

    # Typicality epoch summary
    if typicality_logger is not None:
        typicality_logger.end_epoch(epoch)

    # Build tensorboard stats dict (every epoch)
    dis_tb_stats = None
    if collect_tb:
        lambda_soft = getattr(opt, 'lambda_soft', 0.3)
        dis_tb_stats = {
            'dis_raw_L1': dis_raw_meters['L1'].avg,
            'dis_raw_L2': dis_raw_meters['L2'].avg,
            'dis_raw_L3': dis_raw_meters['L3'].avg,
        }
        for k in ('L1', 'L2', 'L3'):
            dis_tb_stats[f'avg_gate_{k}'] = dis_val_meters[k].avg
            if scaler is not None:
                dis_tb_stats[f'd_mean_{k}'] = scaler.d_mean[k]
                dis_tb_stats[f'd_std_{k}']  = scaler.d_std[k]

    return top1.avg, top5.avg, losses.avg, loss_G_meter.avg, loss_akd_meter.avg, dis_tb_stats


def validate_vanilla(val_loader, model, criterion, opt):
    """validation"""

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            images, labels = batch_data

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
    
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg


def validate_distill(val_loader, module_list, criterion, opt):
    """validation"""
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    # switch to evaluate mode
    for module in module_list:
        module.eval()
    
    model_s = module_list[0]
    model_t = module_list[-1]
    n_batch = len(val_loader)

    with torch.no_grad():
        end = time.time()
        for idx, batch_data in enumerate(val_loader):
            
            images, labels = batch_data

            if opt.gpu is not None:
                images = images.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)
            if torch.cuda.is_available():
                labels = labels.cuda(opt.gpu if opt.multiprocessing_distributed else 0, non_blocking=True)

            # compute output
            output = model_s(images)
            if opt.distill == 'norm':
                n_cls = output.size(1)
                output = F.layer_norm(output, (n_cls,), eps=1e-7) * opt.ceta
            loss = criterion(output, labels)
            losses.update(loss.item(), images.size(0))

            # ===================Metrics=====================
            metrics = accuracy(output, labels, topk=(1, 5))
            top1.update(metrics[0].item(), images.size(0))
            top5.update(metrics[1].item(), images.size(0))
            batch_time.update(time.time() - end)
            
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'GPU: {2}\t'
                      'Time: {batch_time.avg:.3f}\t'
                      'Loss {loss.avg:.4f}\t'
                      'Acc@1 {top1.avg:.3f}\t'
                      'Acc@5 {top5.avg:.3f}'.format(
                       idx, n_batch, opt.gpu, batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))
                
    if opt.multiprocessing_distributed:
        # Batch size may not be equal across multiple gpus
        total_metrics = torch.tensor([top1.sum, top5.sum, losses.sum]).to(opt.gpu)
        count_metrics = torch.tensor([top1.count, top5.count, losses.count]).to(opt.gpu)
        total_metrics = reduce_tensor(total_metrics, 1) # here world_size=1, because they should be summed up
        count_metrics = reduce_tensor(count_metrics, 1)
        ret = []
        for s, n in zip(total_metrics.tolist(), count_metrics.tolist()):
            ret.append(s / (1.0 * n))
        return ret

    return top1.avg, top5.avg, losses.avg
