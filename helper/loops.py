from __future__ import print_function, division
from cProfile import label

import sys
import time
import torch
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
        elif opt.distill == 'soft_pkt2':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, labels)
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


def train_distill_akd(anchor_set, anchor_net, epoch, train_loader, module_list,
                      criterion_list, optimizer, optimizer_anchor, opt, a_feat_t,
                      sigma=None, anchor_labels=None,
                      gcn=None, optimizer_gcn=None, snapshot_store=None,
                      scaler=None):
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

    use_soft = sigma is not None and anchor_labels is not None
    use_gcn = gcn is not None and optimizer_gcn is not None
    is_rw = getattr(opt, 'lambda_mode', 'rw') == 'rw'
    collect_tb  = use_soft and is_rw           # every epoch: AverageMeters for tensorboard
    collect_dis = collect_tb and (epoch % 5 == 0)  # every 5 epochs: full tensor lists for print

    # Jump lambda_alpha to 1.5 at lock_epoch (power path only)
    if scaler is not None:
        opt.lambda_alpha = 1.5 if epoch >= scaler.lock_epoch else 1.0

    if collect_tb:
        dis_raw_meters = {k: AverageMeter() for k in ('L1', 'L2', 'L3')}
        dis_val_meters = {k: AverageMeter() for k in ('L1', 'L2', 'L3')}

    is_power = getattr(opt, 'lambda_fn', 'sigmoid') == 'power'
    if collect_dis:
        if is_power:
            dis_accum = {k: [] for k in ('raw_L1', 'raw_L2', 'raw_L3',
                                         'lam_L1', 'lam_L2', 'lam_L3')}
        else:
            dis_accum = {k: [] for k in ('raw_L1', 'raw_L2', 'raw_L3',
                                         'gate_L1', 'gate_L2', 'gate_L3')}

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
        logit_s = F.layer_norm(logit_s, torch.Size((n_cls,)), None, None, 1e-7) * opt.ceta
        logit_t = F.layer_norm(logit_t, torch.Size((n_cls,)), None, None, 1e-7) * opt.ceta

        # Losses
        loss_cls = criterion_cls(logit_s, labels)
        loss_div = criterion_div(logit_s, logit_t)

        loss_G = None
        if use_soft and use_gcn:
            # Soften sigma using GCN
            sigma_soft, loss_G = soften_sigma_with_gcn(
                sigma, feat_teacher, feat_student,
                a_feat_t, a_feat_s,
                labels, anchor_labels, gcn, alpha=opt.alpha_soft,
                sigma_s_mode=getattr(opt, 'sigma_s_mode', 'ab')
            )
            # Use softened sigma in existing soft_akd_loss
            loss_akd, dis_stats = soft_akd_loss(feat_teacher.detach(), feat_student,
                                     a_feat_t.detach(), a_feat_s,
                                     labels, anchor_labels, sigma_soft,
                                     opt.lambda_soft, opt, return_stats=collect_tb,
                                     scaler=scaler)
        elif use_soft:
            loss_akd, dis_stats = soft_akd_loss(feat_teacher.detach(), feat_student,
                                     a_feat_t.detach(), a_feat_s,
                                     labels, anchor_labels, sigma,
                                     opt.lambda_soft, opt, return_stats=collect_tb,
                                     scaler=scaler)
        else:
            loss_akd = akd_loss(feat_teacher.detach(), feat_student,
                                a_feat_t.detach(), a_feat_s, opt)
            dis_stats = None

        if collect_tb and dis_stats is not None:
            val_prefix = 'lam' if scaler is not None else 'gate'
            for k in ('L1', 'L2', 'L3'):
                raw_t = dis_stats[f'raw_{k}']
                val_t = dis_stats[f'{val_prefix}_{k}']
                dis_raw_meters[k].update(raw_t.mean().item(), raw_t.numel())
                dis_val_meters[k].update(val_t.mean().item(), val_t.numel())

        if collect_dis and dis_stats is not None:
            for k in dis_accum:
                dis_accum[k].append(dis_stats[k].cpu())

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

    # Power scaler: apply per-epoch EMA update
    if scaler is not None:
        scaler.step_epoch(epoch)

    # GCN monitoring at end of epoch
    if use_gcn:
        monitor_gcn(gcn, epoch, n_cls, sigma=sigma,
                    alpha=opt.alpha_soft, snapshot_store=snapshot_store)

    # Disagreement stats every 5 epochs (rw mode only)
    if collect_dis and any(len(v) > 0 for v in dis_accum.values()):
        if is_power:
            from distiller_zoo.SoftAKD2 import print_disagreement_stats_power
            print_disagreement_stats_power(dis_accum, epoch, opt, scaler.d_max)
        else:
            from distiller_zoo.SoftAKD2 import print_disagreement_stats
            print_disagreement_stats(dis_accum, epoch, opt, scaler=scaler)

    # Build tensorboard stats dict (every epoch)
    dis_tb_stats = None
    if collect_tb:
        lambda_soft = getattr(opt, 'lambda_soft', 0.3)
        dis_tb_stats = {
            'dis_raw_L1': dis_raw_meters['L1'].avg,
            'dis_raw_L2': dis_raw_meters['L2'].avg,
            'dis_raw_L3': dis_raw_meters['L3'].avg,
        }
        if is_power:
            for k in ('L1', 'L2', 'L3'):
                dis_tb_stats[f'lam_pct_{k}'] = 100.0 * dis_val_meters[k].avg / lambda_soft
                dis_tb_stats[f'run_dmax_{k}'] = scaler.d_max[k]
        else:
            for k in ('L1', 'L2', 'L3'):
                dis_tb_stats[f'avg_gate_{k}'] = dis_val_meters[k].avg
                dis_tb_stats[f'd_mean_{k}'] = scaler.d_mean[k]
                dis_tb_stats[f'd_std_{k}'] = scaler.d_std[k]

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
