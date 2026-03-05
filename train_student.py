"""
the general training framework
"""

from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import re
import argparse
import time

import random
import numpy
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboard_logger as tb_logger

from models import model_dict
from models.util import ConvReg, SelfA, SRRL

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample
from dataset.imagenet import get_imagenet_dataloader,  get_dataloader_sample

from helper.loops import train_distill as train, train_distill_akd, validate_vanilla, validate_distill
from helper.util import save_dict_to_json, reduce_tensor, adjust_learning_rate

from crd.criterion import CRDLoss
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, VIDLoss, SemCKDLoss, PKT, SoftPKT2, AnchorNet, calculate_anchor_set, GCN

split_symbol = '~' if os.name == 'nt' else ':'

def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    # basic
    parser.add_argument('--print_freq', type=int, default=200, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--gpu_id', type=str, default='0', help='id(s) for CUDA_VISIBLE_DEVICES')
    # parser.add_argument('--num_seed', type=int, default=0, help='number of training seed')


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet'], help='dataset')
    parser.add_argument('--model_s', type=str, default='resnet8x4')
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'hint', 'attention', 'similarity', 'vid',
                                                                      'crd', 'semckd','srrl','pkt', 'soft_pkt2', 'akd', 'soft_akd'])
    parser.add_argument('-c', '--cls', type=float, default=1.0, help='weight for classification')
    parser.add_argument('-d', '--div', type=float, default=1.0, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=0.0, help='weight balance for other losses')
    parser.add_argument('-s', '--soft', type=float, default=1.0, help='attention scale of SemCKD')

    # soft PKT v2
    parser.add_argument('--lambda_soft', type=float, default=0.7, help='blending weight for soft PKT v2 (lambda)')
    parser.add_argument('--sigma_temp', type=float, default=1.0, help='temperature for sharpening sigma (< 1.0 sharpens)')
    parser.add_argument('--sigma_path', type=str, default='save/class_similarity_matrix.npy',
                        help='path to class similarity matrix for soft PKT v2')

    # AKD (Anchor-based Knowledge Distillation)
    parser.add_argument('--l_1', type=float, default=1.0, help='AKD: weight for intra-batch KL (L_1)')
    parser.add_argument('--l_2', type=float, default=0.1, help='AKD: balance between L_2 and L_3 anchor KL terms')
    parser.add_argument('--ceta', type=float, default=2.5, help='AKD: logit scaling factor after layer norm')

    # GCN (learnable confusion matrix for Soft AKD)
    parser.add_argument('--alpha_soft', type=float, default=0.1, help='Soft AKD: GCN blending weight (0 = original sigma, 1 = full GCN)')
    parser.add_argument('--gcn_lr', type=float, default=1e-4, help='Soft AKD: learning rate for GCN')
    parser.add_argument('--sigma_s_mode', type=str, default='ab', choices=['ab', 'aa', 'bb'],
                        help='Soft AKD: how to compute Sigma_s for GCN training. '
                             'ab=batch-to-anchor (default), aa=anchor-to-anchor, bb=batch-to-batch')
    parser.add_argument('--lambda_k', type=float, default=10.0,
                        help='Soft AKD adaptive lambda: sigmoid steepness (default 10.0)')
    parser.add_argument('--lambda_d0', type=float, default=0.3,
                        help='Soft AKD adaptive lambda: sigmoid midpoint on normalised disagreement (default 0.5)')

    # hint layer
    parser.add_argument('--hint_layer', default=1, type=int, choices=[0, 1, 2, 3, 4])

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # multiprocessing
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23451', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--deterministic', action='store_true', default=True, help='Make results reproducible')
    parser.add_argument('--skip-validation', action='store_true', help='Skip validation of teacher')

    # checkpoint and resume
    parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from')
    parser.add_argument('--checkpoint_freq', type=int, default=5, help='save checkpoint every N epochs')
    
    opt = parser.parse_args()

    # set different learning rates for these MobileNet/ShuffleNet models
    if opt.model_s in ['MobileNetV2', 'MobileNetV2_1_0', 'ShuffleV1', 'ShuffleV2', 'ShuffleV2_1_5']:
        opt.learning_rate = 0.01

    # set the path of model and tensorboard
    opt.model_path = './save/students/models'
    opt.tb_path = './save/students/tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    model_name_template = split_symbol.join(['S', '{}_T', '{}_{}_{}_r', '{}_a', '{}_b', '{}_{}'])
    opt.model_name = model_name_template.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                opt.cls, opt.div, opt.beta, opt.trial)
    if opt.distill == 'soft_akd':
        opt.model_name += '_l_{}_a_{}_glr_{}'.format(opt.lambda_soft, opt.alpha_soft, opt.gcn_lr)
        if opt.sigma_temp != 1.0:
            opt.model_name += '_t_{}'.format(opt.sigma_temp)
        if opt.sigma_s_mode != 'ab':
            opt.model_name += '_sm_{}'.format(opt.sigma_s_mode)
        if opt.lambda_k != 10.0:
            opt.model_name += '_lk_{}'.format(opt.lambda_k)
        if opt.lambda_d0 != 0.3:
            opt.model_name += '_ld_{}'.format(opt.lambda_d0)
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    return opt

def get_teacher_name(model_path):
    """parse teacher name"""
    directory = model_path.split('/')[-2]
    pattern = ''.join(['S', split_symbol, '(.+)', '_T', split_symbol])
    name_match = re.match(pattern, directory)
    if name_match:
        return name_match[1]
    segments = directory.split('_')
    if segments[0] == 'wrn':
        return segments[0] + '_' + segments[1] + '_' + segments[2]
    return segments[0]


def load_teacher(model_path, n_cls, gpu=None, opt=None):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    map_location = None if gpu is None else {'cuda:0': 'cuda:%d' % (gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location, weights_only=False)['model'])
    print('==> done')
    return model


best_acc = 0
total_time = time.time()

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    """Save training checkpoint"""
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f'==> Checkpoint saved to {filepath}')

def move_optimizer_to_device(optimizer, device):
    """Move optimizer state to specified device"""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

def load_checkpoint(checkpoint_path, model_s, optimizer, trainable_list, opt):
    """Load checkpoint and restore training state"""
    print(f'==> Loading checkpoint from {checkpoint_path}')

    map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']

    # Load model state
    model_s.load_state_dict(checkpoint['model'])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Load additional trainable modules if they exist
    if 'trainable_states' in checkpoint and len(trainable_list) > 1:
        for i, state_dict in enumerate(checkpoint['trainable_states']):
            if i < len(trainable_list) and state_dict is not None:
                trainable_list[i].load_state_dict(state_dict)

    print(f'==> Checkpoint loaded successfully')
    print(f'==> Resuming from epoch {start_epoch}, best accuracy: {best_acc:.2f}%')

    return start_epoch, best_acc

def main():
    
    opt = parse_option()
    
    # ASSIGN CUDA_ID
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    
    ngpus_per_node = torch.cuda.device_count()
    opt.ngpus_per_node = ngpus_per_node
    if opt.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        world_size = 1
        opt.world_size = ngpus_per_node * world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        main_worker(None if ngpus_per_node > 1 else opt.gpu_id, ngpus_per_node, opt)

def main_worker(gpu, ngpus_per_node, opt):
    global best_acc, total_time
    opt.gpu = int(gpu)
    opt.gpu_id = int(gpu)

    if opt.gpu is not None:
        print("Use GPU: {} for training".format(opt.gpu))

    if opt.multiprocessing_distributed:
        # Only one node now.
        opt.rank = gpu
        dist_backend = 'nccl'
        dist.init_process_group(backend=dist_backend, init_method=opt.dist_url,
                                world_size=opt.world_size, rank=opt.rank)
        opt.batch_size = int(opt.batch_size / ngpus_per_node)
        opt.num_workers = int((opt.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if opt.deterministic:
        torch.manual_seed(int(opt.trial))
        torch.cuda.manual_seed_all(int(opt.trial))
        cudnn.deterministic = True
        cudnn.benchmark = False
        numpy.random.seed(int(opt.trial))
        random.seed(int(opt.trial))

    # model
    n_cls = {
        'cifar100': 100,
        'imagenet': 1000,
    }.get(opt.dataset, None)
    
    model_t = load_teacher(opt.path_t, n_cls, opt.gpu, opt)
    try:
        model_s = model_dict[opt.model_s](num_classes=n_cls)
    except KeyError:
        print("This model is not supported.")

    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)

    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        regress_s = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        if opt.dataset == 'cifar100':
            opt.n_data = 50000
        else:
            opt.n_data = 1281167
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif opt.distill == 'semckd':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = SemCKDLoss()
        self_attention = SelfA(opt.batch_size, s_n, t_n, opt.soft)    
        module_list.append(self_attention)
        trainable_list.append(self_attention)
    elif opt.distill == 'srrl':
        s_n = feat_s[-1].shape[1]
        t_n = feat_t[-1].shape[1]
        model_fmsr = SRRL(s_n= s_n, t_n=t_n)
        criterion_kd = nn.MSELoss()
        module_list.append(model_fmsr)
        trainable_list.append(model_fmsr)
    elif opt.distill == 'pkt':
        from distiller_zoo import PKT
        criterion_kd = PKT()
    elif opt.distill == 'soft_pkt2':
        criterion_kd = SoftPKT2(sigma_path=opt.sigma_path, lambda_soft=opt.lambda_soft)
    elif opt.distill in ['akd', 'soft_akd']:
        # AKD/SoftAKD loss is computed directly in the training loop
        criterion_kd = nn.MSELoss()
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    module_list.append(model_t)

    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # Resume from checkpoint if specified
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            start_epoch, best_acc = load_checkpoint(opt.resume, model_s, optimizer, trainable_list, opt)
        else:
            print(f"==> No checkpoint found at '{opt.resume}', starting from scratch")

    if torch.cuda.is_available():
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if opt.multiprocessing_distributed:
            if opt.gpu is not None:
                torch.cuda.set_device(opt.gpu)
                module_list.cuda(opt.gpu)
                distributed_modules = []
                for module in module_list:
                    DDP = torch.nn.parallel.DistributedDataParallel
                    distributed_modules.append(DDP(module, device_ids=[opt.gpu]))
                module_list = distributed_modules
                criterion_list.cuda(opt.gpu)
            else:
                print('multiprocessing_distributed must be with a specifiec gpu id')
        else:
            criterion_list.cuda()
            module_list.cuda()
        if not opt.deterministic:
            cudnn.benchmark = True

        # Move optimizer state to GPU if resuming from checkpoint
        if opt.resume and os.path.isfile(opt.resume):
            device = torch.device(f'cuda:{opt.gpu}' if opt.multiprocessing_distributed else 'cuda')
            move_optimizer_to_device(optimizer, device)
            print(f'==> Optimizer state moved to {device}')

    # ==================== GPU VERIFICATION ====================
    print("\n" + "="*60)
    print("GPU CONFIGURATION CHECK")
    print("="*60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU ID: {opt.gpu}")
        print(f"GPU Name: {torch.cuda.get_device_name(opt.gpu if opt.multiprocessing_distributed else 0)}")
        # Check if models are actually on GPU
        student_device = next(module_list[0].parameters()).device
        print(f"Student Model Device: {student_device}")
        print(f"Models successfully loaded on GPU: {student_device.type == 'cuda'}")
    else:
        print("WARNING: CUDA is NOT available! Training will use CPU (very slow)")
    print("="*60 + "\n")
    # ==========================================================

    # dataloader
    if opt.dataset == 'cifar100':
        if opt.distill in ['crd']:
            train_loader, val_loader, n_data = get_cifar100_dataloaders_sample(batch_size=opt.batch_size,
                                                                               num_workers=opt.num_workers,
                                                                               k=opt.nce_k,
                                                                               mode=opt.mode)
        else:
            train_loader, val_loader = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers)
    elif opt.dataset == 'imagenet':
        if opt.dali is None:
            if opt.distill in ['crd']:
                train_loader, val_loader, n_data, _, train_sampler = get_dataloader_sample(dataset=opt.dataset, batch_size=opt.batch_size, 
                                                                                num_workers=opt.num_workers, 
                                                                                is_sample=True, 
                                                                                k=opt.nce_k,
                                                                                multiprocessing_distributed=opt.multiprocessing_distributed)
            else:
                train_loader, val_loader, train_sampler = get_imagenet_dataloader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        multiprocessing_distributed=opt.multiprocessing_distributed)
        
    else:
        raise NotImplementedError(opt.dataset)

    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    if not opt.skip_validation:
        # validate teacher accuracy
        teacher_acc, _, _ = validate_vanilla(val_loader, model_t, criterion_cls, opt)

        

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print('teacher accuracy: ', teacher_acc)
    else:
        print('Skipping teacher validation.')

    # ==================== AKD / Soft AKD setup ====================
    anchor_set = anchor_net = optimizer_anchor = a_feat_t = None
    sigma = anchor_labels = None
    gcn = optimizer_gcn = None
    if opt.distill in ['akd', 'soft_akd']:
        img_dim = 32 if opt.dataset == 'cifar100' else 224
        print('==> Computing anchor set for AKD ...')
        anchor_set = calculate_anchor_set(
            model_t, train_loader,
            subset_length="max",
            num_classes=n_cls,
            anchors_per_class=1
        ).cuda(opt.gpu if opt.multiprocessing_distributed else 0)
        print(f'    Anchor set: {anchor_set.shape[0]} anchors')

        anchor_net = AnchorNet(
            anchor_batch_size=anchor_set.size(0),
            DIM=img_dim,
            weight_size=14
        ).cuda(opt.gpu if opt.multiprocessing_distributed else 0)
        anchor_net.train()

        optimizer_anchor = optim.SGD(
            anchor_net.parameters(), lr=1e-3, weight_decay=1e-5
        )

        # Precompute teacher features for anchors (fixed throughout training)
        model_t.eval()
        with torch.no_grad():
            a_feat_list, _ = model_t(anchor_set, is_feat=True)
            a_feat_t = a_feat_list[-1].detach()
        print('    Teacher anchor features precomputed.')

        # Soft AKD: load class similarity matrix and create anchor labels
        if opt.distill == 'soft_akd':
            import numpy as np
            if not os.path.exists(opt.sigma_path):
                raise FileNotFoundError(
                    f"Class similarity matrix not found at {opt.sigma_path}. "
                    f"Please run compute_class_similarity_matrix.py first."
                )
            sigma_np = np.load(opt.sigma_path)
            gpu_device = opt.gpu if opt.multiprocessing_distributed else 0
            sigma = torch.from_numpy(sigma_np).float().cuda(gpu_device)
            # anchors_per_class=1, selected in class order → anchor i = class i
            anchor_labels = torch.arange(n_cls).cuda(gpu_device)
            gcn = GCN(num_classes=n_cls).cuda(gpu_device)
            optimizer_gcn = torch.optim.Adam(gcn.parameters(), lr=opt.gcn_lr)
            print(f'    Soft AKD: sigma loaded from {opt.sigma_path}')
            print(f'    Soft AKD: lambda_soft = {opt.lambda_soft}')
            print(f'    Soft AKD: sigma_temp = {opt.sigma_temp}')
            print(f'    Soft AKD: GCN initialised ({n_cls}x{n_cls})')
            print(f'    Soft AKD: GCN lr = {opt.gcn_lr}, alpha_soft = {opt.alpha_soft}')

    # GCN diagnostic collectors (only used when distill == 'soft_akd' with GCN)
    gcn_snapshot_store = {} if (opt.distill == 'soft_akd' and gcn is not None) else None
    gcn_loss_G_history = []
    gcn_loss_akd_history = []

    # routine
    for epoch in range(start_epoch + 1, opt.epochs + 1):
        torch.cuda.empty_cache()
        if opt.multiprocessing_distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(epoch, opt, optimizer)
        if optimizer_gcn is not None:
            steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
            gcn_lr = opt.gcn_lr * (opt.lr_decay_rate ** steps) if steps > 0 else opt.gcn_lr
            for param_group in optimizer_gcn.param_groups:
                param_group['lr'] = gcn_lr
        print("==> training...")

        time1 = time.time()
        if opt.distill in ['akd', 'soft_akd']:
            train_acc, train_acc_top5, train_loss, epoch_loss_G, epoch_loss_akd = train_distill_akd(
                anchor_set, anchor_net, epoch, train_loader,
                module_list, criterion_list, optimizer, optimizer_anchor, opt, a_feat_t,
                sigma=sigma, anchor_labels=anchor_labels,
                gcn=gcn, optimizer_gcn=optimizer_gcn,
                snapshot_store=gcn_snapshot_store
            )
            if gcn is not None:
                gcn_loss_G_history.append(epoch_loss_G)
                gcn_loss_akd_history.append(epoch_loss_akd)
        else:
            train_acc, train_acc_top5, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()

        if opt.multiprocessing_distributed:
            metrics = torch.tensor([train_acc, train_acc_top5, train_loss]).cuda(opt.gpu, non_blocking=True)
            reduced = reduce_tensor(metrics, opt.world_size if 'world_size' in opt else 1)
            train_acc, train_acc_top5, train_loss = reduced.tolist()

        if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
            print(' * Epoch {}, GPU {}, Acc@1 {:.3f}, Acc@5 {:.3f}, Time {:.2f}'.format(epoch, opt.gpu, train_acc, train_acc_top5, time2 - time1))

            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('train_loss', train_loss, epoch)

        # Validate every 3 epochs or on the last epoch
        if epoch % 3 == 0 or epoch == opt.epochs:
            print('GPU %d validating' % (opt.gpu))
            test_acc, test_acc_top5, test_loss = validate_distill(val_loader, module_list, criterion_cls, opt)



            if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
                print(' ** Acc@1 {:.3f}, Acc@5 {:.3f}'.format(test_acc, test_acc_top5))

                logger.log_value('test_acc', test_acc, epoch)
                logger.log_value('test_loss', test_loss, epoch)
                logger.log_value('test_acc_top5', test_acc_top5, epoch)

                # save the best model
                if test_acc > best_acc:
                    best_acc = test_acc
                    state = {
                        'epoch': epoch,
                        'model': model_s.state_dict(),
                        'best_acc': best_acc,
                    }
                    save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))

                    test_merics = {'test_loss': test_loss,
                                    'test_acc': test_acc,
                                    'test_acc_top5': test_acc_top5,
                                    'epoch': epoch}

                    save_dict_to_json(test_merics, os.path.join(opt.save_folder, "test_best_metrics.json"))
                    print('saving the best model!')
                    torch.save(state, save_file)

                # Save checkpoint every N epochs
                if epoch % opt.checkpoint_freq == 0:
                    # Collect all trainable module states
                    trainable_states = [module.state_dict() for module in trainable_list]

                    checkpoint_state = {
                        'epoch': epoch,
                        'model': model_s.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'trainable_states': trainable_states,
                    }
                    checkpoint_filename = f'checkpoint_epoch_{epoch}.pth'
                    save_checkpoint(checkpoint_state, opt.save_folder, checkpoint_filename)
        else:
            print(f'GPU {opt.gpu} skipping validation (epoch {epoch})')
            
    if not opt.multiprocessing_distributed or opt.rank % ngpus_per_node == 0:
        # This best accuracy is only for printing purpose.
        print('best accuracy:', best_acc)

        # Save GCN diagnostic plots
        if gcn_snapshot_store is not None:
            from distiller_zoo.SoftAKD2 import save_gcn_plots
            save_gcn_plots(gcn_snapshot_store, gcn_loss_G_history, gcn_loss_akd_history,
                           opt.save_folder, opt.alpha_soft)
        
        # save parameters
        save_state = {k: v for k, v in opt._get_kwargs()}
        # No. parameters(M)
        num_params = (sum(p.numel() for p in model_s.parameters())/1000000.0)
        save_state['Total params'] = num_params
        save_state['Total time'] =  (time.time() - total_time)/3600.0
        params_json_path = os.path.join(opt.save_folder, "parameters.json") 
        save_dict_to_json(save_state, params_json_path)

if __name__ == '__main__':
    main()
