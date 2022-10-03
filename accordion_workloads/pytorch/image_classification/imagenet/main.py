import argparse
import datetime
import math
import os
import random
import shutil
import socket
import sys
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# TODO: Figure out a cleaner way of including gavel_iterator.
imagenet_dir = os.path.dirname(os.path.realpath(__file__))
image_classification_dir = os.path.dirname(imagenet_dir)
pytorch_dir = os.path.dirname(image_classification_dir)
workloads_dir = os.path.dirname(pytorch_dir)
gpusched_dir = os.path.dirname(workloads_dir)
scheduler_dir = os.path.join(gpusched_dir, 'scheduler')
sys.path.append(scheduler_dir)
from gavel_iterator import GavelIterator

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--num_minibatches', default=None, type=int,
                    help='number of minibatches to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint_dir',
                    type=str,
                    default='/lfs/1/keshav2/checkpoints/resnet-50',
                    help='Directory for checkpoints')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='Distributed backend')
parser.add_argument('--local_rank', default=0, type=int,
                    help='Local rank')
parser.add_argument('--rank', default=None, type=int,
                    help='Rank')
parser.add_argument('--world_size', default=None, type=int,
                    help='World size')
parser.add_argument('--master_addr', default=None, type=str,
                    help='Master address to use for distributed run')
parser.add_argument('--master_port', default=None, type=int,
                    help='Master port to use for distributed run')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--throughput_estimation_interval', type=int, default=None,
                    help='Steps between logging steps completed')
parser.add_argument('--max_duration', type=int, default=None,
                    help='Maximum duration in seconds')
parser.add_argument('--enable_gavel_iterator', action='store_true',
                    default=False, help='If set, use Gavel iterator')

best_acc1 = 0
total_minibatches = 0
total_elapsed_time = 0

def main():
    global args, best_acc1, total_minibatches, total_elapsed_time
    args = parser.parse_args()
    if args.batch_size > 128:
        args.batch_size = 128
    elif args.batch_size < 16:
        args.batch_size = 16
    print("batch size at start of model is ", args.batch_size)
    torch.cuda.set_device(args.local_rank)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = False
    if args.master_addr is not None:
        args.distributed = True
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.master_port)
        os.environ['NCCL_DEBUG'] = 'INFO'

        # hostname = socket.gethostname()
        # if "node" in hostname: # wisr cluster
        #     socket_ifname = 'enps0f0'# if hostname != 'node1' else 'enp94s0f0'  # rdma nics
        #     # socket_ifname = 'eno4'
        # else:  # ec2
        #     socket_ifname = "ens3"
        # os.environ['NCCL_SOCKET_IFNAME'] = socket_ifname
        
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    model = model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.local_rank],
                output_device=args.local_rank)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    if args.enable_gavel_iterator:
        train_loader = GavelIterator(train_loader, args.checkpoint_dir,
                                     load_checkpoint, save_checkpoint)

    grad_calc_dict = dict()
    original_batch_size = None

    # Load from checkpoint.
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.chkpt')
    if os.path.exists(checkpoint_path):
        if args.enable_gavel_iterator:
            checkpoint = train_loader.load_checkpoint(args, checkpoint_path)
        else:
            checkpoint = load_checkpoint(args, checkpoint_path)

        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            original_batch_size = checkpoint['original_batch_size']
            grad_calc_dict = checkpoint['grad_calc_dict']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(checkpoint_path, checkpoint['epoch']))
    else:
        original_batch_size = args.batch_size

    if args.num_minibatches is not None:
        args.epochs = math.ceil(float(args.num_minibatches) *
                                args.batch_size / len(train_loader))
    
    epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch)

        full_rank_accum = [torch.zeros_like(copy_l) for copy_l in model.parameters()]

        # train for one epoch
        num_minibatches, elapsed_time, finished_epoch, full_rank_accum, cumulative_steps = \
            train(train_loader, model, criterion, optimizer,
                  epoch, total_minibatches,
                  full_rank_accum,
                  max_minibatches=args.num_minibatches,
                  total_elapsed_time=total_elapsed_time,
                  max_duration=args.max_duration)
        grad_calc_dict[epoch] = [torch.norm(pval).item() for pval in full_rank_accum]
        # print(f"[Epoch {epoch}] Gradient norm of all conv2d layers: {grad_calc_dict[epoch]}")
        print(f"[Epoch {epoch}] Sum of the gradient norms: {sum(grad_calc_dict[epoch])}")

        total_minibatches += num_minibatches
        total_elapsed_time += elapsed_time

        # check for critical regime
        # out_of_critical_regime, status_changed = check_critical_regime(grad_calc_dict, epoch, args, original_batch_size)
        out_of_critical_regime, status_changed = check_critical_regime_hardcode(epoch, args.epochs, args.batch_size, original_batch_size)

        if args.enable_gavel_iterator:
            # # FIXME: debug
            # if epoch == 0:
            #     train_loader.update_resource_requirement(big_bs=True, small_bs=False)
            if status_changed:
                if out_of_critical_regime and args.batch_size != 128:
                    train_loader.update_resource_requirement(big_bs=True, small_bs=False)
                if not out_of_critical_regime and args.batch_size != 16:
                    train_loader.update_resource_requirement(big_bs=False, small_bs=True)

        if args.enable_gavel_iterator and train_loader.done:
            break
        elif (args.num_minibatches is not None and
            total_minibatches >= args.num_minibatches):
            if args.enable_gavel_iterator:
                train_loader.complete()
            break
        elif(args.max_duration is not None and
             total_elapsed_time >= args.max_duration):
            if args.enable_gavel_iterator:
                train_loader.complete()
            break

        # evaluate on validation set
        # acc1 = validate(val_loader, model, criterion)

        # remember best acc@1 and save checkpoint
        #best_acc1 = max(acc1, best_acc1)
    if not args.distributed or args.rank == 0:
        state = {
            'epoch': epoch + (1 if cumulative_steps == len(train_loader) else 0),
            'arch': args.arch,
            'state_dict': model.state_dict(),
            # 'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'grad_calc_dict': grad_calc_dict,
            'original_batch_size': original_batch_size,
        }
        if args.enable_gavel_iterator:
            train_loader.save_checkpoint(state, checkpoint_path)
        else:
            save_checkpoint(state, checkpoint_path)

def check_critical_regime(grad_calc_dict, epoch, args, original_batch_size):
    check_freq = 10
    threshold = 0.5

    out_of_critical_regime = False
    status_changed = False
    if epoch % check_freq == 0:
        current_grad_norms = grad_calc_dict[epoch]        
        old_grad_norms = grad_calc_dict[epoch - check_freq] if epoch != 0 else [None] * len(current_grad_norms)
        if epoch != 0:
            # take the sum of gradient norms of all layers
            new_norm_sum = sum(current_grad_norms)
            prev_norm_sum = sum(old_grad_norms)
            # print(f"new_norm_sum is {new_norm_sum}, prev_norm_sum is {prev_norm_sum}")
            ratio = (abs(prev_norm_sum - new_norm_sum))/(prev_norm_sum)
            out_of_critical_regime = ratio < threshold
            if (out_of_critical_regime and args.batch_size == original_batch_size) or \
                ((not out_of_critical_regime) and args.batch_size != original_batch_size):
                status_changed = True
            print(f"[Epoch {epoch}], ratio: {round(ratio,5)}, out_of_critical_regime: {out_of_critical_regime}, status_changed: {status_changed}")
    return out_of_critical_regime, status_changed


def check_critical_regime_hardcode(epoch, total_num_epochs, current_batch_size, original_batch_size):
    # TODO: Run physical tests to make sure the correctness
    if original_batch_size in [16, 32, 64, 128]:
        regime = 10
    critical_regime = [x for x in range(600) if x % 30 < 10]

    critical_regime += list(range(round(0.3 * total_num_epochs)))

    out_of_critical_regime = not (epoch in critical_regime)
    # status_changed = out_of_critical_regime ^ ((epoch - 10) in critical_regime)  # xor
    status_changed = (out_of_critical_regime and current_batch_size == original_batch_size) or \
        ((not out_of_critical_regime) and current_batch_size != original_batch_size)

    return out_of_critical_regime, status_changed


# def out_of_critical_regime(total_minibatches):
#     """switch out of cr after 5 epochs
#     """
#     return (
#         (args.batch_size == 16 and total_minibatches == 31250) or  # 100000/16=6250, 6250*5=31250
#         (args.batch_size == 32 and total_minibatches == 15625) or  # 100000/32=3125, 3125*5=15625
#         (args.batch_size == 64 and total_minibatches == 7815)  # 100000/64=1563, 1563*5=7815
#     )

def gather_grad_array(model, full_rank_accum):
    # grad_array = [param.grad.data for param in model.parameters() if param.ndim == 4]
    grad_array = [param.grad.data for param in model.parameters()]
    for idx, grad_val in enumerate(grad_array):
        full_rank_accum[idx].add_(grad_val.data)

def train(train_loader, model, criterion, optimizer, epoch,
          total_minibatches, full_rank_accum, max_minibatches, total_elapsed_time,
          max_duration):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    elapsed_time = 0

    # switch to train mode
    model.train()

    start = time.time()
    end = time.time()
    finished_epoch = True
    i = 0
    cumulative_steps = 0
    for i, (input, target) in enumerate(train_loader):
        if (total_minibatches is not None and
            i + total_minibatches >= max_minibatches):
            finished_epoch = False
            break
        elif (max_duration is not None and
              total_elapsed_time + elapsed_time >= max_duration):
            finished_epoch = False
            break

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gather_grad_array(model, full_rank_accum)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        elapsed_time += (end - start)
        start = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

        if (args.throughput_estimation_interval is not None and
            i % args.throughput_estimation_interval == 0):
            print('[THROUGHPUT_ESTIMATION]\t%s\t%d' % (time.time(), i))
        
        cumulative_steps += 1
        
    return i, elapsed_time, finished_epoch, full_rank_accum, cumulative_steps

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def load_checkpoint(args, checkpoint_path):
    if os.path.exists(checkpoint_path):
        try:
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path,
                                    map_location='cuda:{}'.format(args.local_rank))
            return checkpoint
        except Exception as e:
            print('=> Could not load from checkpoint: %s' % (e))
            return None
    print("=> no checkpoint found at '{}'".format(checkpoint_path))
    return None

def save_checkpoint(state, checkpoint_filename):
    if checkpoint_filename is not None:
        torch.save(state, checkpoint_filename)
        print("=> saved checkpoint '{}'".format(checkpoint_filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
