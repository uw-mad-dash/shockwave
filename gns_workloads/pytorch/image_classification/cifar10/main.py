'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datetime import datetime

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import math
import socket
import sys
import time
import numpy as np
import pymannkendall as mk

from pynvml import *

from models import *
from utils import progress_bar

# TODO: Figure out a cleaner way of including gavel_iterator.
imagenet_dir = os.path.dirname(os.path.realpath(__file__))
image_classification_dir = os.path.dirname(imagenet_dir)
pytorch_dir = os.path.dirname(image_classification_dir)
workloads_dir = os.path.dirname(pytorch_dir)
gpusched_dir = os.path.dirname(workloads_dir)
scheduler_dir = os.path.join(gpusched_dir, 'scheduler')
sys.path.append(scheduler_dir)
from gavel_iterator import GavelIterator

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--data_dir', required=True, type=str, help='Data directory')
parser.add_argument('--num_epochs', default=None, type=int, help='Number of epochs to train for')
parser.add_argument('--num_steps', default=None, type=int, help='Number of steps to train for')
parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--checkpoint_dir', default='/lfs/1/keshav2/checkpoints/resnet-18',
                    type=str, help='Checkpoint directory')
parser.add_argument('--use_progress_bar', '-p', action='store_true', default=False, help='Use progress bar')
parser.add_argument('--log_interval', type=int, default=100,
                    help='Interval to log')
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
parser.add_argument('--throughput_estimation_interval', type=int, default=None,
                    help='Steps between logging steps completed')
parser.add_argument('--max_duration', type=int, default=None,
                    help='Maximum duration in seconds')
parser.add_argument('--enable_gavel_iterator', action='store_true',
                    default=False, help='If set, use Gavel iterator')

args = parser.parse_args()
if args.batch_size > 256:
    args.batch_size = 256
print("batch size at start of model is ", args.batch_size)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
#random.seed(args.seed)
np.random.seed(0)
os.environ["PYTHONHASHSEED"] = str(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('==> Starting script..')

torch.cuda.set_device(args.local_rank)
if args.num_epochs is not None and args.num_steps is not None:
    raise ValueError('Only one of num_epochs and num_steps may be set')
elif args.num_epochs is None and args.num_steps is None:
    raise ValueError('One of num_epochs and num_steps must be set')

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Model
print('==> Building model..')
# net = ResNet18()
net = resnet18(10)
# net = VGG("VGG11")
net = net.cuda()

args.distributed = False
if args.master_addr is not None:
    args.distributed = True
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = str(args.master_port)
    os.environ['NCCL_DEBUG'] = 'INFO'
    # hostname = socket.gethostname()
    # if "node" in hostname: # wisr cluster
    #     socket_ifname = 'enps0f0' #if hostname != 'node1' else 'enp94s0f0'  # rdma nics
    #     # socket_ifname = 'eno4'
    # else:  # ec2
    #     socket_ifname = "ens3"
    # os.environ['NCCL_SOCKET_IFNAME'] = socket_ifname
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)
    net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank],
            output_device=args.local_rank)
    #CHECK if this is called in single machine as well
    print("args.distributed is ", args.distributed)
    #net = torch.nn.DataParallel(net, device_ids=[args.local_rank],output_device=args.local_rank)

# Data
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
train_sampler = None
if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=0,
                                          sampler=train_sampler)

testset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


cumulative_steps = 0
cumulative_time = 0
#new_params_for_GNS
grad_norm_arr = []
S_arr = []
temp_grad_norm_queue = []
if args.distributed:
    window_size = args.world_size
else:
    window_size = 2
print("window size is ", window_size)
sliding_grad_array = []
noisePercentage = 10
gnsPrev = 0
gns_arr = []

def load_checkpoint(args, checkpoint_path):
    # Load checkpoint.
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path,
                                    map_location='cuda:{}'.format(args.local_rank))
            print('==> Resuming from checkpoint at %s...' % (checkpoint_path))
            return checkpoint
        except Exception as e:
            print('Error reading checkpoint: %s' % (e))
            return None
    return None

def save_checkpoint(checkpoint_path, state):
    print('==> Saving checkpoint at %s...' % (checkpoint_path))
    torch.save(state, checkpoint_path)

if args.checkpoint_dir is not None:
    checkpoint_path = os.path.join(args.checkpoint_dir, 'model.chkpt')

if args.enable_gavel_iterator:
    trainloader = GavelIterator(trainloader, args.checkpoint_dir,
                                load_checkpoint_func=load_checkpoint,
                                save_checkpoint_func=save_checkpoint)
    checkpoint = trainloader.load_checkpoint(args, checkpoint_path)
else:
    checkpoint = load_checkpoint(args, checkpoint_path)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
# optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=5e-4)
optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=0.0001)


grad_calc_dict = dict()
original_batch_size = None

if checkpoint is not None:
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    grad_calc_dict = checkpoint['grad_calc_dict']
    original_batch_size = checkpoint['original_batch_size']
    grad_norm_arr = checkpoint['grad_norm_arr']
    S_arr = checkpoint['S_arr']
    temp_grad_norm_queue = checkpoint['temp_grad_norm_queue']
    gnsPrev = checkpoint['gnsPrev']
    sliding_grad_array = checkpoint['sliding_grad_array']
    window_size = checkpoint['window_size']
    noisePercentage = checkpoint['noisePercentage']
    gns_arr = checkpoint['gns_arr']
    # if args.batch_size != original_batch_size and bs_last_interval == original_batch_size:
    #     print(f"#############Immedate interval after switching to non-critical regime, not loading grad_calc_dict")
    #     # the immediate interval after switching to non-critical regime
    #     # do not check for critical regime
    #     grad_calc_dict = dict()
    # NOTE: It seems that the Gavel checkpoint logic is somehow faulty. It does not account for the situations
    # in which checkpoint happens in the middle of an epoch. After loading the checkpoint, it repeats
    # the previous epoch. In other words, only the num_iterations is followed but this might result
    # in accuracy loss.
    print(f"Loaded from checkpoint, start epoch is {start_epoch}")
    #print("loading checkpoint values S_arr, temp_grad_norm, gnsPrev window_size", S_arr, temp_grad_norm_queue, gnsPrev, window_size)
else:
    original_batch_size = args.batch_size

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size

# Training
def train(epoch, full_rank_accum, cumulative_steps=None, cumulative_time=None, grad_norm_arr=None, S_arr=None, temp_grad_norm_queue=None, sliding_grad_array=None):
    print("Epoch for args local rank and rank is ",epoch, args.local_rank, args.rank)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    done = False
    finished_epoch = True
    start_time = time.time()
    num_completed_iters = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # if args.distributed == True:
        #     temp_grad_norm_queue = []
        #     grad_array_temp = [param.grad.data for param in net.parameters()]
        #     grad_norm_temp_arr = [torch.norm(pval).item() for pval in grad_array_temp]
        #     temp_grad_norm_queue.append(sum(grad_norm_temp_arr))
        #     average_gradients(net)
        
        optimizer.step()

        #if args.distributed and args.world_size > 4:
            #print("args is distributed")
        grad_norm_arr, S_arr, temp_grad_norm_queue, sliding_grad_array  = append_GNS_params(net,grad_norm_arr, S_arr, temp_grad_norm_queue, sliding_grad_array)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if args.use_progress_bar:
          progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        elif batch_idx % args.log_interval == 0 and batch_idx > 0:
            print('Batch: %d, Loss: %.3f, '
                  'Acc: %.3f%% (%d/%d)' % (batch_idx, train_loss/(batch_idx+1),
                                           100.*correct/total, correct, total))
        if cumulative_time is not None:
            cumulative_time += time.time() - start_time
            if (args.max_duration is not None and
                cumulative_time > args.max_duration):
                # print(f"done is True because args.max_duration is not None and cumulative_time > args.max_duration")
                # print(f"args.max_duration: {args.max_duration}, cumulative_time: {cumulative_time}")
                done = True
                finished_epoch = False
            start_time = time.time()
        if cumulative_steps is not None:
            cumulative_steps += 1
            if (args.throughput_estimation_interval is not None and
                cumulative_steps % args.throughput_estimation_interval == 0):
                print('[THROUGHPUT_ESTIMATION]\t%s\t%d' % (time.time(),
                                                           cumulative_steps))
            if args.num_steps is not None and cumulative_steps >= args.num_steps:
                # print(f"done is True because args.num_steps is not None and cumulative_steps >= args.num_steps")
                # print(f"args.num_steps: {args.num_steps}, cumulative_steps: {cumulative_steps}")
                done = True
                finished_epoch = False
                break
        num_completed_iters += 1
        broadcastTensor = torch.zeros(1, 1)
        
    return (cumulative_steps, cumulative_time, done, finished_epoch, num_completed_iters, full_rank_accum, grad_norm_arr, S_arr, temp_grad_norm_queue, sliding_grad_array)


def gather_grad_array(model, full_rank_accum):
    grad_array = [param.grad.data for param in model.parameters() if param.ndim == 4]
    # grad_array = [param.grad.data for param in model.parameters()]
    for idx, grad_val in enumerate(grad_array):
        full_rank_accum[idx].add_(grad_val.data)

def get_grad_norm_big(model, sliding_grad_array):
    grad_array1 = [param.grad.data for param in model.parameters()]
    sliding_grad_array.append(grad_array1)
    sliding_tensor_sum = [torch.zeros_like(copy_l) for copy_l in model.parameters()]
    grad_norm_big= 0
    #print("sliding window size is", len(sliding_grad_array))
    if window_size is not None and len(sliding_grad_array)>=window_size:
        if len(sliding_grad_array)> window_size:
            sliding_grad_array.pop(0)
        for tensor in sliding_grad_array:
            for idx, grad_val in enumerate(tensor):
                sliding_tensor_sum[idx].add_(grad_val.data/float(window_size))
        grad_norm_big_arr = [torch.norm(pval).item() for pval in sliding_tensor_sum]
        grad_norm_big = sum(grad_norm_big_arr)
    return (grad_norm_big, sliding_grad_array)

def append_GNS_params(model,grad_norm_arr, S_arr, temp_grad_norm_queue, sliding_grad_array):
    # if args.distributed == True: 
    #     #print("inside append_gns_param distributed")
    #     grad_norm_small = temp_grad_norm_queue[0]
    #     grad_array_temp = [param.grad.data for param in model.parameters()]
    #     grad_norm_temp_arr = [torch.norm(pval).item() for pval in grad_array_temp]
    #     grad_norm_big = sum(grad_norm_temp_arr)
    #     size = float(dist.get_world_size())
    #     grad_norm_val = (float(size)*grad_norm_big-grad_norm_small)/float(size-1)
    #     S_val = ((float(size)*args.batch_size)*(grad_norm_small-grad_norm_big))/float(size-1)
    #     grad_norm_arr.append(grad_norm_val)
    #     S_arr.append(S_val)
    #     #print("grad norm amall and big is ", grad_norm_small, grad_norm_big)
    #     return (grad_norm_arr,S_arr, temp_grad_norm_queue, sliding_grad_array)   
    (grad_norm_big, sliding_grad_array) = get_grad_norm_big(model,sliding_grad_array)
    grad_array_temp = [param.grad.data for param in model.parameters()]
    grad_norm_temp_arr = [torch.norm(pval).item() for pval in grad_array_temp]
    norm_sum_temp = sum(grad_norm_temp_arr)
    temp_grad_norm_queue.append(norm_sum_temp)
    #print("window siz ia", window_size)
    #print("temp q length is ", len(temp_grad_norm_queue))
    if window_size is not None and len(temp_grad_norm_queue)>=window_size:
        if len(temp_grad_norm_queue)> window_size:
            temp_grad_norm_queue.pop(0) 
        grad_norm_small = temp_grad_norm_queue[0]
        grad_norm_val = (float(window_size)*grad_norm_big-grad_norm_small)/float(window_size-1)
        S_val = ((float(window_size)*args.batch_size)*(grad_norm_small-grad_norm_big))/float(window_size-1)
        grad_norm_arr.append(grad_norm_val)
        S_arr.append(S_val)
    return (grad_norm_arr,S_arr, temp_grad_norm_queue, sliding_grad_array)

def get_GNS(grad_norm_arr,S_arr):
    if len(S_arr)==0:
        print("S_array is 0")
        return 0
    S_avg = sum(S_arr)/float(len(S_arr))
    grad_norm_avg = sum(grad_norm_arr)/float(len(grad_norm_arr))
    gns = S_avg/grad_norm_avg
    return gns

# NOTE: This test function is not used
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving checkpoint at %s...' % (checkpoint_path))
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
        }
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)
        torch.save(state, checkpoint_path)
        best_acc = acc

if args.num_steps is not None:
    args.num_epochs = math.ceil(float(args.num_steps) / len(trainloader))
    print(f"Total num steps: {args.num_steps}, total epochs: {args.num_epochs}")


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
            if out_of_critical_regime and args.batch_size == original_batch_size or \
                not out_of_critical_regime and args.batch_size == 2 * original_batch_size:
                # either previously in critical regime but is now out
                # or previously not in critical regime but is now in
                status_changed = True
            print(f"[Epoch {epoch}], ratio: {round(ratio,5)}, out_of_critical_regime: {out_of_critical_regime}, status_changed: {status_changed}")
    return out_of_critical_regime, status_changed


def adjust_learning_rate(args, optimizer, epoch):
    if args.batch_size > 128:
        # bs=512 for 300 epochs:  start from lr=0.1, scale to 0.4 linearly in 5 epochs, decay at 150 and 250
        lrs = [0.1, 0.175, 0.25, 0.325, 0.4] + 145 * \
            [0.4] + 100 * [0.04] + 5000 * [0.004]
    else:
        # does not scale learning rate if bs <= 128
        lrs = [0.1] * 150 + [0.01] * 100 + [0.001] * 5000
        # # debug
        # lrs = [0.1] * 20 + [0.01] * 230 + [0.001] * 50
    
    if epoch != 0 and lrs[epoch] != lrs[epoch - 1]:
        print(f"Epoch {epoch}, prev lr {lrs[epoch - 1]}, curr lr {lrs[epoch]}\n")
        for group in optimizer.param_groups:
            group['lr'] = lrs[epoch]

def getMemoryInfo():
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print(f'total    : {info.total}')
    print(f'free     : {info.free}')
    print(f'used     : {info.used}')
    return (info.free, info.used)

def linear_learning_rate(optimizer):
    for group in optimizer.param_groups:
            group['lr'] = group['lr']*2

def sqRoot_learning_rate(optimizer):
    for group in optimizer.param_groups:
            group['lr'] = group['lr']*1.414

def adascale_learning_rate(optimizer, gns, batch_size):
    thetaT = batch_size*gns
    rate = ((thetaT/batch_size)+1)*((thetaT/(2*batch_size))+1)
    for group in optimizer.param_groups:
            group['lr'] = group['lr']*rate


# def out_of_critical_regime(cumulative_steps):
#     """switch out of cr after 5 epochs
#     """
#     return (
#         (args.batch_size == 16 and cumulative_steps == 15625) or  # 50000/16=3125, 3125*5=15625
#         (args.batch_size == 32 and cumulative_steps == 7815) or  # 50000/32=1563, 1563*5=7815
#         (args.batch_size == 64 and cumulative_steps == 3910) or  # 50000/64=782, 782*5=3910
#         (args.batch_size == 128 and cumulative_steps == 1955)  # 50000/128=391, 391*5=1955
#     )

print(f"start_epoch is {start_epoch}, args.num_epochs is {args.num_epochs}")

# batch_size_switch_set = set([(16,30),(32,40),(64,50),(128,70), (32,20), (64,30), (128, 50),(64, 10),(128,30),(128, 10)])
# batch_size_switch_set_2gpu = set([(16, 20),(32, 30), (64, 90),(128, 110),(32,10),(64, 20),(128, 40)])
# batch_size_switch_set_4gpu = set([(16, 10),(32, 20),(64, 80),(128, 90),(64, 30),(128, 60),(64, 10),(128,10)])

for epoch in range(start_epoch, args.num_epochs):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}")
    #adjust_learning_rate(args, optimizer, epoch)
    grad_norm_arr = []
    S_arr = []
    temp_grad_norm_queue = []
    sliding_grad_array = []
    full_rank_accum = [torch.zeros_like(copy_l) for copy_l in net.parameters() if copy_l.ndim == 4]
    (cumulative_steps, cumulative_time, done, finished_epoch, num_completed_iters, full_rank_accum, grad_norm_arr, S_arr, temp_grad_norm_queue, sliding_grad_array) =\
            train(epoch, full_rank_accum, cumulative_steps, cumulative_time, grad_norm_arr, S_arr, temp_grad_norm_queue, sliding_grad_array)
    # grad_calc_dict[epoch] = [torch.norm(pval).item() for pval in full_rank_accum]
    # print(f"[Epoch {epoch}] Gradient norm of all layers: {grad_calc_dict[epoch]}")
    # print(f"[Epoch {epoch}] Sum of the gradient norms: {sum(grad_calc_dict[epoch])}")

    # check for critical regime
    # out_of_critical_regime, status_changed = check_critical_regime(grad_calc_dict, epoch, args, original_batch_size)
    if args.enable_gavel_iterator:
        # # FIXME: debug
        # if epoch == 0: 
        #     trainloader.update_resource_requirement(big_bs=True, small_bs=False)
        # (memFree, memUsed) = getMemoryInfo()
        if args.distributed and args.world_size > 4:
            gns = get_GNS(grad_norm_arr,S_arr)
            print("before get gns distributed is and local rank is ",args.distributed, args.rank)
            gns_arr.append(gns)
            print('For epoch, batch_size, gns and gnsprev lr is ',epoch, args.batch_size, gns, gnsPrev, optimizer.param_groups[0]['lr'])
            #(memFree, memUsed) = getMemoryInfo()
            if epoch!=0 and args.batch_size!=256 and epoch%10==0:
                mk_data = gns_arr[-10:]
                avg_ws = sum(mk_data)/10
                print("avg window gns is", avg_ws)
                # ws_per = 0
                # if avg_ws != 0:
                #     ws_per = ((gns-avg_ws)/abs(avg_ws))*100
                # print("window noise is ", ws_per )
                # if args.distributed == True:
                #     checkBsList = [None] * args.world_size
                #     updateBatchSize = 0
                #     if gns>avg_ws:
                #         updateBatchSize = 1
                #     dist.barrier()
                #     dist.all_gather_object(checkBsList,updateBatchSize)
                #     print("checktensor rank and worldsize is ",checkBsList, args.rank, args.world_size)
                #     if float(sum(checkBsList)) >= float(len(checkBsList)/2.0):
                #         linear_learning_rate(optimizer)
                #         trainloader.update_resource_requirement(big_bs=True, small_bs=False)
                #if args.distributed == False and gns>avg_ws:
                if gns>avg_ws:
                    linear_learning_rate(optimizer)
                    #sqRoot_learning_rate(optimizer)
                    #adascale_learning_rate(optimizer, gns, args.batch_size)
                    trainloader.update_resource_requirement(big_bs=True, small_bs=False)
            gnsPrev = gns
            print("gnsPrev and batchsize  for the next epoch",gnsPrev, args.batch_size)
        elif epoch!=0 and args.batch_size!=256 and epoch%10==0:
            print("in gpus metric")
            if not args.distributed:
                if original_batch_size == 16:
                    batch_size_switch_set = set([(16,30),(32,40),(64,50),(128,70)])
                elif original_batch_size == 32:
                    batch_size_switch_set = set([(32,20),(64,30),(128,50)])
                elif original_batch_size == 64:
                    batch_size_switch_set = set([(64,10),(128,30)])
                elif original_batch_size == 128:
                    batch_size_switch_set = set([(128,10)])
                else:
                    batch_size_switch_set = set([])
                if (args.batch_size,epoch) in batch_size_switch_set:
                    print("batch size and lr updated single", args.batch_size, epoch, optimizer.param_groups[0]['lr'])
                    linear_learning_rate(optimizer)
                    trainloader.update_resource_requirement(big_bs=True, small_bs=False)
            elif args.world_size == 2:
                if original_batch_size == 16:
                    batch_size_switch_set_2gpu = set([(16,20),(32,30),(64,90),(128,110)])
                elif original_batch_size == 32:
                    batch_size_switch_set_2gpu = set([(32,10),(64,20),(128,40)])
                elif original_batch_size == 64:
                    batch_size_switch_set_2gpu = set([(64,20),(128,40)])
                elif original_batch_size == 128:
                    batch_size_switch_set_2gpu = set([(128,40)])
                else:
                    batch_size_switch_set_2gpu = set([])
                if (args.batch_size,epoch) in batch_size_switch_set_2gpu:
                    print("batch size and lr updated 2gpu", args.batch_size, epoch, optimizer.param_groups[0]['lr'])
                    linear_learning_rate(optimizer)
                    trainloader.update_resource_requirement(big_bs=True, small_bs=False)
            elif args.world_size == 4:
                if original_batch_size == 16:
                    batch_size_switch_set_4gpu = set([(16,10),(32,20),(64,80),(128,90)])
                elif original_batch_size == 32:
                    batch_size_switch_set_4gpu = set([(32,20),(64,30),(128,60)])
                elif original_batch_size == 64:
                    batch_size_switch_set_4gpu = set([(64,10),(128,60)])
                elif original_batch_size == 128:
                    batch_size_switch_set_4gpu = set([(128,10)])
                else:
                    batch_size_switch_set_4gpu = set([])
                if (args.batch_size,epoch) in batch_size_switch_set_4gpu:
                    print("batch size and lr updated 4gpu", args.batch_size, epoch, optimizer.param_groups[0]['lr'])
                    #if (original_batch_size != 16 or not (original_batch_size == 16 and args.batch_size == 64 and epoch == 30)):
                    linear_learning_rate(optimizer)
                    trainloader.update_resource_requirement(big_bs=True, small_bs=False)
        # if status_changed:
        #     # if out_of_critical_regime and args.batch_size == original_batch_size or \
        #     #     not out_of_critical_regime and args.batch_size == 2 * original_batch_size:
        #     if out_of_critical_regime and args.batch_size != 256:
        #         # bs: 16, 32, 64, 128
        #         trainloader.update_resource_requirement(big_bs=True, small_bs=False)
        #     elif not out_of_critical_regime and args.batch_size != 16:
        #         # bs: 32, 64, 128
        #         trainloader.update_resource_requirement(big_bs=False, small_bs=True)
        if trainloader.done:
            # GavelIterator.done is true after detecting resource change
            break
        elif done:
            trainloader.complete()
            break
    elif done:
        break

if not args.distributed or args.rank == 0:
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch + (1 if num_completed_iters == len(trainloader) else 0),
        'grad_calc_dict': grad_calc_dict,
        'original_batch_size': original_batch_size,
        'grad_norm_arr': grad_norm_arr,
        'S_arr': S_arr,
        'temp_grad_norm_queue': temp_grad_norm_queue,
        'sliding_grad_array': sliding_grad_array,
        'gnsPrev': gnsPrev,
        'window_size': window_size,
        'noisePercentage': noisePercentage,
        'gns_arr': gns_arr 
    }
if not os.path.isdir(args.checkpoint_dir):
    os.mkdir(args.checkpoint_dir)
if not args.distributed or args.rank == 0:
    if args.enable_gavel_iterator:
        # TODO: more information (e.g., optimizer state) should be saved in the checkpoint, 
        # refer to jigsaw. Will come back to this later
        trainloader.save_checkpoint(checkpoint_path, state)
    else:
        save_checkpoint(checkpoint_path, state)
