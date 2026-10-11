# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
'''https://blog.csdn.net/u013841196/article/details/82941410 采用网络多阶段特征融合'''
'''比train.py多了portion_independent和Direct Prediction的判断'''
from re import S
import torch

import torch.nn as nn
from torch.nn.modules import module
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision.models.inception import InceptionOutputs
import torchvision.transforms as transforms
import os
import argparse
from models import *
from models import myresnet
# from models.myresnet import resnet101,resnet50
from timm.models import *
from timm.models import create_model
from utils.utils import progress_bar, load_for_transfer_learning, logtxt, check_dirs
from utils_data1 import get_DataLoader
import pdb

from tqdm import tqdm
import random
import numpy as np
from collections import OrderedDict
import csv
import random
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss
import statistics
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from datetime import datetime


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

def set_seed(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--wd', default=0.9, type=float, help='weight decay')  # 5e-4
parser.add_argument('--min_lr', default=2e-4, type=float, help='minimal learning rate')  # 2e-4
parser.add_argument('--dataset',
                    choices=["nutrition_rgbd", "nutrition_rgb", "food101", "food172", "cub200/CUB_200_2011", "cifar10",
                             "cifar100"], default='nutrition_rgb',
                    help='cifar10 or cifar100')
parser.add_argument('--b', type=int, default=4,
                    help='batch size')
parser.add_argument('--resume', '-r', type=str,
                    help='resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num_classes', type=int, default=1024, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--model', default='resnet101', type=str, metavar='MODEL',
                    help='Name of models to train (default: "countception"：必须和t2t_vit.py中的 default_cfgs 命名相同')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop_connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop_block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img_size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => models default)')
parser.add_argument('--bn_tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn_momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn_eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--initial_checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize models from this checkpoint (default: none)')
# Transfer learning
parser.add_argument('--transfer_learning', default=False,
                    help='Enable transfer learning')
parser.add_argument('--transfer_model', type=str, default=None,
                    help='Path to pretrained models for transfer learning')
parser.add_argument('--transfer_ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')

parser.add_argument('--data_root', type=str, default=r"D:\Food_Calorie_Prediction\nutrition5k\nutrition5k_dataset",
                    help="our dataset root")
parser.add_argument('--run_name', type=str, default="editname")
parser.add_argument('--print_freq', type=int, default=200, help="the frequency of write to logtxt")
parser.add_argument("--warmup_steps", default=500, type=int,
                    help="Step of training to perform learning rate warmup for.")
# parser.add_argument('--cls_num', default=101, type=int, metavar='N', help='class number') #172 #见上面 num_classes
parser.add_argument('--mul_cls_num', default=174, type=int, metavar='N', help='ingradient class number')  # 353
parser.add_argument('--multi_task', action='store_true', help='multi-task classification')
parser.add_argument('--pool', default='spoc', type=str, help='pool function')
parser.add_argument('--embed_dim', default=384, type=int, help='T2t_vit_7,T2t_vit_10,T2t_vit_12:256;\
T2t_vit_14:384; T2t_vit_19:448; T2t_vit_24:512')
parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
parser.add_argument('--portion_independent', action='store_true', help='Nutrition5K: Portion Independent Model')
parser.add_argument('--direct_prediction', action='store_true', help='Nutrition5K: direct_prediction Model')
parser.add_argument('--rgbd', action='store_true', help='4 channels')
parser.add_argument('--gradnorm', action='store_true', help='GradNorm')
parser.add_argument('--alpha', '-a', type=float, default=0.12)
parser.add_argument('--sigma', '-s', type=float, default=100.0)
parser.add_argument('--rgbd_zscore', action='store_true', help='4 channels')  # train+test标准化
parser.add_argument('--rgbd_zscore_foronly_train_or_test_respectedly', action='store_true',
                    help='4 channels')  # 分别对train标准化和对test标准化
parser.add_argument('--rgbd_minmax', action='store_true', help='4 channels')
parser.add_argument('--rgbd_after_check', action='store_true', help='remained data after we check the dataset')
parser.add_argument('--rnn_layers', type=int, default=1)
parser.add_argument('--mixup', action='store_true', help='data augmentation')
parser.add_argument('--use_detect_label', action='store_true', help='data augmentation')
parser.add_argument('--use_detect_label_cutfeaturemap', action='store_true',
                    help='需要把transforms.CenterCrop((256,256))去除')

args = parser.parse_args()

set_seed(args)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
global_step = 0  # lmj 记录tensorboard中的横坐标

writer = SummaryWriter(log_dir=os.path.join(r"D:\Food_Calorie_Prediction\python_food\runs\news", f'checkpoint_{args.dataset}_{args.model}_{args.run_name}'))

# Data
print('==> Preparing data..')

print(f'learning rate:{args.lr}, weight decay: {args.wd}')

print('==> Building models..')
global net
print('==> Load checkpoint..')

if args.model == 'resnet101':
    pretrained_dict = torchvision.models.resnet101(pretrained=True).state_dict()
    net = myresnet.resnet101(rgbd=args.rgbd)
elif args.model == 'resnet50':
    pretrained_dict = torchvision.models.resnet50(pretrained=True).state_dict()
    net = myresnet.resnet50(rgbd=args.rgbd)
elif args.model == 'resnet152':
    pretrained_dict = torchvision.models.resnet152(pretrained=True).state_dict()
    net = myresnet.resnet152(rgbd=args.rgbd)

model_dict = net.state_dict()
new_state_dict = OrderedDict()
for k, v in pretrained_dict.items():
    # pdb.set_trace()
    if k in model_dict:  # update the same part
        # strip `module.` prefix
        name = k[7:] if k.startswith('module') else k
        new_state_dict[name] = v

model_dict.update(new_state_dict)
net.load_state_dict(model_dict)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# criterion = nn.L1Loss()
criterion = nn.SmoothL1Loss()
parameters = net.parameters()

optimizer = torch.optim.Adam([
        {'params': net.parameters(), 'lr': 1e-5, 'weight_decay': 5e-4},  # 5e-4
    ])

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)  # 0.99 (RGBD )

if args.resume:
    # Load checkpoint. resume时不同于上面加载预训练模型，checkpoint网络结构与当前网络一致，因此不需要if k in net.state_dict():
    print('==> Resuming from checkpoint..')
    models_state_dict = torch.load(args.resume)

    new_state_dict_rgb = OrderedDict()
    for k, v in models_state_dict['net'].items():
        name = k[7:] if k.startswith('module') else 'module.' + k
        new_state_dict_rgb[name] = v
    net.load_state_dict(new_state_dict_rgb)

    optimizer.load_state_dict(models_state_dict['optimizer'])

    start_epoch = models_state_dict['epoch']
# gradnorm
weights = []
task_losses = []
loss_ratios = []
grad_norm_losses = []

trainloader, testloader = get_DataLoader(args)

# Training
def train(epoch, net):
    # lmj global_step作为全局变量
    global global_step
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    print(current_time)
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    epoch_iterator = tqdm(trainloader,
                          desc="Training (X / X Steps) (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    for batch_idx, x in enumerate(epoch_iterator):  # (inputs, targets,ingredient)
        '''Portion Independent Model'''

        inputs = x[0].to(device)
        total_calories = x[2].to(device).float()
        total_mass = x[3].to(device).float()
        total_fat = x[4].to(device).float()
        total_carb = x[5].to(device).float()
        total_protein = x[6].to(device).float()

        optimizer.zero_grad()  # 0817

        outputs = net(inputs)

        total_calories_loss = total_calories.shape[0] * criterion(outputs[0],total_calories) / total_calories.sum().item()
        total_mass_loss = total_calories.shape[0] * criterion(outputs[1], total_mass) / total_mass.sum().item()
        total_fat_loss = total_calories.shape[0] * criterion(outputs[2], total_fat) / total_fat.sum().item()
        total_carb_loss = total_calories.shape[0] * criterion(outputs[3], total_carb) / total_carb.sum().item()
        total_protein_loss = total_calories.shape[0] * criterion(outputs[4],total_protein) / total_protein.sum().item()

        loss = total_calories_loss + total_mass_loss + total_fat_loss + total_carb_loss + total_protein_loss

        loss.backward()
        optimizer.step()
        global_step += 1

        train_loss += loss.item()
        calories_loss += total_calories_loss.item()
        mass_loss += total_mass_loss.item()
        fat_loss += total_fat_loss.item()
        carb_loss += total_carb_loss.item()
        protein_loss += total_protein_loss.item()

        '''
        if (batch_idx % 100) == 0:
            # epoch_iterator.set_description(
            #             "Training Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %f" % (epoch, train_loss/(batch_idx+1), calories_loss/(batch_idx+1), mass_loss/(batch_idx+1), fat_loss/(batch_idx+1), carb_loss/(batch_idx+1),protein_loss/(batch_idx+1), optimizer.param_groups[0]['lr'])
            #         )
            print("\nTraining Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %.5f" % (epoch, train_loss/(batch_idx+1), calories_loss/(batch_idx+1), mass_loss/(batch_idx+1), fat_loss/(batch_idx+1), carb_loss/(batch_idx+1),protein_loss/(batch_idx+1), optimizer.param_groups[0]['lr']))
        '''

        if (batch_idx + 1) % args.print_freq == 0 or batch_idx + 1 == len(trainloader):
            logtxt(log_file_path, 'Epoch: [{}][{}/{}]\t'
                                  'EpochTime(s): {:.2f}\t'
                                  'Loss: {:2.5f} \t'
                                  'calorieloss: {:2.5f} \t'
                                  'massloss: {:2.5f} \t'
                                  'fatloss: {:2.5f} \t'
                                  'carbloss: {:2.5f} \t'
                                  'proteinloss: {:2.5f} \t'
                                  'lr:{:.7f}'.format(
                epoch, batch_idx + 1, current_time,len(trainloader),
                       train_loss / (batch_idx + 1),
                       calories_loss / (batch_idx + 1),
                       mass_loss / (batch_idx + 1),
                       fat_loss / (batch_idx + 1),
                       carb_loss / (batch_idx + 1),
                       protein_loss / (batch_idx + 1),
                optimizer.param_groups[0]['lr']))

best_loss = 10000

def test(epoch, net):
    # writer  写入tensorboard
    # global best_acc
    global best_loss
    net.eval()

    test_loss = 0
    calories_loss = 0
    mass_loss = 0
    fat_loss = 0
    carb_loss = 0
    protein_loss = 0

    epoch_iterator = tqdm(testloader,
                          desc="Testing... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True)
    csv_rows = []
    with torch.no_grad():
        for batch_idx, x in enumerate(epoch_iterator):  # testloader

            inputs = x[0].to(device)
            total_calories = x[2].to(device).float()
            total_mass = x[3].to(device).float()
            total_fat = x[4].to(device).float()
            total_carb = x[5].to(device).float()
            total_protein = x[6].to(device).float()

            optimizer.zero_grad()

            outputs = net(inputs)

            calories_total_loss = total_calories.shape[0] * criterion(outputs[0],total_calories) / total_calories.sum().item()
            mass_total_loss = total_calories.shape[0] * criterion(outputs[1], total_mass) / total_mass.sum().item()
            fat_total_loss = total_calories.shape[0] * criterion(outputs[2], total_fat) / total_fat.sum().item()
            carb_total_loss = total_calories.shape[0] * criterion(outputs[3], total_carb) / total_carb.sum().item()
            protein_total_loss = total_calories.shape[0] * criterion(outputs[4],total_protein) / total_protein.sum().item()

            loss = calories_total_loss + mass_total_loss + fat_total_loss + carb_total_loss + protein_total_loss

            if epoch % 10 == 0:
                    # 每10个epoch将当前测试集所有预测值写入csv
                    for i in range(len(x[1])):  # IndexError: tuple index out of range  最后一轮的图片数量不到32，不能被batchsiz
                        dish_id = x[1][i]
                        calories = outputs[0][i]
                        mass = outputs[1][i]
                        fat = outputs[2][i]
                        carb = outputs[3][i]
                        protein = outputs[4][i]
                        dish_row = [dish_id, calories.item(), mass.item(), fat.item(), carb.item(), protein.item()]
                        csv_rows.append(dish_row)

            test_loss += loss.item()
            calories_loss += calories_total_loss.item()
            mass_loss += mass_total_loss.item()
            fat_loss += fat_total_loss.item()
            carb_loss += carb_total_loss.item()
            protein_loss += protein_total_loss.item()

            epoch_iterator.set_description(
                "Testing Epoch[%d] | loss=%2.5f | calorieloss=%2.5f | massloss=%2.5f| fatloss=%2.5f | carbloss=%2.5f | proteinloss=%2.5f | lr: %.5f" % (
                epoch, test_loss / (batch_idx + 1), calories_loss / (batch_idx + 1), mass_loss / (batch_idx + 1),
                fat_loss / (batch_idx + 1), carb_loss / (batch_idx + 1), protein_loss / (batch_idx + 1),
                optimizer.param_groups[0]['lr'])
            )
        # lmj writer to tensorboard
        writer.add_scalar("test/loss", scalar_value=test_loss/(len(testloader)), global_step=global_step)
        writer.add_scalar("test/calorieloss", scalar_value=calories_loss/(len(testloader)), global_step=global_step) # calories_loss/(batch_idx+1)
        writer.add_scalar("test/massloss", scalar_value=mass_loss/(len(testloader)), global_step=global_step)
        writer.add_scalar("test/fatloss", scalar_value=fat_loss/(len(testloader)), global_step=global_step)
        writer.add_scalar("test/carbloss", scalar_value=carb_loss/(len(testloader)), global_step=global_step)
        writer.add_scalar("test/proteinloss", scalar_value=protein_loss/(len(testloader)), global_step=global_step)

        logtxt(log_file_path, 'Test Epoch: [{}][{}/{}]\t'
                              'Loss: {:2.5f} \t'
                              'calorieloss: {:2.5f} \t'
                              'massloss: {:2.5f} \t'
                              'fatloss: {:2.5f} \t'
                              'carbloss: {:2.5f} \t'
                              'proteinloss: {:2.5f} \t'
                              'lr:{:.7f}\n'.format(
            epoch, batch_idx + 1, len(testloader),
                   test_loss / len(testloader),
                   calories_loss / len(testloader),
                   mass_loss / len(testloader),
                   fat_loss / len(testloader),
                   carb_loss / len(testloader),
                   protein_loss / len(testloader),
            optimizer.param_groups[0]['lr']))

    if best_loss > test_loss:
        best_loss = test_loss
        print('Saving..')
        net = net.module if hasattr(net, 'module') else net
        state = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        savepath = f"./saved/new/regression_{args.dataset}_{args.model}_{args.run_name}"
        check_dirs(savepath)
        torch.save(state, os.path.join(savepath, f"ckpt_best.pth"))


    if epoch % 1 == 0:
        new_csv_rows = []
        predict_values = dict()
        # pdb.set_trace()
        key = ''
        for iterator in csv_rows:
            if key != iterator[0]:
                key = iterator[0]
                predict_values[key] = []
                predict_values[key].append(iterator[1:])
            else:
                predict_values[key].append(iterator[1:])
        # pdb.set_trace()
        for k, v in predict_values.items():
            nparray = np.array(v)
            predict_values[k] = np.mean(nparray, axis=0)  # 每列求均值
            new_csv_rows.append(
                [k, predict_values[k][0], predict_values[k][1], predict_values[k][2], predict_values[k][3],
                 predict_values[k][4]])

        headers = ["dish_id", "calories", "mass", "fat", "carb", "protein"]
        csv_file_path = os.path.join("logs_nutrition2", f'checkpoint_{args.dataset}_{args.model}_{args.run_name}',
                                     "epoch{}_result_image.csv".format(epoch))
        # 每张图片的结果写入csv
        # with open(csv_file_path,'w')as f:
        #     f_csv = csv.writer(f)
        #     f_csv.writerow(headers)
        #     f_csv.writerows(csv_rows)
        # 每个dish写入csv
        csv_file_path2 = os.path.join(r"D:\Food_Calorie_Prediction\python_food\logs\news",
                                      f'checkpoint_{args.dataset}_{args.model}_{args.run_name}',
                                      "epoch{}_result_dish.csv".format(epoch))
        with open(csv_file_path2, 'w') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(new_csv_rows)


log_file_path = os.path.join(r"D:\Food_Calorie_Prediction\python_food\logs\news",
                             f'checkpoint_{args.dataset}_{args.model}_{args.run_name}', "train_log.txt")
check_dirs(
    os.path.join(r"D:\Food_Calorie_Prediction\python_food\logs\news", f'checkpoint_{args.dataset}_{args.model}_{args.run_name}'))
logtxt(log_file_path, str(vars(args)))
for epoch in range(start_epoch, start_epoch + 100):
    train(epoch, net)
    test(epoch, net)
    scheduler.step()

