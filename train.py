from __future__ import print_function
import os
import warnings
warnings.filterwarnings('ignore')

import time
import torch
import shutil
import argparse
from cfenet import build_net
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from layers.modules import MultiBoxLoss
from data import CFENET_ANCHOR_PARAMS
from data import COCODetection, VOCDetection, detection_collate, preproc
from configs.CC import Config

parser = argparse.ArgumentParser(description='CFENet Training')
parser.add_argument('-c', '--config', default='configs/cfenet300_vgg16.py')
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')
args = parser.parse_args()

if args.tensorboard:
    from logger import Logger
    date = time.strftime("%m_%d_%H_%M") + '_log'
    log_path = './' + date
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    logger = Logger(log_path)
global cfg
cfg = Config.fromfile(args.config)
anchor_config = CFENET_ANCHOR_PARAMS['{}_{}'.format(args.dataset, cfg.model.input_size)]
num_classes = getattr(cfg.model.num_classes, args.dataset)
Dataloader_function = {'VOC': VOCDetection, 'COCO': COCODetection}
net = build_net('train', 
                cfg = cfg.model, 
                num_classes = num_classes)
if cfg.model.resume_net:
    if cfg.model.backbone == 'vgg':
        net.init_model(cfg.model.pretrained)
    elif cfg.model.backbone == 'seresnet50':
        pass
else:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu>1:
    net = torch.nn.DataParallel(net)
if cfg.train_cfg.cuda:
    net.cuda()
    cudnn.benchmark = True
optimizer = optim.SGD(net.parameters(), 
                      lr = cfg.train_cfg.init_lr, 
                      momentum = cfg.optimizer.momentum, 
                      weight_decay = cfg.optimizer.weight_decay)
criterion = MultiBoxLoss(num_classes, 
                         overlap_thresh = cfg.loss.overlap_thresh, 
                         prior_for_matching = cfg.loss.prior_for_matching, 
                         bkg_label = cfg.loss.bkg_label,
                         neg_mining = cfg.loss.neg_mining,
                         neg_pos = cfg.loss.neg_pos,
                         neg_overlap = cfg.loss.neg_overlap,
                         encode_target = cfg.loss.encode_target)

priorbox = PriorBox(anchor_config)
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.train_cfg.cuda:
        priors = priors.cuda()

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    if epoch <= 5:
        lr = cfg.train_cfg.end_lr + (cfg.train_cfg.init_lr-cfg.train_cfg.end_lr)\
         * iteration / (epoch_size * cfg.train_cfg.warmup)
    else:
        lr = cfg.train_cfg.init_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    net.train()
    epoch = args.resume_epoch
    print('Loading Dataset...')
    _preproc = preproc(cfg.model.input_size, cfg.model.rgb_means, cfg.model.p)
    _Dataloader_function = Dataloader_function[args.dataset]
    dataset = _Dataloader_function(cfg.COCOroot if args.dataset == 'COCO' else cfg.VOCroot,
                                   getattr(cfg.dataset, args.dataset)['train_sets'],
                                   _preproc)
    epoch_size = len(dataset) // (cfg.train_cfg.per_batch_size * args.ngpu)
    max_iter = getattr(cfg.train_cfg.step_lr,args.dataset)[-1] * epoch_size
    stepvalues = [_*epoch_size for _ in getattr(cfg.train_cfg.step_lr, args.dataset)[:-1]]
    print('Training CFENet on ' + args.dataset)
    step_index = 0
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            batch_iterator = iter(data.DataLoader(dataset, 
                                                  cfg.train_cfg.per_batch_size * args.ngpu, 
                                                  shuffle=True, 
                                                  num_workers=cfg.train_cfg.num_workers, 
                                                  collate_fn=detection_collate))
            if epoch % cfg.model.save_eposhs == 0:
                torch.save(net.state_dict(), cfg.model.weights_save +\
                 'CFENet_{}_size{}_net{}_epoch{}.pth'.format(args.dataset,
                                                             cfg.model.input_size,
                                                             cfg.model.backbone,
                                                             epoch))
            epoch += 1
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, cfg.train_cfg.gamma, epoch, step_index, iteration, epoch_size)
        images, targets = next(batch_iterator)
        if cfg.train_cfg.cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        out = net(images)
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = loss_l + loss_c
        if args.tensorboard:
            info = {
                'loc_loss': loss_l.item(),
                'conf_loss': loss_c.item(),
                'loss': loss.item(),
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, iteration)
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        if iteration % cfg.train_cfg.print_epochs == 0:
            print('Time:{}||Epoch:{}||EpochIter:{}/{}||Iter:{}||Loss_L:{loss_l:.4f}||Loss_C:{loss_c:.4f}||Batch_Time:{bt:.4f}||LR:{lr:.7f}'.format(
                time.ctime(),
                epoch,
                iteration % epoch_size,
                epoch_size,
                iteration,
                loss_l=loss_l.item(),
                loss_c=loss_c.item(),
                bt=load_t1 - load_t0,
                lr=lr))
    torch.save(net.state_dict(), cfg.model.weights_save + \
            'Final_CFENet_{}_{}_{}.pth'.format(args.dataset,
                                               cfg.model.input_size,
                                               cfg.model.backbone))
