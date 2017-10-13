#!/usr/bin/env python
# coding: utf-8

import os
import time
import importlib
import json
from collections import OrderedDict
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from dataloader import get_loader

torch.backends.cudnn.benchmark = True

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument('--block_type', type=str, required=True)
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--base_channels', type=int, default=16)

    # run config
    parser.add_argument('--outdir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)

    # optim config
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--base_lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--milestones', type=str, default='[80, 120]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # TensorBoard
    parser.add_argument(
        '--tensorboard', dest='tensorboard', action='store_true')

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False

    model_config = OrderedDict([
        ('arch', 'resnet'),
        ('block_type', args.block_type),
        ('depth', args.depth),
        ('base_channels', args.base_channels),
        ('input_shape', (1, 3, 32, 32)),
        ('n_classes', 10),
    ])

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('base_lr', args.base_lr),
        ('weight_decay', args.weight_decay),
        ('momentum', args.momentum),
        ('nesterov', args.nesterov),
        ('milestones', json.loads(args.milestones)),
        ('lr_decay', args.lr_decay),
    ])

    data_config = OrderedDict([
        ('dataset', 'CIFAR10'),
    ])

    run_config = OrderedDict([
        ('seed', args.seed),
        ('outdir', args.outdir),
        ('num_workers', args.num_workers),
        ('tensorboard', args.tensorboard),
    ])

    config = OrderedDict([
        ('model_config', model_config),
        ('optim_config', optim_config),
        ('data_config', data_config),
        ('run_config', run_config),
    ])

    return config


def load_model(config):
    module = importlib.import_module(config['arch'])
    Network = getattr(module, 'Network')
    return Network(config)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def train(epoch, model, optimizer, criterion, train_loader, run_config,
          writer):
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    accuracy_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(train_loader):
        global_step += 1

        if run_config['tensorboard'] and step == 0:
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        data = Variable(data.cuda())
        targets = Variable(targets.cuda())

        optimizer.zero_grad()

        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.data[0]
        correct_ = preds.eq(targets).cpu().sum().data.numpy()[0]
        num = data.size(0)

        accuracy = correct_ / num

        loss_meter.update(loss_, num)
        accuracy_meter.update(accuracy, num)

        if run_config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_, global_step)
            writer.add_scalar('Train/RunningAccuracy', accuracy, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'Accuracy {:.4f} ({:.4f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            accuracy_meter.val,
                            accuracy_meter.avg,
                        ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/Accuracy', accuracy_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)


def test(epoch, model, criterion, test_loader, run_config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    start = time.time()
    for step, (data, targets) in enumerate(test_loader):
        if run_config['tensorboard'] and epoch == 0 and step == 0:
            image = torchvision.utils.make_grid(
                data, normalize=True, scale_each=True)
            writer.add_image('Test/Image', image, epoch)

        data = Variable(data.cuda(), volatile=True)
        targets = Variable(targets.cuda(), volatile=True)

        outputs = model(data)
        loss = criterion(outputs, targets)

        _, preds = torch.max(outputs, dim=1)

        loss_ = loss.data[0]
        correct_ = preds.eq(targets).cpu().sum().data.numpy()[0]
        num = data.size(0)

        loss_meter.update(loss_, num)
        correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(test_loader.dataset)

    logger.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, accuracy))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if run_config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Test/Accuracy', accuracy, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    return accuracy


def main():
    # parse command line arguments
    config = parse_args()
    logger.info(json.dumps(config, indent=2))

    run_config = config['run_config']
    optim_config = config['optim_config']

    # TensorBoard SummaryWriter
    writer = SummaryWriter() if run_config['tensorboard'] else None

    # set random seed
    seed = run_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = run_config['outdir']
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # save config as json file in output directory
    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(config, fout, indent=2)

    # data loaders
    train_loader, test_loader = get_loader(optim_config['batch_size'],
                                           run_config['num_workers'])

    # model
    model = load_model(config['model_config'])
    model.cuda()
    n_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    logger.info('n_params: {}'.format(n_params))

    criterion = nn.CrossEntropyLoss(size_average=True)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=optim_config['base_lr'],
        momentum=optim_config['momentum'],
        weight_decay=optim_config['weight_decay'],
        nesterov=optim_config['nesterov'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optim_config['milestones'],
        gamma=optim_config['lr_decay'])

    # run test before start training
    test(0, model, criterion, test_loader, run_config, writer)

    for epoch in range(1, optim_config['epochs'] + 1):
        scheduler.step()

        train(epoch, model, optimizer, criterion, train_loader, run_config,
              writer)
        accuracy = test(epoch, model, criterion, test_loader, run_config,
                        writer)

        state = OrderedDict([
            ('config', config),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('accuracy', accuracy),
        ])
        model_path = os.path.join(outdir, 'model_state.pth')
        torch.save(state, model_path)

    if run_config['tensorboard']:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)


if __name__ == '__main__':
    main()
