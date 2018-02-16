import os
import torch
import numpy as np
import cv2
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

import faster_rcnn.roi_data_layer.roidb as rdl_roidb
from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
import torchvision
import torch.nn as nn
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def log_print(text):
    print(text)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()

# hyper-parameters
# ------------
imdb_name = 'voc_2007_trainval'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'models/resnet101-caffe.pth'
output_dir = 'models/saved_model3'

start_step = 0
end_step = 200000
lr_decay_steps = {80000, 120000, 160000}
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY

# load data
imdb = get_imdb(imdb_name)
rdl_roidb.prepare_roidb(imdb)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG, training = True)

net.cuda()
net.train()
net.apply(weight_init) #-- parameters initialize

#----download resnet101 weights-----
pretrained_state = torch.load(pretrained_model)
net.resnet.load_state_dict({k:v for k, v in pretrained_state.items() if k in net.resnet.state_dict()})
for p in net.resnet.conv1.parameters(): p.requires_grad=False
for p in net.resnet.bn1.parameters(): p.requires_grad=False
for p in net.resnet.layer1.parameters(): p.requires_grad=False
for p in net.resnet.layer2.parameters(): p.requires_grad=False

params = []
#params = list(net.parameters())
for p in list(net.parameters()):
    if p.requires_grad == False: continue
    params.append(p)
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
time = Timer()
time.tic()
for step in range(start_step, end_step+1):

    # get one batch
    blobs = data_layer.forward()
    im_data = blobs['data']
    im_info = blobs['im_info'] #---[heiht, width, scale]
    gt_boxes = blobs['gt_boxes'] #---[[x1, y1, x2, y2, label]]
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss
    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()

    if step % disp_interval == 0:
        duration = time.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_print(log_text)

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
            print " "
        re_cnt = True


    if (step % 40000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.pth'.format(step))
        torch.save(net.state_dict(), save_name)
        print('save model: {}'.format(save_name))

    if step in lr_decay_steps:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
  
    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        time.tic()
        re_cnt = False  
