from __future__ import print_function
import os
import warnings
warnings.filterwarnings('ignore')

import torch
import pickle
import argparse
import numpy as np
from cfenet import build_net
from utils.timer import Timer
from utils.nms_wrapper import nms
import torch.backends.cudnn as cudnn
from layers.functions import Detect,PriorBox
from data import CFENET_ANCHOR_PARAMS
from data import COCODetection, VOCDetection, BaseTransform
from configs.CC import Config
from tqdm import tqdm



parser = argparse.ArgumentParser(description='CFENet Testing')
parser.add_argument('-c', '--config', default='configs/cfenet300_vgg16.py', type=str)
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO version')
parser.add_argument('-m', '--trained_model', default=None, type=str, help='Trained state_dict file path to open')
args = parser.parse_args()

global cfg
cfg = Config.fromfile(args.config)
if not os.path.exists(cfg.test_cfg.save_folder):
    os.mkdir(cfg.test_cfg.save_folder)

anchor_config = CFENET_ANCHOR_PARAMS['{}_{}'.format(args.dataset, cfg.model.input_size)]
Dataloader_function = {'VOC': VOCDetection, 'COCO': COCODetection}

priorbox = PriorBox(anchor_config)
with torch.no_grad():
    priors = priorbox.forward()
    if cfg.test_cfg.cuda:
        priors = priors.cuda()

def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    num_images = len(testset)
    print('=> Total {} images to test.'.format(num_images))
    num_classes = getattr(cfg.model.num_classes, args.dataset)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    tot_detect_time = 0
    tot_nms_time = 0
    print('Begin to evaluate')
    for i in tqdm(range(num_images)):
        img = testset.pull_image(i)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        with torch.no_grad():
            x = transform(img).unsqueeze(0)
            if cuda:
                x = x.cuda()
                scale = scale.cuda()

        _t['im_detect'].tic()
        out = net(x)
        boxes, scores = detector.forward(out,priors)
        detect_time = _t['im_detect'].toc()

        boxes = boxes[0]
        scores=scores[0]
        boxes *= scale
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()

        _t['misc'].tic()
        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            
            soft_nms = cfg.test_cfg.soft_nms
            keep = nms(c_dets, cfg.test_cfg.iou, force_cpu=soft_nms)
            keep = keep[:cfg.test_cfg.keep_per_class]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        tot_detect_time += detect_time if i > 0 else 0
        tot_nms_time += nms_time if i > 0 else 0

        if i % 20 and False:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    print('Evaluating detections')
    testset.evaluate_detections(all_boxes, save_folder)

    print('Detect time per image: {:.3f}s'.format(tot_detect_time / (num_images-1)))
    print('Nms time per image: {:.3f}s'.format(tot_nms_time / (num_images - 1)))
    print('Total time per image: {:.3f}s'.format((tot_detect_time + tot_nms_time) / (num_images - 1)))
    print('FPS: {:.3f} fps'.format((num_images - 1) / (tot_detect_time + tot_nms_time)))


if __name__ == '__main__':

    num_classes = getattr(cfg.model.num_classes, args.dataset)
    net = build_net('test', 
                    cfg = cfg.model,
                    num_classes=num_classes)
    state_dict = torch.load(args.trained_model)
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
    net.eval()
    print('Finished loading model!')
    _Dataloader_function = Dataloader_function[args.dataset]
    testset = _Dataloader_function(cfg.COCOroot if args.dataset == 'COCO' else cfg.VOCroot,
                                   getattr(cfg.dataset, args.dataset)['eval_sets'], # change to test_sets for MS-COCO if want test-dev
                                   None)
    if cfg.test_cfg.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    else:
        net = net.cpu()
    detector = Detect(num_classes, cfg.loss.bkg_label, anchor_config)
    save_folder = os.path.join(cfg.test_cfg.save_folder, args.dataset)
    _preprcess = BaseTransform(cfg.model.input_size, cfg.model.rgb_means, (2, 0, 1))

    test_net(save_folder, 
             net, 
             detector, 
             cfg.test_cfg.cuda, 
             testset, 
             transform = _preprcess, 
             max_per_image = cfg.test_cfg.topk, 
             thresh = cfg.test_cfg.score_threshold)
