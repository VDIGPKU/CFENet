# config.py
import math
import os.path

# gets home dir cross platform
home = os.path.expanduser("~")
ddir = os.path.join(home,"data/VOCdevkit/")

VOCroot = ddir
COCOroot = os.path.join(home,"data/coco/")


def mk_reg_layer_size(size, num_layer, size_the):
    reg_layer_size = []
    for i in range(num_layer + 1):
        size = math.ceil(size / 2.)
        if i >= 2:
            reg_layer_size += [size]
            if i == num_layer and size_the != 0:
                reg_layer_size += [size - size_the]
    return reg_layer_size

def mk_size(size, size_pattern):
    size_list = []
    for x in size_pattern:
        size_list += [round(x * size, 2)]
    return  size_list

def mk_as_ra(num):
    as_ra = []
    for _ in range(num-2):
        as_ra += [[2, 3]]
    as_ra += [[2], [2]]
    return as_ra

def mk_config(size, multiscale_size, size_pattern, step_pattern, num_reg_layer, param = 2):
    cfg = {}
    cfg['feature_maps'] = mk_reg_layer_size(size, num_reg_layer, param if size >= multiscale_size else 0)
    cfg['min_dim'] = size
    cfg['steps'] = step_pattern
    cfg['min_sizes'] = mk_size(multiscale_size, size_pattern[:-1])
    cfg['max_sizes'] = mk_size(multiscale_size, size_pattern[1:])
    cfg['aspect_ratios'] = mk_as_ra(num_reg_layer)
    cfg['variance'] = [0.1, 0.2]
    cfg['clip'] = True
    return cfg

step_pattern_300 = [8, 16, 32, 64, 100, 300]

step_pattern_512 = [8, 16, 32, 64, 128, 256, 512]

size_pattern_VOC_300 = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05]

size_pattern_COCO_300 = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]

size_pattern_VOC_512 = [0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05]

size_pattern_COCO_512 = [0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06]

VOC_300 = mk_config(300, 300, size_pattern_VOC_300, step_pattern_300, 6, 2)

VOC_512 = mk_config(512, 512, size_pattern_VOC_512, step_pattern_512, 7, 1)

COCO_300 = mk_config(300, 300, size_pattern_COCO_300, step_pattern_300, 6, 2)

COCO_512 = mk_config(512, 512, size_pattern_COCO_512, step_pattern_512, 7, 1)

CFENET_ANCHOR_PARAMS = {
    'VOC_300': VOC_300,
    'COCO_300': COCO_300,
    'VOC_512': VOC_512,
    'COCO_512': COCO_512,
}
