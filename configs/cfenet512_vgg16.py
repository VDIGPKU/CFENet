model = dict(
    type = 'cfenet',
    input_size = 512,
    backbone = 'vgg',
    resume_net = True,
    pretrained = 'weights/vgg16_reducedfc.pth',
    CFENET_CONFIGS = {
        'maps': 7,
        'lat_cfes': 2,
        'channels': [512, 1024, 512, 512, 256, 256, 256],
        'ratios': [6, 6, 6, 6, 6, 4, 4],
    },
    backbone_out_channels = (512, 1024, 1024),
    rgb_means = (104, 117, 123),
    p = 0.6,
    num_classes = dict(
        VOC = 21,
        COCO = 81, # for VOC and COCO
        ),
    save_eposhs = 10,
    weights_save = 'weights/'
    )

train_cfg = dict(
    cuda = True,
    warmup = 5,
    per_batch_size = 8, # for 4 gpus
    init_lr = 0.002,
    gamma = 0.1,
    end_lr = 1e-6,
    step_lr = dict(
        VOC = [90, 120, 140, 160],
        COCO = [150, 200, 250, 300],
        ),
    print_epochs = 10,
    num_workers= 8,
    )

test_cfg = dict(
    topk = 0,
    iou_jaccard = 0.45,
    soft_nms = True,
    score_threshold = 0.1,
    save_folder = 'eval'
    )
loss = dict(overlap_thresh = 0.5,
            prior_for_matching = True,
            bkg_label = 0,
            neg_mining = True,
            neg_pos = 3,
            neg_overlap = 0.5,
            encode_target = False)

optimizer = dict(type='SGD', momentum=0.9, weight_decay=0.0005)

dataset = dict(
    VOC = dict(
        train_sets = [('2007', 'trainval'), ('2012', 'trainval')],
        test_sets = [('2007', 'test')],
        ),
    COCO = dict(
        train_sets = [('2014', 'train'), ('2014', 'valminusminival')],
        eval_sets = [('2014', 'minival')],
        test_sets = [('2015', 'test-dev')]
    )
)
import os
home = os.path.expanduser("~")
VOCroot = os.path.join(home,"data/VOCdevkit/")
COCOroot = os.path.join(home,"data/coco/")
