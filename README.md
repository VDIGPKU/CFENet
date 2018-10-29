#### CFENet: Object Detection with Comprehensive Feature Enhancement Module

--------

we will release code after the paper publicly available, which is accepted by ACCV2018 recently.

##### Single Model Records(Non-ensemble results):

- 1st on ***UA-DETRAC***

- 6th on ***KITTI car detection***

- 2nd on ***WAD workshop***

- 1st on ***Visdrone object detection in videos***

- 6th on ***Wider Face for face detection***
- ***AP of 43.5 on COCO(ResNet101-CFENet512)***


#### Note that CFENet is only an efficient one-stage detector, which can achieve 23+fps on MS-COCO when single-scale inference(VGG-CFENet800).

Now, we have opened the [**working** branch](https://github.com/qijiezhao/CFENet/tree/working), we wish you can try to train it with different configurations, this can help us find BUGs.

What's more, we are training the referenced models recently(CFENet300-SEResNet50, CFENet512-VGG16), Once it finished, we will open it at master branch.
