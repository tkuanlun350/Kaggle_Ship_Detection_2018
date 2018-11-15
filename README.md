# Faster-RCNN / Mask-RCNN
Airbus Ship Detection Challenge 10th Code.
Code mostly improted from tensorpack (see below)
Modified from https://github.com/ppwwyyxx/tensorpack/tree/master/examples/FasterRCNN.


## Dependencies
+ Python 3; TensorFlow >= 1.4.0
+ pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
+ pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
+ Tensorpack@0.8.5 (https://github.com/tensorpack/tensorpack) (pip install -U git+https://github.com/ppwwyyxx/tensorpack.git@0.8.5)
+ OpenCV
+ Pre-trained [ResNet model](https://goo.gl/6XjK9V) from tensorpack model zoo.

## What Works
1. I implemented [Online-Hard-Example-Mining](https://arxiv.org/abs/1604.03540) selecting top 128 roi's for training

2. Multi-Scale Training: Since small object are hard for detection, I resize the image randomly from 1200 ~ 2000.
3. I used [softnms](https://arxiv.org/abs/1704.04503) for post-processing. I used mask's overlap instead of box overlap to rescore each instance to avoid dealing with the rotated box problem.
4. MaskRCNN is prone to overfitting (The lesson learned from DSB2018). Data augmentation with MotionBlur, GaussNoise, ShiftScale, Rotate, CLAHE, RandomBrightness, RandomContrast ...
5. Enlarge mask crop from 28 to 56, using diceloss + BCE loss.

## Doesn't Work ##
1.  Add stride 2 featuremap in FPN and add size 16 anchor. Improve local cv to 0.61 but worse public LB and private LB.
2. [Cascade-RCNN][https://arxiv.org/abs/1712.00726]. No improvement.

I have implemented TTA and checkpoint ensemble (see eval.py) but all results in worse public LB. Turns out they are better in private LB (best 0.83). The final score are based on single model without TTA.
