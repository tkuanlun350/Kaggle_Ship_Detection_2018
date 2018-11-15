# Faster-RCNN / Mask-RCNN
A minimal multi-GPU implementation of Faster-RCNN / Mask-RCNN (without FPN).
Modified from https://github.com/ppwwyyxx/tensorpack/tree/master/examples/FasterRCNN.


## Dependencies
+ Python 3; TensorFlow >= 1.4.0
+ pip install git+git://github.com/waspinator/pycococreator.git@0.2.0
+ pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
+ Tensorpack@0.8.5 (https://github.com/tensorpack/tensorpack) (pip install -U git+https://github.com/ppwwyyxx/tensorpack.git@0.8.5)
+ OpenCV
+ Pre-trained [ResNet model](https://goo.gl/6XjK9V) from tensorpack model zoo.
+ OAR data. It assumes the following directory structure:
```
ct_segmentation_data_fine/
  train/*.pkl
  test/*.pkl
  val/*.pkl
  processed_image/
    /*.png
  fusion/
    /*.png
```

### File Structure
This is a minimal implementation that simply contains these files:
+ `preprocess.py`: preprocess data and save png to the directory `processed_image`. If mri image is available, images will be saved to the directory `fusion`.
+ `config.py`: configuration for all
+ `oar.py`: load OAR data
+ `data.py`: prepare data for training
+ `common.py`: common data preparation utilities
+ `basemodel.py`: implement resnet
+ `model.py`: implement rpn/faster-rcnn/mask-rcnn
+ `train.py`: main training script
+ `utils/`: third-party helper functions
+ `eval.py`: evaluation utilities
+ `viz.py`: visualization utilities

## Usage
Change config in `config.py`:
1. Set `MODE_MASK` to switch Faster-RCNN or Mask-RCNN.

Train:
```
./train.py --load /path/to/ImageNet-ResNet50.npz
```
The code is only for training with 1, 2, 4 or 8 GPUs.
Otherwise, you probably need different hyperparameters for the same performance.

Predict on an image (and show output in a window):
```
./train.py --predict input.jpg --load /path/to/model
```

### Implementation Notes

Data:

1. You can easily add more augmentations such as rotation, but be careful how a box should be
	 augmented. The code now will always use the minimal axis-aligned bounding box of the 4 corners,
	 which is probably not the optimal way.
	 A TODO is to generate bounding box from segmentation, so more augmentations can be naturally supported.

Model:

1. Floating-point boxes are defined like this:

<p align="center"> <img src="https://user-images.githubusercontent.com/1381301/31527740-2f1b38ce-af84-11e7-8de1-628e90089826.png"> </p>

2. We use ROIAlign, and because of (1), `tf.image.crop_and_resize` is __NOT__ ROIAlign.

3. We only support single image per GPU.

4. Because of (3), BatchNorm statistics are not supposed to be updated during fine-tuning.
	 This specific kind of BatchNorm will need [my kernel](https://github.com/tensorflow/tensorflow/pull/12580)
	 which is included since TF 1.4. If using an earlier version of TF, it will be either slow or wrong.

Speed:

1. The training will start very slow due to convolution warmup, until about 3k steps to reach a maximum speed.
	 Then the training speed will slowly decrease due to more accurate proposals.

2. Inference is not quite fast, because either you disable convolution autotune and end up with
	 a slow convolution algorithm, or you spend more time on autotune.
	 This is a general problem of TensorFlow when running against variable-sized input.

3. With a large roi batch size (e.g. >= 256), GPU utilitization should stay above 90%.

