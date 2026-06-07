# CNN image regression: clock reading and triangle counting

Two image-regression tasks with small convolutional networks on synthetically
generated greyscale images. The datasets are drawn on the fly with PIL/ImageDraw
(720 distinct clock images, one per minute over 12 hours; random triangle images),
so no data files are stored.

## Tasks

- Clock reading: given an analogue clock-face image, predict the time as total
  minutes (0-719).
- Triangle counting: given an image with 0-20 triangles, predict the count.

## Clock task: circular target encoding

Time is circular: 11:59 and 12:00 are one minute apart, but the scalar encodings
719 and 0 are maximally far apart. A scalar-in-[0,1] target with a sigmoid output
cannot represent that wrap-around. The target is instead placed on the unit
circle as two values:

```
fraction   = total_minutes / 720
sin_target = sin(2*pi*fraction)
cos_target = cos(2*pi*fraction)
```

Standard MSE then works without a special circular loss, and minutes are recovered
at inference with `atan2(sin, cos)`. This encoding is the main design decision in
the task.

## Models

Clock: a 4-block CNN with BatchNorm after each convolution, AdaptiveAvgPool(4) to
retain some spatial information, Dropout(0.3) and weight decay for regularization,
a ReduceLROnPlateau scheduler, on-the-fly Gaussian-noise augmentation, and a
two-value (sin, cos) output with no sigmoid. The training set is 504 images.

Triangle counting: the same backbone with 3 convolution blocks and a single
sigmoid output (count normalized to [0, 1] by dividing by 20), trained with plain
MSE.

## Outputs

When run, the notebook produces training and validation loss curves,
predicted-vs-true clock grids, and error histograms. The committed notebook has
its cell outputs stripped; run it to regenerate them.

## Run

```bash
pip install torch torchvision matplotlib pillow numpy
jupyter notebook cnn_image_regression.ipynb
```
