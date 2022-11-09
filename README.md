
Customized implementation of the [U-Net](https://arxiv.org/abs/1505.04597)

I've hacked the upstream of this fork to pieces to get this working for my one-time purposes, 
I don't suggest you use this for anything. Lots of options don't work with certain other options, so please check the code is doing what it's supposed to. Again, it's a bad idea to use this fork.

Changes from upstream:
 - support PNG palettes for image masks (resizing is not supported)
 - add a handful of commandline arguments to specify mask color mappings, validation/training directories, and more
 - arbitrary kwarg passing when torch.hub loading (eg, changing number of classes)
 - @mac-op - loading data directly from .npy files
 

## Quick start

there is no quick start, you will need to read the source code to get things set up

## Usage
**Note : Use Python 3.6 or newer**

```
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]
                [--bilinear] [--classes CLASSES] [--mapping MAPPING]
                [--data-dir DATA_DIR] [--device DEVICE] [--workers WORKERS]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
  --bilinear            Use bilinear upsampling
  --classes CLASSES, -c CLASSES
                        Number of classes
  --mapping MAPPING, -m MAPPING
                        JSON of (R,G,B) -> Class. Implies/overides -c
  --data-dir DATA_DIR, -d DATA_DIR
                        directory to find data in
  --device DEVICE
  --workers WORKERS
  --load-from-npy       Load image arrays from .npy files
```
