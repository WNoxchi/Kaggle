import cv2
import numpy as np
from fastai.vision.image import Image, pil2tensor
# from fastai.vision.image import *

# adapted from https://www.kaggle.com/iafoss/pretrained-resnet34-with-rgby-0-460-public-lb
def open_4_channel(fname):
    fname = str(fname)
    # strip extension before adding color
    if fname.endswith('.png'):
        fname = fname[:-4]
    colors = ['red','green','blue','yellow']
    flags = cv2.IMREAD_GRAYSCALE
    img = [cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
           for color in colors]    
    x = np.stack(img, axis=-1)
    return Image(pil2tensor(x, np.float32).float())

## DEBUGGING
# def open_4_channel(fname):
#     fname = str(fname)
#     # strip extension before adding color
#     if fname.endswith('.png'):
#         fname = fname[:-4]
#     colors = ['red','green','blue','yellow']
#     flags = cv2.IMREAD_GRAYSCALE
#     img = []
#     for color in colors:
#         try:
#             im = cv2.imread(fname+'_'+color+'.png', flags).astype(np.float32)/255
#             img.append(im)
#         except AttributeError:
#             print(f"color: {color}")
#             print(f"fname: {fname}")
#             print(f"flags: {flags}")
#             print(f"Error: OpenCV was unable to open: {fname+'_'+color+'.png'}")
#     x = np.stack(img, axis=-1)
#     return Image(pil2tensor(x, np.float32).float())