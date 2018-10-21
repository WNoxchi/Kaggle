import torch
import torchvision
import numpy as np
import cv2

import matplotlib.pyplot as plt

fname = 'scifigun.jpg'
image = cv2.imread(fname, cv2.COLOR_BGR2RGB)
cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()