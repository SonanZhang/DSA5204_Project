import torch
import numpy as np
import cv2
from pathlib import Path
import os
from Model import Yolo

def test(pathname):
    imgList = os.listdir(pathname)
    imgList.remove('.DS_Store')
    size = len(imgList)
    arr = np.zeros((size, 3, 416, 416))
    index = 0
    for filename in imgList:
        img = cv2.imread(pathname + '/' + filename)
        img = cv2.resize(img, (416, 416))
        B, G, R = cv2.split(img)
        img = np.array([R, G, B])
        arr[index] = img
        index += 1
    arr = torch.from_numpy(arr)
    arr = arr.float()
    num_classes = 20
    Model = Yolo(num_classes=num_classes)
    img_size = 416
    out = Model(arr)
    assert out[0].shape == (50, 3, img_size//32, img_size//32, 5 + num_classes)
    assert out[1].shape == (50, 3, img_size//16, img_size//16, 5 + num_classes)
    assert out[2].shape == (50, 3, img_size//8, img_size//8, 5 + num_classes)

pathname = ''
test(pathname)