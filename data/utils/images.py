import numpy as np
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd

def crop_image(img, save_path):
    # when taking a look at the images it becomes evident that all images have a unnecessary white border
    # this function crops the images to the smallest possible size

    old_im = Image.open(img).convert('L') # loads image in grayscale (L mode)
    img_data = np.asarray(old_im, dtype=np.uint8) # convert to numpy array
    nnz_inds = np.where(img_data!=255) # returns tupel ([...], [...]) of indices where img_data is not 255
    if len(nnz_inds[0]) == 0:
        old_im.save(save_path)
        return
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
    old_im.save(save_path)


def resize_image(img, save_path, width, height):
    img=cv2.imread(img)
    img_height, img_width=img.shape[0],img.shape[1]

    if img_height / img_width < height / width:
        new_height = height / width * img_width
        pad = (new_height - img_height) / 2
        img_padded= cv2.copyMakeBorder(img, math.ceil(pad), math.floor(pad), 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    elif img_height / img_width > height / width:
        new_width = img_height * width / height
        pad = (new_width - img_height)
        img_padded= cv2.copyMakeBorder(img, 0, 0 , math.ceil(pad),math.floor(pad), cv2.BORDER_CONSTANT, value=[255, 255, 255])

    else:
        img_padded = img

    img_rescaled = img_padded #img_padded / (255 / 2) - 1 # Rescale to [-1, 1]
    img_resize = cv2.resize(img_rescaled, [width, height])
    cv2.imwrite(save_path, img_resize)


