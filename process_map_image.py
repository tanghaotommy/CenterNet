import numpy as np
import os
import matplotlib.pyplot as plt
import json
import cv2
from collections import defaultdict
import itertools
from tqdm import tqdm
import subprocess
import argparse
import shutil
import sys


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--img-path', metavar='N', type=str, help='test image path', default='data/house_detection/ExportE364N085.jpg')


def preprocess_test_image(img_path, save_dir, reso=512):
    print('Loading image from {}'.format(img_path))
    image = cv2.imread(img_path)

    # flip the image to match the annotation coordinate

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = np.flip(image, axis=0)
    image = np.ascontiguousarray(image)
    
    H, W, C = image.shape
    ys = list(range(0, H // reso + 1))
    xs = list(range(0, W // reso + 1))
    it = itertools.product(ys, xs)

    t = 0
    fid = img_path.split('/')[-1].split('.')[0]
    for cnt, (i, j) in tqdm(enumerate(it), total=(len(ys) - 1) * (len(xs) - 1), desc=fid):
        y_s, y_e = i * reso, (i + 1) * reso
        x_s, x_e = j * reso ,(j + 1) * reso
        # print((x_s, y_s), (x_e, y_e))
        crop = image[y_s:y_e, x_s:x_e]
       # need to pad
        if crop.shape[0] < reso or crop.shape[1] < reso:
            pad = [[0, reso - crop.shape[0]], [0, reso - crop.shape[1]], [0, 0]]
            crop = np.pad(crop, pad, constant_values=0)
        elif crop.shape[0] > reso:
            raise NotImplementedError
        cv2.imwrite(os.path.join(save_dir, '{}_{:04d}_{:04d}.jpg'.format(fid, x_s, y_s)), crop)
        np.save(os.path.join(save_dir, '{}_{:04d}_{:04d}.npy'.format(fid, x_s, y_s)), {'bboxes': [[0, 0, 1, 1, 1]]})


def main():
    reso = 512
    args = parser.parse_args()
    img_path = args.img_path
    save_dir = 'data/house_detection/preprocess/test'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    preprocess_test_image(img_path, save_dir)


if __name__ == "__main__":
    main()