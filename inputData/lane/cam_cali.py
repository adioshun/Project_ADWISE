#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from settings import INPUTDATA_DIR

import glob
import numpy as np
import cv2

images = glob.glob(INPUTDATA_DIR+'lane/chessboard/calibration*.jpg')
base_objp = np.zeros((6 * 9, 3), np.float32)
base_objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []
shape = None

for imname in images:
    img = cv2.imread(imname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if shape is None:
       shape = gray.shape[::-1]

    #print('Finding chessboard corners on {}'.format(imname))
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret:
        objpoints.append(base_objp)
        imgpoints.append(corners)

np.save(INPUTDATA_DIR+'lane/objpoints', objpoints)
np.save(INPUTDATA_DIR+'lane/imgpoints', imgpoints)
np.save(INPUTDATA_DIR+'lane/shape', shape)
