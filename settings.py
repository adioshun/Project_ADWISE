#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime

PROJECT_DIR = "/home/adioshun/Github/Project_ADWISE/"
TRAINDATA_DIR = "/home/adioshun/datasets/"
TESTDATA_DIR = "/home/adioshun/Github/Project_ADWISE/testData/"
INPUTDATA_DIR = "/home/adioshun/Github/Project_ADWISE/inputData/"
RESULT_DIR = "/home/adioshun/Github/Project_ADWISE/resultData/"

VEHICLE_CV_CLF_DUMP = INPUTDATA_DIR+'vehicle/vehicle_cv_clf.pkl'
VEHICLE_LIST_DUMP = INPUTDATA_DIR+'vehicle/vehicle_list.csv'


TEST_VIDEO = TESTDATA_DIR+'project_video.mp4'#"test_video.mp4"


RESULT_VIDEO = RESULT_DIR+'Resut_video_'+datetime.now().strftime('%Y%m%d')+'.mp4'
RESULT_VIDEO2 = RESULT_DIR+'Resut_video2_'+datetime.now().strftime('%Y%m%d')+'.mp4'