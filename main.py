#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 08:16:43 2017

@author: adioshun
"""


from settings import TEST_VIDEO, RESULT_VIDEO
from slidingWindowSearch import pred_window


from moviepy.editor import VideoFileClip
from moviepy.editor import CompositeVideoClip


clip1 = VideoFileClip(TEST_VIDEO)
vid_clip = clip1.fl_image(pred_window) #NOTE: this function expects color images!!
vid_clip.write_videofile(RESULT_VIDEO, audio=False)

