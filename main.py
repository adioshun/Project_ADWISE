#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 08:16:43 2017

@author: adioshun
"""


from settings import TEST_VIDEO, RESULT_VIDEO, RESULT_VIDEO2
from vehicle_cv_detection import vehicle_cv_detector

from lane_cv_detection import lane_cv_detector

from moviepy.editor import VideoFileClip




clip = VideoFileClip(TEST_VIDEO)
lane_clip = clip.fl_image(lane_cv_detector) #NOTE: this function expects color images!!


clip = VideoFileClip(TEST_VIDEO)
vehicle_clip = clip.fl_image(vehicle_cv_detector) #NOTE: this function expects color images!!


vehicle_clip.write_videofile(RESULT_VIDEO2, audio=False)







# Clip videos together.
# clip1 = VideoFileClip("bin-opt.mp4")
# clip2 = VideoFileClip("bin-opt-raw.mp4")
# clip3 = clip2.resize(0.25)
# clip4 = VideoFileClip('peaks-opt.mp4')
# clip5 = clip4.resize(0.25)
# video = CompositeVideoClip([clip1,
#                            clip3.set_pos((480,360)),
#                            clip5.set_pos((480,180))])
# %time video.write_videofile('composite.mp4', audio=False)
