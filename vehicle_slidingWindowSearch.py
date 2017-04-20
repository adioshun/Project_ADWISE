#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from vehicle_get_classifier import hog_features, get_classfier

import cv2
import numpy as np
import os

from sklearn.preprocessing import StandardScaler

#from helper import draw_boxes


#%%

def draw_pred_boxes(img, box, color=(0, 0, 255), thick=6):
    imcopy = np.copy(img)
    cv2.rectangle(imcopy, (box[0], box[1]), (box[2], box[3]), color, thick)
    return imcopy




#%%

# Return x and y coordinates for all computed window search positions.
def slide_window(img, xy_overlap = 0.5, xy_window = 64, y_start_stop=[380, None]):
# WuStangDan's CODE

    xcenter = np.int(img.shape[1]/2)

    # If y start/stop positions not defined, set to image size.
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # Initialize a list to append window positions to.
    window_list = []


    starty = y_start_stop[0]
    ny_window = 3
    growth = 1.104

    for yi in range(0,ny_window):
        step = np.int(xy_window *(1 - xy_overlap))
        endy = starty + xy_window
        nx_window = np.int(img.shape[1]/(step*2))
        if yi == 0:
            nx_window = nx_window - 1*np.int(xy_window/step)
            # How many xy_window tall pixels should be covered in the y direction.
            ystack = 2
            xy_overlap = 0.8
        if yi == 1:
            nx_window = nx_window - 0*np.int(xy_window/step)
            ystack = 2
            # Used for largest window, not current (yi == 2).
            xy_overlap = 0.8
        if yi == 2:
            nx_window = nx_window - 0*np.int(xy_window/step)
            ystack = 2

        # Ensures that both overlap and non-overlap cases cover the same number of pixels
        # in the y direction. Ex smallest boxes want to cover 2 * 64 pixels. With non-overlap
        # thats two sets of 64 pixel tall windows stacked on top of eachother. For 50% overlap
        # that requires 3 sets of 64 pixels tall windows with 50% overlap.
        if xy_overlap != 0:
            ystack =  np.int(ystack*(xy_window/step)) - 1

        for yii in range(ystack):
            for xi in range(nx_window):
                startx = xcenter + xi*step
                endx = startx + xy_window
                # Only add to list if within image bounds.
                if endx < img.shape[1]:
                    if startx > 0:
                        if endy < img.shape[0]:
                            window_list.append( ((startx, starty), (endx, endy)) )

                endx = xcenter - xi*step
                startx = endx - xy_window
                # Only add to list if within image bounds.
                if startx > 0:
                    if endx < img.shape[1]:
                        if endy < img.shape[0]:
                            window_list.append( ((startx, starty), (endx, endy)) )

            starty = starty + step
            endy = starty + xy_window

        starty = y_start_stop[0]
        xy_window = np.int(xy_window **growth)



    # Return the list of window positions.
    return window_list


#%%

def combine_pred(img, pred_windows):
    # Find overlaping or touching windows and group them.

    # If no prediction windows return original image.
    if pred_windows == None:
        return img, []

    # If only one prediction window return image with one low confidence box.
    if pred_windows.size == 4:
        #img = draw_pred_boxes(img, pred_windows, (255,0,255))
        return img, []

    prednum = np.zeros( (len(pred_windows),len(pred_windows)) )

    # Calculate centroids of each window.
    cent = []
    for window in pred_windows:
        xcent = np.int((window[0] + window[2])/2)
        ycent = np.int((window[1] + window[3])/2)

        cent.append((xcent, ycent))

    # Classify windows as a group if distance to another centroid is within set number
    # of pixels in x or y direction.
    for i in range(len(cent)):
        group = i+1

        # If this window already has an existing match
        # set its group number to that number.
        temppred = prednum[:,i]
        if len(temppred[temppred > 0]) > 0:
            group = temppred[temppred >0][0]

        for j in range(len(cent)):
            if i == j:
                continue

            # Compare x distance.
            if (abs(cent[i][0] - cent[j][0]) <= 100): # was 110
                # Compare y distance.
                if (abs(cent[i][1] - cent[j][1]) <= 64):
                    # If a match is found, check if the match is already assigned
                    # a group number.
                    temppred = prednum[:,j]
                    if len(temppred[temppred > 0]) > 0:
                        group = temppred[temppred > 0][0]

                    # Assign the group number to that location.
                    # i represents the window being compared and j represents
                    # the window that is found to match.
                    prednum[i,j] = group


    # Represent grouping of windows in 1D array.
    predgroup = np.zeros(len(cent))
    for i in range(len(cent)):
        predgroup[i] = max(prednum[i,:])

    #print('predgroup:',predgroup)


    # Draw pink boxes around all low confidence (single window prediction) detections.
#     singleloc = np.where(predgroup == 0)
#     if len(singleloc[0]) > 0:
#         for i in range(len(singleloc[0])):
#             img = draw_pred_boxes(img, pred_windows[singleloc[0][i]], (255,0,255))

    # Set group window to largest rectangle containing all the windows that group.
    groupwindoutput = []
    for i in range(1, len(cent)+1):
        grouploc = np.where(predgroup == i)
        grouplen = len(grouploc[0])
        if grouplen > 1:
            groupwind = np.array([img.shape[1], img.shape[0], 0, 0])
            for j in range(grouplen):
                windloc = grouploc[0][j]
                # Replace group window location with windows to create largest
                # encasing rectangle.
                if pred_windows[windloc][0] < groupwind[0]:
                    groupwind[0] = pred_windows[windloc][0]
                if pred_windows[windloc][1] < groupwind[1]:
                    groupwind[1] = pred_windows[windloc][1]
                if pred_windows[windloc][2] > groupwind[2]:
                    groupwind[2] = pred_windows[windloc][2]
                if pred_windows[windloc][3] > groupwind[3]:
                    groupwind[3] = pred_windows[windloc][3]

            # Draw box containing all prediction windows.
            if grouplen > 3:
                img = draw_pred_boxes(img, groupwind)
                groupwindoutput.append( ((groupwind[0], groupwind[1], groupwind[2], groupwind[3])) )
            #else:
                #img = draw_pred_boxes(img, groupwind, (255,0,255))
            #groupwindoutput.append( ((groupwind[0], groupwind[1], groupwind[2], groupwind[3])) )
    #return predgroup
    return img, groupwindoutput

# %%

# Need another function that when a car might have been detected, another focused sliding window search
# is performed on that specific area to have a better chance at detection.
# Return x and y coordinates for all computed window search positions.
def slide_window_focus(img, center, xy_overlap = 0.5):
    xcenter = center[0]

    # If y start/stop positions not defined, set to image size.
    xy_window = 32

    # Initialize a list to append window positions to.
    window_list = []
    ny_window = 5
    growth = 1.104

    for yi in range(0,ny_window):
        step = np.int(xy_window *(1 - xy_overlap))

        # Individual settings for each size of window.
        if yi == 0:
            nx_window = 0
            ystack = 0#3
        if yi == 1:
            nx_window = 1
            ystack = 2#2
        if yi == 2:
            nx_window = 1#3
            ystack = 2#2
        if yi == 3:
            nx_window = 1#2
            ystack = 0#2
        if yi == 4:
            nx_window = 1
            ystack = 0

        # Ensures that y distance is consistent with and without overlap.
        if xy_overlap != 0:
            ystack =  np.int(ystack*(xy_window/step)) - 1


        if ystack > 0:
            starty = np.int(center[1] - (xy_window))
        else:
            starty = np.int(center[1] - xy_window/2)
        endy = starty + xy_window

        # If only one window is meant to be created, window centered around
        # x and y centers.
        if nx_window == 1:
            startx = xcenter - np.int(xy_window/2)
            endx = startx + xy_window
            if startx > 0:
                if endx < img.shape[1]:
                    if starty > 0:
                        if endy < img.shape[0]:
                            window_list.append( ((startx, starty), (endx, endy)) )


        for yii in range(ystack):
            for xi in range(nx_window):
                startx = xcenter + xi*step
                endx = startx + xy_window
                # Only add to list if within image bounds.
                if endx < img.shape[1]:
                    if startx > 0:
                        if starty > 0:
                            if endy < img.shape[0]:
                                window_list.append( ((startx, starty), (endx, endy)) )

                endx = xcenter - xi*step
                startx = endx - xy_window
                # Only add to list if within image bounds.
                if startx > 0:
                    if endx < img.shape[1]:
                        if starty > 0:
                            if endy < img.shape[0]:
                                window_list.append( ((startx, starty), (endx, endy)) )

            starty = starty + step
            endy = starty + xy_window


        xy_window = np.int(xy_window **growth)



    # Return the list of window positions.
    return window_list


# %%

# Crop window area of image
def crop_resize(img, window):

    crop = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]

    resize = cv2.resize(crop, (64, 64))

    return resize

#%%


#%%

#class Car():
#    def __init__(self):
#        self.loc = []
#        self.allwind = []
#        self.count = -1
#
#


def vehicle_cv_detection(image):

#    Cars.count = Cars.count + 1
#    if ((Cars.count % 12) != 0):
#        for groups in Cars.loc:
#            image = draw_pred_boxes(image, groups)
#        return image


    # Use previous frame information about car detection to add more detection windows
    # in area where car was detected to increase chances of detecting it in this frame.
#    windgroup = []
#    for wind in Cars.loc:
#        xcenter = np.int((wind[0] + wind[2])/2)
#        ycenter = np.int((wind[1] + wind[3])/2)
#        windgroup.extend(slide_window_focus(img, [xcenter, ycenter]))
##         image = draw_boxes(image,windgroup)
#
    windows = slide_window(image,0.75)
#    windows.extend(windgroup)
#

    pred_windows = None


    featdata = np.zeros((len(windows),1764)).astype('float32')

    # Loop and gather all features data for each cropped window.
    for i in range(1,len(windows)):
        cropimg = crop_resize(image, windows[i])
        #feat = np.append(hog_features(cropimg, 'yuv', 0), hog_features(cropimg, 'hsv', 2))
        #featdata[i] = np.append(hog_features(cropimg, 'yuv', 0), hog_features(cropimg, 'hsv', 2))
        featdata[i] = hog_features(cropimg, 'hsv', 0)

    del i
    del cropimg


    linsvc = get_classfier()

    # Run classifier on all of the feature data.
    X_scaler = StandardScaler().fit(featdata)   #Origin xdata, change to featdata
    featscaled = X_scaler.transform(featdata)

    croppred = linsvc.predict(featdata) #Predict confidence scores for samples.


    pred_windows = None


    for i in range(len(windows)):
        # Decision function is used with a cut off 0.3 to filter out predictions that
        # the SVC isn't confident about.
        if croppred[i] >= .5:
            if pred_windows == None:
                pred_windows = np.hstack(windows[i])
                #image = draw_boxes(image, windows[i:i+1])
            else:
                temp = np.hstack(windows[i])
                pred_windows = np.vstack((pred_windows, temp)).astype(int)
                #image = draw_boxes(image, windows[i:i+1])


    imgpred, groupwind = combine_pred(image,pred_windows)

#    Cars.loc = groupwind
#    Cars.allwind = pred_windows
    return imgpred


#%%



#%%
