from settings import PROJECT_DIR, TESTDATA_DIR, INPUTDATA_DIR, CV_CLF_DUMP

from get_classifier import hog_features

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os 
import pickle

os.chdir(PROJECT_DIR)

#from helper import draw_boxes



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



# %%



# %%

# Crop window area of image 
def crop_resize(img, window):
    
    crop = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
    
    resize = cv2.resize(crop, (64, 64))
    
    return resize

#%%
CV_CLF_DUMP = INPUTDATA_DIR+"cv_clf_20170419.pkl"

image = mpimg.imread(TESTDATA_DIR+'test5.jpg')
windows = slide_window(image,0.75)


featdata = np.zeros((len(windows),1764)).astype('float32')

# Loop and gather all features data for each cropped window.
for i in range(1,len(windows)):
    cropimg = crop_resize(image, windows[i])
    #feat = np.append(hog_features(cropimg, 'yuv', 0), hog_features(cropimg, 'hsv', 2))
    #featdata[i] = np.append(hog_features(cropimg, 'yuv', 0), hog_features(cropimg, 'hsv', 2))
    featdata[i] = hog_features(cropimg, 'hsv', 0)
    
del i
del cropimg

# Run classifier on all of the feature data.
X_scaler = StandardScaler().fit(featdata)   #Origin xdata, change to featdata
featscaled = X_scaler.transform(featdata)

#%%


with open(CV_CLF_DUMP, 'rb') as f:
    linsvc = pickle.load(f)


#%%

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


            
plt.imshow(image)

