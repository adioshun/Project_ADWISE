import pandas as pd
import numpy as np
import cv2
import matplotlib.image as mpimg
from datetime import datetime
from sklearn import svm
from sklearn.externals import joblib
from tqdm import tqdm #pregress bar
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def hog_features(img, cspace='gry', chan=0, orientbin=9, cellpix=8, cellb=2, visual=False, vector=True):
    
    # Pick feature and channel to perform HOG on.
    if cspace == 'rgb':
        newimg = img[:,:,chan]
    if cspace == 'gry':
        newimg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if cspace == 'hsv':
        tmpimg = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        newimg = tmpimg[:,:,chan]
    if cspace == 'luv':
        tmpimg =  cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        newimg = tmpimg[:,:,chan]
    if cspace == 'hls':
        tmpimg = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        newimg = tmpimg[:,:,chan]
    if cspace == 'yuv':
        tmpimg = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        newimg = tmpimg[:,:,chan]    
    
    if visual == True:
        features, visual_image = hog(newimg, orientations=orientbin,
                                     pixels_per_cell=(cellpix, cellpix),
                                     cells_per_block=(cellb, cellb),
                                     transform_sqrt=True,
                                     visualise=True, feature_vector=True)
        return features, visual_image
    else:
        features = hog(newimg, orientations=orientbin,
                       pixels_per_cell=(cellpix, cellpix),
                       cells_per_block=(cellb, cellb),
                       transform_sqrt=True,
                       visualise=False, feature_vector=True)
        return features


#%%

df_vehicles = pd.read_csv("/home/adioshun/Github/Project_ADWISE/trainData/vehicle/df_vehicles.csv")



xdata = np.zeros((len(df_vehicles),1764)).astype('float32') #1764 = feature length 
ydata =df_vehicles['Label'].as_matrix() #pandas to numpy


for i in tqdm(range(0,len(df_vehicles))):
    img = df_vehicles.ix[i,'File_Path']
    img = mpimg.imread(img)
    feat1 = hog_features(img, 'yuv', 0)
    #feat2 = np.append(feat1, hog_features(imgd, 'hsv', 2)) # add another feature 
    xdata[i] = feat1
 


# 특징값을 Normalize
X_scaler = StandardScaler().fit(xdata)
scaledx = X_scaler.transform(xdata)


# 데이터 분리 xdata = x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.2, random_state=1337)
#train_test_split: Split arrays or matrices into random train and test subsets


#%% SVC
linsvc = svm.LinearSVC(loss='squared_hinge',tol=1e-4, max_iter=1000) # SVC설정
clf = linsvc.fit(x_train, y_train) # fit이용 학습


joblib.dump(clf, 'clf_'+datetime.now().strftime('%Y%m%d')+'.pkl', compress=0)


   
# and later you can load it
#clf = joblib.load('filename.pkl')

