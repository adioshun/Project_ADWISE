# udacity data handling 1 
import os
import glob
from itertools import chain
feed = lambda pattern, y: ((f, y) for f in glob.glob(pattern))

os.getcwd()
os.chdir("/home/adioshun/datasets/vehicle/")
Allfilename = list(chain(feed("udacity/vehicles/**/*.png",1),feed("udcity/non-vehicles/**/*.png",0))) #total : 8,792


xdata = np.zeros((18458,3528)).astype('float32') #3528=len(feat2)

# udacity data handling 2

imagename = sorted(glob.glob('/home/adioshun/datasets/vehicle/udacity/non-vehicles/Extras/*.png'))
imagename2 = sorted(glob.glob('/home/adioshun/datasets/vehicle/udacity/non-vehicles/GTI/*.png'))
nonVehicle_imageName = imagename + imagename2 #total 8,968


foldername = sorted(glob.glob('/home/adioshun/datasets/vehicle/udacity/vehicles/*'))
for fold in foldername:
    imagename2 = sorted(glob.glob(fold + '/*.png'))
    Vehicle_imageName= imagename + imagename2 #total 11,034
    

All = nonVehicle_imageName + Vehicle_imageName #total 20,002




# crowdai data handling

import pandas as pd

dir = "/home/adioshun/datasets/vehicle/CrowdAI/object-detection-crowdai"


## Read From file 1
df_files1 = pd.read_csv(dir+'/labels.csv', header=0)
df_files1.head()
df_vehicles1 = df_files1[(df_files1['Label']=='Car') | (df_files1['Label']=='Truck')].reset_index()
df_vehicles1 = df_vehicles1.drop('index', 1)
df_vehicles1['File_Path'] =  dir + '/' +df_vehicles1['Frame']
df_vehicles1 = df_vehicles1.drop('Preview URL', 1)
df_vehicles1.head()


# Autti data handling


dir = "/home/adioshun/datasets/vehicle/Autti/object-dataset"
df_files2 = pd.read_csv(dir+'/new_Autti_labels.csv', header=0)


df_vehicles2 = df_files2[(df_files2['Label']=='car') | (df_files2['Label']=='truck')].reset_index()
df_vehicles2 = df_vehicles2.drop('index', 1)
df_vehicles2 = df_vehicles2.drop('RM', 1)
df_vehicles2 = df_vehicles2.drop('ind', 1)


df_vehicles2['File_Path'] = dir + '/' +df_vehicles2['Frame']