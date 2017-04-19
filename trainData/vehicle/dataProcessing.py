# udacity data handling 1 
import os
import glob
import numpy as np
import pandas as pd
from itertools import chain
feed = lambda pattern, y: ((f, y) for f in glob.glob(pattern))

os.getcwd()
os.chdir("/home/adioshun/Github/Project_ADWISE/trainData/vehicle")
Allfilename = list(chain(feed("/home/adioshun/datasets/vehicle/udacity/vehicles/**/*.png",1),feed("/home/adioshun/datasets/vehicle/udcity/non-vehicles/**/*.png",0))) #total : 8,792


xdata = np.zeros((18458,3528)).astype('float32') #3528=len(feat2)

# udacity data handling 2

imagename = sorted(glob.glob('/home/adioshun/datasets/vehicle/udacity/non-vehicles/Extras/*.png')) #5068
imagename2 = sorted(glob.glob('/home/adioshun/datasets/vehicle/udacity/non-vehicles/GTI/*.png')) #3900
nonVehicle_imageName = np.asarray(imagename + imagename2) #total 8,968
nonVehicle_label = np.zeros(len(nonVehicle_imageName)).astype(int) #non Vehicle = 0
nonvehicle = np.column_stack((nonVehicle_imageName, nonVehicle_label))


foldername = sorted(glob.glob('/home/adioshun/datasets/vehicle/udacity/vehicles/*'))
for fold in foldername:
    vehicle_imageName = sorted(glob.glob(fold + '/*.png')) #5965

vehicle_imageName= np.array(vehicle_imageName) 

Vehicle_label = np.ones(len(vehicle_imageName)).astype(int) # vehicle =1
vehicle = np.column_stack((vehicle_imageName, Vehicle_label))



np_vehicles = np.concatenate((nonvehicle, vehicle), axis=0)

df_vehicles = pd.DataFrame(np_vehicles)
df_vehicles.columns =['File_Path','Label']

df_vehicles.to_csv("df_vehicles.csv", index = False)



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


import pandas as pd
dir = "/home/adioshun/datasets/vehicle/Autti/object-dataset"
df_files2 = pd.read_csv(dir+'/new_Autti_labels.csv', header=0)
# df_files2.label.unique()
df_vehicles2 = df_files2[(df_files2['label']==' car') | (df_files2['label']==' truck')].reset_index() #스트링 앞에 공백 추가
df_vehicles2 = df_vehicles2.drop('occluded', 1)
df_vehicles2 = df_vehicles2.drop('attributes', 1)
df_vehicles2 = df_vehicles2.drop('index', 1)
df_vehicles2['File_Path'] = dir + '/' +df_vehicles2['Frame']

### Combine data frames
df_vehicles = pd.concat([df_vehicles1,df_vehicles2]).reset_index()
df_vehicles = df_vehicles.drop('index', 1)
df_vehicles.columns =['File_Path','Frame','Label','ymin','xmin','ymax','xmax']
df_vehicles.head()