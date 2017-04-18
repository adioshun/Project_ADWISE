# ADWISE
Advanced Drive Warning-Information System for the Elderly

![](http://i.imgur.com/PrqGzAR.png)

## Folder 
- trainData 
	- vehicle
    	- [udacity](https://www.udacity.com/) : Dataset provided by Udacity [[vehicle]](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip), [[non-vehicle]](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) 
        - [CrowdAI](http://crowdai.com/) : Car, Truck, Pedestrian, [object-detection-crowdai.tar.gz(1.5G)](http://bit.ly/udacity-annoations-crowdai)
        - [Autti](http://autti.co/) : Car, Truck, Pedestrian, Streat Light, [object-dataset.tar.gz(3.3G)](http://bit.ly/udacity-annotations-autti)



## trainData

- vehicle/udacity: The dataset is a combination of KITTI vision benchmark suite and GTI vehicle image database. GTI car images are grouped into far, left, right, middle close.

- [vehicle/CrowdAI](https://github.com/udacity/self-driving-car/tree/master/annotations): xmin,xmax,ymin,ymax,frame,label,preview url for frame
	- The dataset includes driving in Mountain View California and neighboring cities during daylight conditions. It contains over 65,000 labels across 9,423 frames collected from a Point Grey research cameras running at full resolution of 1920x1200 at 2hz. The dataset was annotated by CrowdAI using a combination of machine learning and humans.

- [vehicle/Autti](https://github.com/udacity/self-driving-car/tree/master/annotations): frame,xmin,ymin,xmax,ymax,occluded,label,attributes (Only appears on traffic lights)
	- This dataset is similar to dataset 1 but contains additional fields for occlusion and an additional label for traffic lights. The dataset was annotated entirely by humans using Autti and is slightly larger with 15,000 frames.

> IMPORTANT: The xmin, xmax, ymin and ymax values were marked incorrectly in the Udacity data, so I corrected them. This correction can be found in the code block where data frames are defined. Further, as data from two sources are combined, the column names were modified to match.


## March Week 4 
- <del> Planning </del>
- <del> Device Purchase </del>

---

## April 
### Week 1
- <del> System Setup (OS, ROS, OpenCV, TensorFlow) : [[Ref]](https://github.com/adioshun/Project_ADWISE/wiki/System-Setting)</del>
- <del> OBD Foramt </del> 

### Week 2
- <del> OBD Installation </del>
- <del> Raspberry Installation </del>
- OBD Communication

### Week 3 
- <del> Raspberry Installation </del>
- Data Storage(file)
