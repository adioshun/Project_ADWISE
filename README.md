
<img src="https://docs.google.com/drawings/d/1xSuijiu-CvhSJvg79Gl82rKOAH6X8lPy-HQkul30kFs/pub?w=960&amp;h=720">

[System Diagram_Google Draw](https://docs.google.com/drawings/d/1xSuijiu-CvhSJvg79Gl82rKOAH6X8lPy-HQkul30kFs/edit?usp=sharing)

test

# ADWISE
Advanced Drive Warning-Information System for the Elderly



## Folder
- trainData
	- vehicle
	- trafficSign
	- trafficLight
- inputData
	- vehicle : cv_clf_*.pkl, df_vehicles.csv
	- lane : calibration 
	- trafficSign
	- trafficLight
- resultData(Symbol Link) : Video clips
- testData(Symbol Link) : Video clips for test

├── evaluation.py # evaluation.py

├── images # model architectures

│   ├── resnet.png

│   ├── vggnet5.png

│   └── vggnet.png

├── MNIST # mnist data (not included in this repo)
│   ├── t10k-images-idx3-ubyte.gz
│   ├── t10k-labels-idx1-ubyte.gz
│   ├── train-images-idx3-ubyte.gz
│   └── train-labels-idx1-ubyte.gz
├── model # model weights
│   ├── resnet.h5
│   ├── vggnet5.h5
│   └── vggnet.h5
├── model.py # base model interface
├── README.md
├── utils.py # helper functions
├── resnet.py
├── vgg16.py
└── vgg5.py
