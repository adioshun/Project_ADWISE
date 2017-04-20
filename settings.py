from datetime import datetime

PROJECT_DIR = "/home/adioshun/Github/Project_ADWISE/"
TRAINDATA_DIR = "/home/adioshun/datasets/"
TESTDATA_DIR = "/home/adioshun/Github/Project_ADWISE/testData/"
INPUTDATA_DIR = "/home/adioshun/Github/Project_ADWISE/inputData/"
RESULT_DIR = "/home/adioshun/Github/Project_ADWISE/resultData/"

CV_CLF_DUMP = INPUTDATA_DIR+'cv_clf_'+datetime.now().strftime('%Y%m%d')+'.pkl'



TEST_VIDEO = TESTDATA_DIR+'project_video.mp4'#"test_video.mp4"


RESULT_VIDEO = RESULT_DIR+'Resut_video_'+datetime.now().strftime('%Y%m%d')+'.mp4'