import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from datetime import datetime
import glob
import argparse
import os
import numpy as np


os.chdir('resized_images')
image_names = glob.glob('*.jpg')
image_names.sort()


def tf_method(imgNumber):
    '''tf_method is used for loading the images by tf libraries and check the time'''
    start_time = datetime.now()
    imgdata = []
    for i in range(imgNumber):
        img = load_img(image_names[i])
        img = img_to_array(img)
        imgdata.append(img)
    print(len(imgdata))
    imgdata = np.array(imgdata)
    print(imgdata.shape)
    del imgdata
    total_time = datetime.now() - start_time
    print(total_time)

def cv2_method(imgNumber):
    start_time  = datetime.now()
    imgdata = []
    for i in range(imgNumber):
        img = cv2.imread(image_names[i])
        imgdata.append(img)
    print(len(imgdata))
    imgdata = np.array(imgdata)
    print(imgdata.shape)
    del imgdata
    total_time = datetime.now() - start_time
    print(total_time)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--function', required=True, help= 'tf_method & cv2_method')
    ap.add_argument('-n','--imgNumber', required=True, help= 'how much image are we gonna calculate')
    args = vars(ap.parse_args())

    if args['function'] == 'tf_method': 
        tf_method(int(args['imgNumber']))
    if args['function'] == 'cv2_method':
        cv2_method(int(args['imgNumber']))

################## results #################3
'''
for 2k images with tf method
2000
(2000, 224, 224, 3)
0:00:07.044477

for 2k images with cv2 method 
2000
(2000, 224, 224, 3)
0:00:02.416770

for 4k image >> tf failed because it took much larger RAM then the cv2
'''