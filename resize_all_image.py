# this code will run before one directory of the train image folder (2k data)
# resizing 224,224 to all the images
# img_dir needed to be changed according to folder
# !mkdir resize_dir >> name

## the main operation of this code is to imwrite all the images on the resize dir and see time


import os
import glob
import cv2
from datetime import date, datetime

start_time = datetime.now()
base_dir = os.getcwd()
resize_dir = os.path.join(base_dir, 'resized_images')
img_dir = 'train/Human body/'
os.chdir(img_dir)
image_names = glob.glob('*.jpg')
image_names.sort()
count = 0

for i in range(len(image_names)):
    img = cv2.imread(image_names[i])
    img = cv2.resize(img,(224,224))
    cv2.imwrite(os.path.join(resize_dir, str(i) + '.jpg'), img)
    count+=1
    print(count)

total_time = datetime.now() - start_time
print('total_time {}'.format(total_time))


############# results ######## 
#total_time 0:01:22.864677 -- >> for 2k higher dimensional images 
#total_time 0:04:13.289267 --- >>> for 8k higher dimensional images