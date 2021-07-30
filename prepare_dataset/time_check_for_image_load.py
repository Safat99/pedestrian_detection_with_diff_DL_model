from datetime import datetime
from numpy.lib.npyio import load
import pandas as pd
import os
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img


df = pd.read_csv('all_informations_normalized.csv')
df = df.iloc[:,:1]
# print(df.head(5))
# print(df.tail(5))

def time_check_with_list():
    img_data = []
    start_time = datetime.now()
    for i in df['image']:
        image = load_img(str(i) + '.jpg', target_size=(224,224))
        image = img_to_array(image)
        img_data.append(image)
    img_data = np.array(img_data,dtype='float32')
    end_time = datetime.now()
    total_time = end_time - start_time
    print('time taking for list : {}'.format(total_time))
    print(img_data.shape)
    
    return total_time

def time_check_with_numpy():
    img_data_np = np.array([])
    start_time = datetime.now()
    for i in df['image']:
        image = load_img(str(i) + '.jpg', target_size=(224,224))
        image = img_to_array(image)
        img_data_np = np.append(img_data_np, image)

    tmp = img_data_np.shape[0]
    tmp = tmp / (224 * 224 * 3)
    img_data_np = np.reshape(img_data_np,(int(tmp),224,224,3))
    end_time = datetime.now()
    total_time = end_time - start_time
    print('time taking for numpy : {}'.format(total_time))
    print(img_data_np.shape)
    
    return total_time

def time_check_with_list_comprehension():
    start_time = datetime.now()
    def func(i):
        image = load_img(str(i) + '.jpg', target_size=(224,224))
        image = img_to_array(image)
        return image
    img_data_ls = [func(i) for i in df['image']]
    img_data_ls = np.array(img_data_ls,dtype='float32')
    end_time = datetime.now()
    total_time = end_time - start_time
    print('time taking for list comprehension : {}'.format(total_time))
    print(img_data_ls.shape)
    return total_time



if __name__ == '__main__':
    a = time_check_with_list()
    b = time_check_with_numpy()
    c = time_check_with_list_comprehension()
    print(a)
    print(b)
    print(c)
    print('difference is {} sec'.format(abs(a - b)))
    print('differenece of list vs list comprehension is {} sec'.format(abs(a-c)))


''' resutls>>> 
time taking for list : 0:00:00.191791
(27, 224, 224, 3)
time taking for numpy : 0:00:00.403852
(27, 224, 224, 3)
time taking for list comprehension : 0:00:00.193769
(27, 224, 224, 3)
0:00:00.191791
0:00:00.403852
0:00:00.193769
difference is 0:00:00.212061 sec
differenece of list vs list comprehension is 0:00:00.001978 sec
'''