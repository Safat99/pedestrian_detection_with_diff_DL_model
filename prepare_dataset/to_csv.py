#to make a dataframe >>
#assign data to lists
#and columns will be the keys of the dictionary and lists will be the value


##txt file gulake pandas diye aramse read kore kaaj kora jaay >> test01 code e tai kora hoise 
# for i in names:
#     df = pd.read_csv(i, delimiter=' ', names=['class', 'name', 'left', 'top','right','bottom'])

'''
to append a row inside pandas dataframe 
tmp = [[,,,,,]]
df = df.append(pd.DataFrame(tmp, columns=[,,,,]), ignore_index = True)
'''

import glob
import pandas as pd
import cv2

names = glob.glob('*.txt')
pics = glob.glob('*.jpg')
names.sort()
pics.sort()

list = []
image =[]

## all txt files will be openend to make the dataframe
for count,i in enumerate(names):
    with open(i, 'r') as f:
        for j in f:
            j=j[:-1]
            list.append(j.split(' '))
            image.append(count)

df = pd.DataFrame(list, columns=['label','label2','left', 'top', 'right', 'bottom'])
df['image'] = image # image add korlam 
df['left'] = df['left'].astype(float) 
df['bottom'] = df['bottom'].astype(float) 
df['right'] = df['right'].astype(float) 
df['top'] = df['top'].astype(float) 

# since 'human body' has to be at the same column 
df['label'] = df['label'].astype(str) + ' ' + df['label2'].astype(str) 
del df['label2']

#rearranging the order of the columns
df = df[['image', 'left', 'top', 'right', 'bottom', 'label']]

#dataFrame to CSV
df.to_csv('all_informations.csv',index=False)


####################### now it's time to normalize the bounding box's values ######################

#equivalent to arduino's map function
def _map(x, in_min, in_max, out_min, out_max):
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


#in the pandas documentation they suggest to use iloc instead of direct dictionary access
'''
for i in range(df.shape[0]):
    img = cv2.imread(str(df['image'][i]) + '.jpg', 0)
    h,w = img.shape
    df['left'][i] = df['left'][i] / w
    df['right'][i] = df ['right'][i] / w
    df['top'][i] = df ['top'][i] / h
    df['bottom'][i] = df ['bottom'][i] / h 
'''
for i in range(df.shape[0]):
    img = cv2.imread(str(df['image'][i]) + '.jpg', 0)
    h,w = img.shape
    df.iloc[i,1] = df.iloc[i,1] / w
    df.iloc[i,2] = df.iloc[i,2] / h
    df.iloc[i,3] = df.iloc[i,3] / w
    df.iloc[i,4] = df.iloc[i,4] / h
     
df.to_csv('all_informations_normalized.csv', index=False)











