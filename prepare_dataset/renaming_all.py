import glob
import os

names = glob.glob('*.txt')
pics = glob.glob('*.jpg')
names.sort()
pics.sort()

for i in range(len(names)):
    txt = str(i) + '.txt'
    jpg = str(i) + '.jpg'

    os.rename(names[i], txt)
    os.rename(pics[i], jpg)
    