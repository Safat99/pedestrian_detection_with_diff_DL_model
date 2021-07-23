import argparse
import cv2
# import imutils
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True, help='name of the input image')
ap.add_argument('-t', '--txt', required=True, help='name of the text file')
args = vars(ap.parse_args())

input_image = args['input']
input_text = args['txt']

img = cv2.imread(input_image)
imgcopy = img

df = pd.read_csv(input_text, delimiter=' ', names=['label', 'left', 'top', 'right', 'bottom'])
# fixed width formatted lines >> pd.read_fwf 

for i in range(df.shape[0]):
    cv2.rectangle(img, (int(df.left[i]), int(df.top[i])),(int(df.right[i]), int(df.bottom[i])), (0,255,0),2)
    cv2.putText(img, df.label[i], (int(df.left[i]), int(df.top[i])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)


cv2.imshow('input', img)
cv2.imshow('output',imgcopy)

cv2.waitKey(0)
cv2.destroyAllWindows()
