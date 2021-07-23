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

img = cv2.imread(input_image, 0)
h,w = img.shape[:2]
cv2.imshow('input', img)

img = cv2.resize(img, (224,224))

df = pd.read_csv(input_text, delimiter=' ', names=['label', 'left', 'top', 'right', 'bottom'])
# fixed width formatted lines >> pd.read_fwf 

#equivalent to arduino's map function
def _map(x, in_min, in_max, out_min, out_max):
    return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min) 

for i in range(df.shape[0]):
    df.left[i] = _map(df.left[i], 0, w, 0, 224)
    df.right[i] = _map(df.right[i], 0, w, 0, 224)
    df.top[i] = _map(df.top[i], 0, h, 0, 224)
    df.bottom[i] = _map(df.bottom[i], 0, h, 0, 224)


for i in range(df.shape[0]):
    cv2.rectangle(img, (int(df.left[i]), int(df.top[i])),(int(df.right[i]), int(df.bottom[i])), (0,255,0),2)
    cv2.putText(img, df.label[i], (int(df.left[i]), int(df.top[i])+10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)



cv2.imshow('output',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
