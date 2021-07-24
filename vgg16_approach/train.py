import config
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pandas as pd

# in this project I have manually downloaded the test images, so there is no need to split the images
# but for the test portion the process has to repeat again... so in this code i'm using train_test_split
# in the notebook I have done the laborous "repeated" process for the test data
# mainly the whole process is done in the notebook section 
# but first I wrote all the codes here 

data,test_data = [] # images 
bboxes = [] # targets
filenames = [] # for making test files inside this dataset 



persons = pd.read_csv(config.annots_path)
test_persons =pd.read_csv(config.annots_test_path)

for i in persons.image:
    img = load_img(os.path.join(config.base_path, str(i) + '.jpg'), target_size=(224,224))
    img = img_to_array(img)

    data.append(img)
    filenames.append(i)

tmp = []

for i in range(len(persons)):
    for j in range(1,5):
        tmp.append(persons.iloc[i,j])

    bboxes.append(tmp)
    tmp = []

# convert the data and labels to NumPy arrays, scaling the input
# pixel intensities from the range [0, 255] to [0, 1]
data = np.array(data, dtype="float32") / 255.0
bboxes = np.array(bboxes, dtype='float32')

#using train_test_split
train_images, test_images, train_bboxes, test_bboxes, train_filenames, test_filenames = train_test_split(data, bboxes,filenames, test_size=0.10, random_state= 42)


################ vgg16 starts here ##################
## this is VGG16 with pre-trained "ImageNet" weights >> we are fine tuning it for getting our purposes

# load the VGG16 network, ensuring the head FC layers are left off
vgg = VGG16(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# freeze all VGG layers so they will *not* be updated during the
# training process
vgg.trainable = False

# flatten the max-pooling output of VGG
flatten = vgg.output
flatten = Flatten()(flatten)

# construct a fully-connected layer header to output the predicted
# bounding box coordinates
bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid")(bboxHead)

# construct the model we will fine-tune for bounding box regression
model = Model(inputs=vgg.input, outputs=bboxHead)

# initialize the optimizer, compile the model, and show the model
# summary
opt = Adam(learning_rate=1e-4)
model.compile(loss="mse", optimizer=opt, metrics= ['accuracy'])
print(model.summary())

epoch = 25
batch = 32
# train the network for bounding box regression
print("[INFO] training bounding box regressor...")
H = model.fit(
	train_images, train_bboxes,
	validation_data=(test_images, test_bboxes),
	batch_size=batch,
	epochs=epoch,
	verbose=1)

# serialize the model to disk
print("[INFO] saving object detector model...")
model.save(config.MODEL_PATH, save_format="h5")

print('all variables from the training history')
print(H.history.split())

# plot the model training history loss & accuracy
N = epoch
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="upper right")
plt.savefig(config.loss_plot_path)
plt.show()
plt.close()

#creating another figure for accuraies
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")
plt.title("Bounding Box Regression Accuracy on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.savefig(config.loss_plot_path)
plt.show()
plt.close()




