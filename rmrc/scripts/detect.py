import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import sys
import cv2

img_width, img_height = 224, 224

top_model_weights_path = 'bottleneck_fc_model.h5'
optimised_weights_path = 'weights-improvement-18-0.87.hdf5'
confusion_weights_path = 'confusion26.hdf5'
train_data_dir = 'train/'
validation_data_dir = 'test/'

epochs = 50
batch_size = 16

confusion_labels=['dangerous', 'flammableliquid', 'flammablesolid', 'organicperoxide', 'spontaneous', 'oxidizer']

def predict(image_path):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    #image_path = str(sys.argv[1])

    #orig = cv2.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(optimised_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    return label

    # get the prediction label
   # print("Image ID: {}, Label: {}".format(inID, label))

    # display the predictions with the image
    #cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
    #            cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    #cv2.imshow("Classification", orig)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def predict_confusion(image_path):
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('class_indices_confusion.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    #image_path = str(sys.argv[1])

    #orig = cv2.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(confusion_weights_path)

    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    return label

def main():
    #image=cv2.VideoCapture('http://192.168.1.212:8080')
    #success, img=image.read()
    query_path=str(sys.argv[1])
    img=cv2.imread(query_path)
    rows, cols, channels = img.shape
    row4=int(rows/2)
    col4=int(cols/2)
    labels=[]

    for i in range(2):
        for j in range(2):
            new = img[i*row4:(i+1)*row4, j*col4:(j+1)*col4]
            cv2.imwrite("single"+str(i)+str(j)+".jpg", new)
            label=predict("single"+str(i)+str(j)+".jpg")
            #if label in confusion_labels:
                #print('CONFUSION')
                #label=predict_confusion("single"+str(i)+str(j)+".jpg")
            labels.append(label)
            
    labels=np.array(labels)
    labels=labels.reshape((2,2))
    print (labels)

main()
