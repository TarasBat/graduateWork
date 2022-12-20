from tensorflow.python.keras.models import load_model
from keras_preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
import cv2
import numpy as np


def recognizeMask():
    # load face detector model
    prototxtPath = "graduate work/deploy.prototxt"
    weightsPath = "graduate work/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load prepared model for mask recognition
    imagePath = "image_examples/12.jpg"
    # modelPath = "results/CNN-size-64-bs-32-lr-0.0001.h5"
    modelPath = "results/VGG16-size-64-bs-32-lr-0.0001.h5"
    model = load_model(modelPath)
    testImage = cv2.imread(imagePath)

    (h, w) = testImage.shape[:2]
    blobFromImage = cv2.dnn.blobFromImage(testImage, scalefactor=1.0, mean=(104.0, 177.0, 123.0))

    net.setInput(blobFromImage)
    netForward = net.forward()


    for i in range(0, netForward.shape[2]):
        confidence = netForward[0, 0, i, 2]
        if confidence < 0.5:
            # Skip low confidence
            continue

        rectangle = netForward[0, 0, i, 3:7] * np.array([w, h, w, h])
        (start_X, start_Y, end_X, end_Y) = rectangle.astype("int")
        (start_X, start_Y) = (max(0, start_X), max(0, start_Y))
        (end_X, end_Y) = (min(w - 1, end_X), min(h - 1, end_Y))

        extractedFace = testImage[start_Y:end_Y, start_X:end_X]
        extractedFace = cv2.cvtColor(extractedFace, cv2.COLOR_BGR2RGB)
        extractedFace = cv2.resize(extractedFace, (64, 64))
        extractedFace = img_to_array(extractedFace)
        extractedFace = preprocess_input(extractedFace)
        extractedFace = np.expand_dims(extractedFace, axis=0)

        mask = model.predict(extractedFace)[0]

        label = ""
        color = (0, 0, 0)
        if mask < 0.5:
            label = "No Mask"
            color = (0, 0, 255)
        else:
            label = "Mask"
            color = (0, 255, 0)

        # Add rectangle with a tips
        cv2.putText(testImage, label, (start_X, start_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.rectangle(testImage, (start_X, start_Y), (end_X, end_Y), color, 2)


    cv2.imshow("Result", testImage)
    cv2.waitKey(0)

if __name__ == "__main__":
    recognizeMask()
