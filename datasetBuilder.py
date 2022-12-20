import os
import cv2
import numpy as np
from tqdm import tqdm
import random


def prepareDataset():
    data_dir = "data/dataset_raw/"
    output_dir = "data/64x64_dataset"
    size = 64
    confidence = 0.5
    face_model = "face_detector"

    prototxtPath = os.path.join(face_model, "deploy.prototxt")
    weightsPath = os.path.join(face_model, "res10_300x300_ssd_iter_140000.caffemodel")
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    trainMask = os.path.join(data_dir, 'train/Mask')
    trainNonMask = os.path.join(data_dir, 'train/Non Mask')
    os.makedirs(trainMask, exist_ok=True)
    os.makedirs(trainNonMask, exist_ok=True)

    testMask = os.path.join(data_dir, 'test/Mask')
    testNonMask = os.path.join(data_dir, 'test/Non Mask')
    os.makedirs(testMask, exist_ok=True)
    os.makedirs(testNonMask, exist_ok=True)

    maskImages = os.listdir(trainMask)
    maskImages = [os.path.join(trainMask, f) for f in maskImages if f.endswith('.jpg')]

    nonMaskImages = os.listdir(trainNonMask)
    nonMaskImages = [os.path.join(trainNonMask, f) for f in nonMaskImages if f.endswith('.jpg')]

    testMaskImages = os.listdir(testMask)
    testMaskImages = [os.path.join(testMask, f) for f in testMaskImages if f.endswith('.jpg')]

    testNonMaskImages = os.listdir(testNonMask)
    testNonMaskImages = [os.path.join(testNonMask, f) for f in testNonMaskImages if f.endswith('.jpg')]

    # Split images into 80% train and 20% dev
    random.seed(1121)
    maskImages.sort()
    nonMaskImages.sort()
    random.shuffle(maskImages)
    random.shuffle(nonMaskImages)

    partOfMaskImages = int(0.8 * len(maskImages))
    trainMaskImages = maskImages[:partOfMaskImages]
    validationMaskImages = maskImages[partOfMaskImages:]

    partOfNonMaskImages = int(0.8 * len(nonMaskImages))
    trainNonMaskImages = nonMaskImages[:partOfNonMaskImages]
    validationNonMaskImages = nonMaskImages[partOfNonMaskImages:]

    filenames = {'train/Mask': trainMaskImages, 'train/Non Mask': trainNonMaskImages,
                 'test/Mask': testMaskImages, 'test/Non Mask': testNonMaskImages,
                 'validation/Mask': validationMaskImages, 'validation/Non Mask': validationNonMaskImages}

    for split in filenames.keys():
        output_dir_split = os.path.join(output_dir, split)
        os.makedirs(output_dir_split, exist_ok=True)

        for filename in tqdm(filenames[split]):
            get_face(filename, output_dir_split, net, size, confidence)


def get_face(filename, output_dir, net, size, expected_confidence):
    image = cv2.imread(filename)
    filename_out = filename.split('/')[-1].split('.')[0]
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(128, 128), mean=(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > expected_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            frame = image[startY:endY, startX:endX]
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            if i > 0:
                image_out = os.path.join(output_dir, '%s_%s.jpg' % (filename_out, i))
            else:
                image_out = os.path.join(output_dir, '%s.jpg' % filename_out)
            cv2.imwrite(image_out, frame)


if __name__ == '__main__':
    prepareDataset()
