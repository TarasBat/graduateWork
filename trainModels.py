from keras.preprocessing.image import ImageDataGenerator
from keras.applications import xception, vgg16, mobilenet_v2,  inception_v3
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint,  EarlyStopping
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.layers import Conv2D, MaxPool2D,  Flatten, Dense, MaxPooling2D, Dropout, SpatialDropout2D, AveragePooling2D, Input
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # for windows without gpu
tf.get_logger().setLevel('WARNING')

def modelXception(learning_rate, input_shape):
    baseModel = xception.Xception(include_top=False, input_tensor=Input(shape=input_shape))
    for layer in baseModel.layers:
        layer.trainable = False

    modelXception = Sequential()
    modelXception.add(baseModel)
    modelXception.add(AveragePooling2D(pool_size=(2, 2)))
    modelXception.add(Flatten())
    modelXception.add(Dense(512, activation="relu"))
    modelXception.add(Dropout(0.5))
    modelXception.add(Dense(50, activation="relu"))
    modelXception.add(Dropout(0.5))
    modelXception.add(Dense(1, activation='sigmoid'))

    modelXception.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=adam_v2.Adam(learning_rate=learning_rate))
    return modelXception

def modelMobileNetV2(learning_rate, input_shape):
    baseModel = mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False)
    for layer in baseModel.layers[:-4]:
        layer.trainable = False

    modelMobileNetV2 = Sequential()
    modelMobileNetV2.add(baseModel)
    modelMobileNetV2.add(AveragePooling2D(pool_size=(2, 2)))
    modelMobileNetV2.add(Flatten())
    modelMobileNetV2.add(Dense(512, activation="relu"))
    modelMobileNetV2.add(Dropout(0.5))
    modelMobileNetV2.add(Dense(50, activation="relu"))
    modelMobileNetV2.add(Dropout(0.5))
    modelMobileNetV2.add(Dense(1, activation='sigmoid'))

    modelMobileNetV2.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=adam_v2.Adam(learning_rate=learning_rate))
    return modelMobileNetV2

def modelCNN(learning_rate, input_shape):
    modelCNN = Sequential()
    modelCNN.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
    modelCNN.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=input_shape, activation='relu'))
    modelCNN.add(MaxPooling2D(pool_size=(2, 2)))
    modelCNN.add(Dropout(0.5))
    modelCNN.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    modelCNN.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    modelCNN.add(MaxPooling2D(pool_size=(2, 2)))
    modelCNN.add(Dropout(0.5))
    modelCNN.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
    modelCNN.add(MaxPooling2D(pool_size=(2, 2)))
    modelCNN.add(Dropout(0.5))
    modelCNN.add(Flatten())
    modelCNN.add(Dense(256, activation='relu'))
    modelCNN.add(Dropout(0.5))
    modelCNN.add(Dense(50, activation="relu"))
    modelCNN.add(Dropout(0.5))
    modelCNN.add(Dense(1, activation='sigmoid'))

    modelCNN.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=adam_v2.Adam(learning_rate=learning_rate))
    return modelCNN

def modelVGG16(learning_rate, input_shape):
    baseModel = vgg16.VGG16(include_top=False, input_tensor=Input(shape=input_shape))
    for layer in baseModel.layers:
        layer.trainable = False

    modelVGG16 = Sequential()
    modelVGG16.add(baseModel)
    modelVGG16.add(AveragePooling2D(pool_size=(2, 2)))
    modelVGG16.add(Flatten())
    modelVGG16.add(Dense(512, activation="relu"))
    modelVGG16.add(Dropout(0.5))
    modelVGG16.add(Dense(50, activation="relu"))
    modelVGG16.add(Dropout(0.5))
    modelVGG16.add(Dense(1, activation='sigmoid'))

    modelVGG16.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer=adam_v2.Adam(learning_rate=learning_rate))
    return modelVGG16


# get memory analytics
def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes( layer, batch_size=batch_size)
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum([tf.keras.backend.count_params(p) for p in model.trainable_weights])
    non_trainable_count = sum( [tf.keras.backend.count_params(p) for p in model.non_trainable_weights])

    total_memory = ( batch_size * shapes_mem_count + internal_model_mem_count + trainable_count + non_trainable_count)
    return total_memory


if __name__ == "__main__":

    net_type = "CNN"
    data_dir = "data/64x64_dataset"
    epochs = 30
    model = "mask_detector.model"
    defaultSize = 64

    bs = 32
    lr = 0.0001
    size = (defaultSize, defaultSize)
    shape = (defaultSize, defaultSize, 3)
    epochs = epochs
    show_history = True

    trainData = ImageDataGenerator(rescale=1./255, rotation_range=5, zoom_range=0.2, shear_range=0.2, brightness_range=[0.9, 1.1], horizontal_flip=True)
    validData = ImageDataGenerator(rescale=1./255, rotation_range=5, zoom_range=0.2, shear_range=0.2, brightness_range=[0.9, 1.1], horizontal_flip=True)
    testData = ImageDataGenerator(rescale=1./255)

    trainData = trainData.flow_from_directory(os.path.join(data_dir, 'train'), target_size=size, shuffle=True, batch_size=bs, class_mode='binary')
    validData = validData.flow_from_directory(os.path.join(data_dir, 'test'), target_size=size, shuffle=True, batch_size=bs, class_mode='binary')
    testData = testData.flow_from_directory(os.path.join(data_dir, 'validation'), target_size=size, shuffle=True, batch_size=bs, class_mode='binary')

    print(trainData.class_indices)
    print(trainData.image_shape)

    # Prepare model
    model = modelVGG16(lr, shape)
    #model = modelCNN(lr, shape)
    #model = modelXception(lr, shape)
    #model = modelMobileNetV2(lr, shape)
    model.build()
    model.summary()

    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, mode='auto')
    tensorBoard = TensorBoard(log_dir=os.path.join("logs", "filelog"))
    modelCheckpoint = ModelCheckpoint(os.path.join("results", f"test1" + f"-size-{size[0]}" + f"-bs-{bs}" + f"-lr-{lr}.h5"), monitor='val_loss',save_best_only=True, verbose=1)
    # model training
    history = model.fit(trainData, epochs=epochs, validation_data=validData, batch_size=bs, callbacks=[earlyStopping, tensorBoard, modelCheckpoint], shuffle=True)
    testLoss, testAccuracy = model.evaluate(testData)
    print(pd.DataFrame(history.history).head(10))

    print('Test loss: ', testLoss)
    print('Test accuracy: ', testAccuracy)
    print('Memory consumption: %s bytes' % keras_model_memory_usage_in_bytes(model, batch_size=bs))
    
    # save model
    model.save(model, save_format="h5")

    if show_history:
        plt.subplot(211)
        plt.title('Loss')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()

        plt.subplot(212)
        plt.title('Accuracy')
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.show()
