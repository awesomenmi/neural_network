import os
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.layers.advanced_activations import PReLU
from keras_vggface.vggface import VGGFace
from keras.optimizers import Adam
import matplotlib.pyplot as plt


img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model_xxx.h5'
train_data_dir = r'data1/train'
validation_data_dir = r'data1/validation'
test_data_dir = r'data1/recognition'
nb_train_samples = 18400
nb_validation_samples = 1840
nb_test_samples = 920
epochs = 10
batch_size = 10
num_classes = 184

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    model = VGGFace(model = 'vgg16', 
                    include_top = False, 
                    input_shape = (img_width, img_height,3))

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = False)
    
    np.save('class_indices.npy', generator.class_indices)
    
    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('bottleneck_features_train.npy',
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False)

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train():

    datagen_top = ImageDataGenerator(rescale=1. / 255)

    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    train_data = np.load('bottleneck_features_train.npy')

    train_labels = generator_top.classes
    train_labels = to_categorical(
        train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    validation_data = np.load('bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)
    

    model = Sequential()

    model.add(Flatten(input_shape = train_data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(2048, kernel_initializer = 'glorot_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(1024, kernel_initializer = 'glorot_uniform'))
    model.add(PReLU())
    model.add(Dense(num_classes, kernel_initializer = 'glorot_uniform', activation='softmax'))

    model.summary()

    model.compile(optimizer=Adam(lr=1e-5),
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_data = (validation_data,validation_labels))

    model.save_weights('model_train.h5')

    (eval_loss, eval_accuracy) = model.evaluate(  
    validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))  
    print("[INFO] Loss: {}".format(eval_loss)) 

    plt.figure(1)  
   
    plt.subplot(211)  
    plt.plot(history.history['acc'])  
    plt.plot(history.history['val_acc'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    
    plt.subplot(212)  
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.show()  


save_bottlebeck_features()
train()

