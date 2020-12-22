## setup
import os
import os.path
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.layers.advanced_activations import PReLU
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import matplotlib.pyplot as plt
import math
import dlib
from skimage import io
from preprocessing import FaceDetector, FaceAligner, clip_to_range
from tqdm import tqdm
from itertools import repeat
from load_data import load_file

img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model_xxx.h5'
train_data_dir = r'data1/train'
validation_data_dir = r'data1/validation'
nb_train_samples = 18400
nb_validation_samples = 1840
epochs = 50
batch_size = 5
num_classes = 184
PROBABILITY = False

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

#fd = FaceDetector()
#fa = FaceAligner('.\model\shape_predictor_68_face_landmarks.dat', '.\model\face_template.npy')

IMAGE_FORMATS = {'.jpg', '.jpeg', '.png'}

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

    np.save('bottleneck_features_train_xxx.npy',
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

    np.save('bottleneck_features_validation_xxx.npy',
            bottleneck_features_validation)


def train():

    datagen_top = ImageDataGenerator(rescale=1. / 255)

    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    train_data = np.load('bottleneck_features_train_xxx.npy')

    train_labels = generator_top.classes
    train_labels = to_categorical(
        train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_data = np.load('bottleneck_features_validation_xxx.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)
    
        
    model = Sequential()

    model.add(Flatten(input_shape = train_data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(2048, kernel_initializer = 'glorot_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256, kernel_initializer = 'glorot_uniform'))
    model.add(PReLU())
    model.add(Dense(num_classes, kernel_initializer = 'glorot_uniform', activation = 'softmax'))

    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs = epochs,
                        batch_size = batch_size,
                        validation_data = (validation_data,validation_labels))

    model.save_weights(top_model_weights_path)

    model.save('model_train_xxx.h5')

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

def predict():

    lookup = {
    0: 'Aaron_Eckhart', 
    1: 'Adam_Brody', 
    2: 'Adrien_Brody', 
    3: 'Alan_Alda', 
    4: 'Alan_Rickman', 
    5: 'Alec_Baldwin', 
    6: 'Alexander_Skarsgard', 
    7: 'Allison_Janney', 
    8: 'Alyson_Hannigan', 
    9: 'Alyssa_Milano', 
    10: 'Amaury_Nolasco', 
    11: 'America_Ferrera', 
    12: 'Andrea_Bowen', 
    13: 'Andy_Garcia', 
    14: 'Angie_Harmon', 
    15: 'Anne_Hathaway', 
    16: 'Anthony_Stewart_Head', 
    17: 'Ashley_Benson', 
    18: 'Ashley_Jones', 
    19: 'Ben_Affleck', 
    20: 'Ben_McKenzie', 
    21: 'Ben_Stiller', 
    22: 'Bill_Hader', 
    23: 'Billy_Burke', 
    24: 'Brad_Pitt', 
    25: 'Bradley_Cooper', 
    26: 'Bruce_Greenwood', 
    27: 'Bruce_Willis', 
    28: 'Candace_Cameron_Bure', 
    29: 'Candice_Bergen', 
    30: 'Cary_Elwes', 
    31: 'Casey_Affleck', 
    32: 'Charlie_Sheen', 
    33: 'Cheryl_Hines', 
    34: 'Christa_Miller', 
    35: 'Christian_Slater', 
    36: 'Christina_Applegate', 
    37: 'Ciara_Bravo', 
    38: 'Colin_Farrell', 
    39: 'Colin_Firth', 
    40: 'Colin_Hanks', 
    41: 'Courteney_Cox', 
    42: 'Dana_Delany', 
    43: 'Daniel_Day-Lewis', 
    44: 'Daniel_Radcliffe', 
    45: 'Danny_Masterson', 
    46: 'David_Boreanaz', 
    47: 'David_Schwimmer', 
    48: 'Debi_Mazar', 
    49: 'Debra_Messing', 
    50: 'Dermot_Mulroney', 
    51: 'Dianna_Agron', 
    52: 'Edie_Falco', 
    53: 'Elijah_Wood', 
    54: 'Elizabeth_Berkley', 
    55: 'Emile_Hirsch', 
    56: 'Ethan_Hawke', 
    57: 'Eva_Longoria', 
    58: 'Ewan_McGregor', 
    59: 'Felicity_Huffman', 
    60: 'Fran_Drescher', 
    61: 'Freddy_Prinze_Jr', 
    62: 'Gabriel_Macht', 
    63: 'Gary_Oldman', 
    64: 'Geena_Davis', 
    65: 'Geoffrey_Rush', 
    66: 'George_Clooney', 
    67: 'George_Lopez', 
    68: 'Gerard_Butler', 
    69: 'Glenn_Close', 
    70: 'Hayden_Christensen', 
    71: 'Heath_Ledger', 
    72: 'Heather_Locklear', 
    73: 'Helen_Hunt', 
    74: 'Holly_Marie_Combs', 
    75: 'Hugh_Grant', 
    76: 'James_Frain', 
    77: 'James_Franco', 
    78: 'James_Marsden', 
    79: 'James_McAvoy', 
    80: 'Jane_Lynch', 
    81: 'January_Jones', 
    82: 'Jared_Padalecki', 
    83: 'Jason_Bateman', 
    84: 'Jason_Biggs', 
    85: 'Jason_Sudeikis', 
    86: 'Jay_Baruchel', 
    87: 'Jeffrey_Tambor',
    88: 'Jennette_McCurdy', 
    89: 'Jennie_Garth', 
    90: 'Jennifer_Aniston', 
    91: 'Jeremy_Irons', 
    92: 'Jerry_Seinfeld', 
    93: 'Jesse_Eisenberg', 
    94: 'Jessica_Capshaw', 
    95: 'Jim_Caviezel', 
    96: 'Joan_Collins', 
    97: 'Joanna_Garcia', 
    98: 'Joe_Manganiello', 
    99: 'John_Cusack', 
    100: 'Johnny_Depp', 
    101: 'Jon_Hamm', 
    102: 'Jon_Voight', 
    103: 'Jonah_Hill', 
    104: 'Jonathan_Rhys_Meyers', 
    105: 'Josh_Brolin', 
    106: 'Josh_Duhamel', 
    107: 'Joshua_Jackson', 
    108: 'Jude_Law', 
    109: 'Julia_Louis-Dreyfus', 
    110: 'Julianna_Margulies', 
    111: 'Julie_Benz', 
    112: 'Julie_Bowen', 
    113: 'Kal_Penn', 
    114: 'Karl_Urban', 
    115: 'Katherine_Kelly_Lang', 
    116: 'Kevin_Bacon', 
    117: 'Kevin_Connolly', 
    118: 'Kevin_McKidd', 
    119: 'Kiefer_Sutherland', 
    120: 'Kim_Cattrall', 
    121: 'Kirstie_Alley', 
    122: 'Kit_Harington', 
    123: 'Kristin_Chenoweth', 
    124: 'Lacey_Chabert', 
    125: 'Laura_Leighton', 
    126: 'Lea_Michele', 
    127: 'Lisa_Kudrow', 
    128: 'Lori_Loughlin', 
    129: 'Lorraine_Bracco', 
    130: 'Marcia_Cross', 
    131: 'Marg_Helgenberger', 
    132: 'Mark_Ruffalo', 
    133: 'Mark_Wahlberg', 
    134: 'Matt_Czuchry', 
    135: 'Matt_Damon', 
    136: 'Matt_LeBlanc', 
    137: 'Matthew_Broderick', 
    138: 'Matthew_Lillard', 
    139: 'Matthew_Perry', 
    140: 'Mayim_Bialik', 
    141: 'Mel_Gibson', 
    142: 'Michael_Douglas', 
    143: 'Michael_Vartan', 
    144: 'Michael_Weatherly', 
    145: 'Milo_Ventimiglia', 
    146: 'Miranda_Cosgrove', 
    147: 'Nadia_Bjorlin', 
    148: 'Neve_Campbell', 
    149: 'Nicolas_Cage', 
    150: 'Orlando_Bloom', 
    151: 'Patricia_Arquette', 
    152: 'Patrick_Dempsey', 
    153: 'Peter_Sarsgaard', 
    154: 'Pierce_Brosnan', 
    155: 'Rachel_Dratch', 
    156: 'Robert_Knepper', 
    157: 'Robert_Redford', 
    158: 'Roma_Downey', 
    159: 'Roseanne_Barr', 
    160: 'Rue_McClanahan', 
    161: 'Russell_Crowe', 
    162: 'Ryan_Phillippe', 
    163: 'Ryan_Reynolds', 
    164: 'Sarah_Drew', 
    165: 'Sarah_Hyland', 
    166: 'Sarah_Michelle_Gellar', 
    167: 'Sasha_Alexander', 
    168: 'Sean_Bean', 
    169: 'Selena_Gomez', 
    170: 'Seth_Rogen', 
    171: 'Shannen_Doherty', 
    172: 'Shia_LaBeouf', 
    173: 'Simon_Pegg', 
    174: 'Stana_Katic', 
    175: 'Steve_Carell', 
    176: 'Summer_Glau', 
    177: 'Susan_Lucci', 
    178: 'Tamala_Jones', 
    179: 'Teri_Hatcher', 
    180: 'Tina_Fey', 
    181: 'Tobey_Maguire', 
    182: 'Valerie_Bertinelli', 
    183: 'Victoria_Justice'}

    class_dictionary = np.load('class_indices.npy').item()

    image_path = r'.\data1\recognition\Christina_Applegate\Christina_Applegate_6089_2990.jpeg'

    image = load_file(image_path, 150, 0)
    #cascade_file_src = "haarcascade_frontalface_default.xml"
    #faceCascade = cv2.CascadeClassifier(cascade_file_src)

    #img = io.imread(image_path)
    #faces = fd.detect_faces(img, get_top=1)

    #face = fa.align_face(img, faces[0], dim=img_width, border = 0).reshape(1, img_width, img_height, 3)
    #face = clip_to_range(face)

    #face = face.astype(np.float32)


    print("[INFO] loading and preprocessing image...")
    image = cv2.resize(cv2.imread(image_path),(img_width, img_height))
    image = image.astype(np.float32)
    #win1 = dlib.image_window()
    #win1.clear_overlay()
    #win1.set_image(image)

    image = img_to_array(image)

    image = image / 255
    image = np.expand_dims(image, axis=0)

    model = VGGFace(model = 'vgg16', 
                    include_top = False, 
                    input_shape = (img_width, img_height,3))

    bottleneck_prediction = model.predict(image)

    model = Sequential()
    model.add(Flatten(input_shape = bottleneck_prediction.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(2048, kernel_initializer = 'glorot_uniform'))
    model.add(PReLU())
    model.add(Dropout(0.5))
    model.add(Dense(256, kernel_initializer = 'glorot_uniform'))
    model.add(PReLU())
    model.add(Dense(num_classes, kernel_initializer = 'glorot_uniform', activation='softmax'))

    model.load_weights(top_model_weights_path)

    predict_proba = model.predict(bottleneck_prediction)
    if (np.count_nonzero(predict_proba)) < 4:
        if PROBABILITY:
            locs = map(lambda x:lookup[x], range(num_classes))
            proba_order = zip(predict_proba.tolist()[0], locs)
            print(sorted(proba_order, reverse=True))

        class_predicted = model.predict_classes(bottleneck_prediction)
        inID = class_predicted[0]
        inv_map = {v: k for k, v in class_dictionary.items()}
        label = inv_map[inID]
        print("\nImage ID: {}, Label: {}".format(inID, label))
    else: print("\nThis person isnt in this database")

#save_bottlebeck_features()
#train()
predict()
