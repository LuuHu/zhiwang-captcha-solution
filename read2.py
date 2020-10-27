
import cv2 as cv
import numpy as np
import tensorflow as tf

cvtTable = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9'}
cvtTable = dict(zip(cvtTable.values(),cvtTable.keys()))

cha = []
label = []

threshold_value = [208,216,224,232,240]

def prosess_img(gray_img,thres=[224]):
    get = []
    for o in thres:
        _, pic = cv.threshold(gray_img, o, 255, cv.THRESH_BINARY_INV)
        pic = cv.resize(pic,(90,25))
        get.append(pic[0:25,  0:25])
        get.append(pic[0:25, 23:48])
        get.append(pic[0:25, 45:70])
        get.append(pic[0:25, 65:90])
    return get

with open('./key0.txt','r') as f:
    files = f.readlines()
    
for file in files:
    file = file.split(' ')
    pic = cv.imread('./img0/'+file[0])
    pic = cv.cvtColor(pic,cv.COLOR_BGR2GRAY)
    cha.extend(prosess_img(pic,threshold_value))
    tt = [cvtTable[at] for at in file[1][:-1]]*len(threshold_value)
    label.extend(tt)

with open('./key1.txt','r') as f:
    files = f.readlines()
    
for file in files:
    file = file.split(' ')
    pic = cv.imread('./img1/'+file[0])
    pic = cv.cvtColor(pic,cv.COLOR_BGR2GRAY)
    cha.extend(prosess_img(pic,threshold_value))
    tt = [cvtTable[at] for at in file[1][:-1]]*len(threshold_value)
    label.extend(tt)

label = np.array(label)
cha = np.array(cha)
cha = cha/127.5-1
cha = cha[...,tf.newaxis]
#print(label)
label = tf.one_hot(label,36)

def nn():
    inpt = tf.keras.Input(shape=(25,25,1))
    x = tf.keras.layers.Conv2D(64,3,padding='valid')(inpt)
    x = tf.keras.layers.Conv2D(96,3,padding='same')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, 5, padding='same')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(144, 3, padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Dense(126)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LeakyReLU(0.1)(x)
    x = tf.keras.layers.Dense(96)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    res = tf.keras.layers.Dense(36,activation='softmax')(x)
    model = tf.keras.Model(inputs=[inpt],outputs=[res])
    return model

mode = nn()

mode.compile(optimizer='Adam',loss=tf.keras.losses.categorical_crossentropy,metrics="MSE")
mode.fit(cha,label,32,100,validation_split=0.2,callbacks=[tf.keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)])

mode.save('./model.h5')

