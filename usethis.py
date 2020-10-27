import tensorflow as tf
import cv2 as cv
import urllib
import numpy as np


cvtTable = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z',26:'0',27:'1',28:'2',29:'3',30:'4',31:'5',32:'6',33:'7',34:'8',35:'9'}

model = tf.keras.models.load_model('./model.h5')

def local(path):
    cha = []
    content=''
    image = cv.imread(path)
    
    # 把下面两行取消注释就可以看图片
    #cv.imshow('code',image)
    #cv.waitKey(2000)
    
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    _, image = cv.threshold(image, 224, 255, cv.THRESH_BINARY_INV)
    image = cv.resize(image,(90,25))
    
    cha.append(image[0:25, 0:25])
    cha.append(image[0:25, 23:48])
    cha.append(image[0:25, 45:70])
    cha.append(image[0:25, 65:90])

    cha = np.array(cha)
    cha = cha/127.5-1
    cha = cha[...,tf.newaxis]

    res = model.predict(cha)
    for i in range(4):  content += cvtTable[res[i].argmax()] 
    
    print(content)
    return content



def hahaha(url):
    cha = []
    content = ''
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    
    # 把下面两行取消注释就可以看图片
    #cv.imshow('code',image)
    #cv.waitKey(2000)
    
    image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    _, image = cv.threshold(image, 224, 255, cv.THRESH_BINARY_INV)
    image = cv.resize(image,(90,25))
    
    cha.append(image[0:25, 0:25])
    cha.append(image[0:25, 23:48])
    cha.append(image[0:25, 45:70])
    cha.append(image[0:25, 65:90])
    
    cha = np.array(cha)
    cha = cha/127.5-1
    cha = cha[...,tf.newaxis]

    res = model.predict(cha)
    for i in range(4):  content += cvtTable[res[i].argmax()] 
    
    print(content)
    return content

if __name__ == "__main__":
    #hahaha('https://kdoc.cnki.net/kdoc/request/ValidateCode.ashx?t=1577242936454')
    local('./123.jpg')