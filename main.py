from keras.models import *           #从TensorFlow Keras库中导入所有模型相关的函数，使用Keras提供的所有模型类和函数
import pandas as pd                             #
import cv2
import numpy as np
def model():
    model = load_model('model1.h5')
    return model

def read(path):

    img = cv2.imread(path)
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
    img = img / 255
    img = img.reshape(1, 256, 256, 3)
    return img
def pre(model,img):
    pred = model.predict(img)
    y = np.argmax(pred, axis=-1)
    labels= {0:'1',1:'2',2:'3',3:'4',4:'5',5:'6',6:'7',7:'8',8:'9',9:'10',10:'11',11:'12',12:'13',13:'14',14:'15',15:'16',16:'17'}
    y = pd.DataFrame(y)
    y[0]=y[0].map(labels)
    y = y.values.flatten()
    print('此花为：',y)
    return y

if __name__ == '__main__':
    path = r'flowers/image_0073.jpg'
    img = read(path)
    model = model()
    pred = pre(model,img)
    print(pred)
