import os
import numpy as np
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
def precision(y_true, y_pred):
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
     precision = true_positives / (predicted_positives + K.epsilon())
     return precision

def recall(y_true, y_pred):
     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
     recall = true_positives / (possible_positives + K.epsilon())
     return recall

def f1(y_true, y_pred, beta=1):
   if beta < 0:    
       raise ValueError('The lowest choosable beta is zero (only precision).')   
   
   # If there are no true positives, fix the F score at 0 like sklearn.    
   if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:   
       return 0        
   p = precision(y_true, y_pred)   
   r = recall(y_true, y_pred)    
   bb = beta ** 2    
   fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())   
   return fbeta_score

img_width, img_height = 128,128
model_path = 'models/model_norm.h5'
model_weights_path = 'models/weights_norm.h5'
model = load_model(model_path,custom_objects={"precision":precision,"f1":f1,"recall":recall})
model.load_weights(model_weights_path)

img = {
    0:'Golf-Swing-Back',
    1:'Golf-Swing-Front',
    2:'Golf-Swing-Side',
    3:'Kicking-Front',
    4:'Kicking-Side',
    5:'Lifting',
    6:'Riding-Horse',
    7:'Running',
    8:'SkateBoarding',
    9:'Swing-Bench',
    10:'Swing-SideAngle',
    11:'Walking'
}

def predict(files):
    answer = []
    for f in files:
        x = load_img(f, target_size=(img_width,img_height))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        array = model.predict(x)
        result = array[0]
        answer.append(np.argmax(result) + 1)
    for i in answer:
        print(i)

def predictVerbose(file):
    answer = []
   
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    answer = img[np.argmax(result)]
    return answer
        

from subprocess import call
path = input()

rem_x = ''
def split_video(path):
    path2 = path.split('/')
    o = []
    for i in path2:
        if i != '':
            o.append(i)
    p = '/'.join(o[:-1])
    #print(p)
    rem_x = p + "/temp"
    #print("X",rem_x)
    if not os.path.exists(rem_x):
        os.makedirs(rem_x)
    dest = rem_x + '/' + path2[-1] + '-%04d.jpg'
    #print(dest)
    call(["ffmpeg", "-i", path, dest])
    return path2[-1]

w = []
for files in os.listdir(path):
    if files.endswith('avi'):
        w.append(path+'/'+files)
#print(w)
def predictFinal(path):
    import glob
    import random
    from collections import Counter
    import shutil
    try:
#         print("Removing previous files...")
    #     os.remove()
        shutil.rmtree('/'.join(path.split('/')[:-1])+'/temp', ignore_errors=True)
    except Exception as e:
        print(e)
        pass
    vid = split_video(path)
    images = glob.glob(os.path.join('predict', 'temp', '*.jpg'))
    # print(images)
    d = []

    for i in range(0,5):
        sample = random.randint(0, len(images) - 1)
        image = images[sample]
        answer = predictVerbose(image)
        d.append(answer)

    answer = Counter(d).most_common(1)[0][0]
    #Merging these sub categories into their main categories
    if answer.startswith('Golf'):
        answer = 'golfswing'
    if answer.startswith('Kick'):
        answer = 'kicking'
    return vid + " " + answer.lower()

answer = []
for i in w:
    answer.append(predictFinal(i))

for i in answer:
    print(i)
