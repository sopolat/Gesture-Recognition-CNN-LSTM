
# Load dependencies
from tensorflow.keras.models import Sequential #The calling file will be changed in our application 
from tensorflow.keras.layers import Dense,TimeDistributed,Conv2D,MaxPooling2D,Flatten,LSTM,InputLayer,Masking #The calling file will be changed in our application
import numpy as np #The calling file will be changed in our application
import tensorflow as tf #The calling file will be changed in our application 
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import cv2

from os import listdir
from os.path import isfile, join
import pandas as pd
# Num rows
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
batch_size = 1 #This will also be changed accordingly depending on our data set.

l2id={"Zooming In With Two Fingers":0,
    "Zooming Out With Two Fingers":1,
    "Swiping Left":2,
    "Swiping Right":3,
    "Sliding Two Fingers Left":4,
    "Sliding Two Fingers Right":5,
    "Swiping Down":6,
    "Thumb Up":7,
    "Thumb Down":7,
    "Stop Sign":7,
    "No gesture":7,
    "Shaking Hand":7}
class_weight = {0: 1.,
                1: 1.,
                2: 1.,
                3:1.,
                4:1.,
                5:1.,
                6:1.,
                7:0.2}
# Load data
def generate_arrays_from_file(path,data, batchsize):
    inputs = []
    targets = []
    batchcount = 0
    index=0
    while True:
        if(index>=len(data)):
            index=0
        i=str(data.iloc[index,0])
        label=data.iloc[index,1]
        index+=1
        images = [f for f in listdir(path+i) if isfile(join(path+i, f))]
        ims=[]
#        ls=[]
        if len(images)>64:
            continue
#        for k in range(64-len(images)):
#            ims.append(np.zeros((100,176,3)))
#            ls.append(7)
        for im in images:
            ims.append(cv2.resize(cv2.imread (path+i+"/"+im),(176,100)))
#            ls.append(l2id[label])

        vid=np.array(ims)
#        vidl=np.array(ls)
        inputs.append(vid)
        targets.append(l2id[label])
        batchcount += 1
        if batchcount >= batchsize:
            X = np.array(inputs, dtype='float32')
            y = np.array(targets, dtype='float32')
            yield (X, y)
            inputs = []
            targets = []
            batchcount = 0
        


pdata=pd.read_csv("positivesT.csv")
ndata=pd.read_csv("negativesT.csv")
data = pd.concat([pdata, ndata], axis=0)
#model = Sequential()
#
#
#
#model.add(InputLayer(input_shape=(None,100,176,3),batch_size=batch_size))
#model.add(TimeDistributed(Conv2D(32, (5,5), activation='relu', padding='same')))
#model.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3))))
#model.add(TimeDistributed(Conv2D(32, (5,5), activation='relu', padding='same')))
#model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
#model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same')))
#model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
#model.add(TimeDistributed(Conv2D(32, (3,3), activation='relu', padding='same')))
#model.add(TimeDistributed(Flatten()))
#
#model.add(LSTM(128,return_sequences=False))
#model.add(Dense(64, activation='relu'))
#model.add(Dense(8))

#model.add(LSTM(128,return_sequences=True))
#model.add(TimeDistributed(Dense(64, activation='relu')))
#model.add(TimeDistributed(Dense(8)))
#
model=tf.keras.models.load_model("test4/model-08.hdf5")

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'],steps_per_execution=64)
checkpointer=tf.keras.callbacks.ModelCheckpoint(filepath="test4/model-{epoch:02d}.hdf5",verbose=1)
#print(model.summary())
for i in [3,4,5,6,7,8,9]:
    data = pd.concat([pdata, ndata.sample(frac=0.2).reset_index(drop=True)], axis=0)
    data = data.sample(frac=1).reset_index(drop=True)
    history = model.fit(generate_arrays_from_file("20bn-jester-v1/",data, batch_size), epochs=i+1, 
                        steps_per_epoch=33955,initial_epoch=i,
                        callbacks=[checkpointer])
#asd=model.predict(generate_arrays_from_file("20bn-jester-v1/",data, batch_size),steps=6259)