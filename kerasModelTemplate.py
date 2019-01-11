
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
import tensorflow as tf
import keras 
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K



lr_base = 0.001
epochs = 5000
lr_power = 0.9

batchSize  = 32
saveStep = 3

# train/val目录下文件格式为：不同类别图片存入不同文件夹，文件夹名即类别名
trainDataPath = "../../train"
valDataPath = "../../val"
modelSaveDir = "../../checkpoint"
# In[2]:

targetImageSize = (300,300)


trainDataGen = ImageDataGenerator(rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='nearest',
                              data_format='channels_last')

trainGenerator = trainDataGen.flow_from_directory(trainDataPath,
                                      target_size=targetImageSize,
                                       color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=batchSize, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='resnet',
                            save_format='jpeg',
                            follow_links=False)

valDataGen = ImageDataGenerator(rotation_range=40,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True,
                              vertical_flip=True,
                              fill_mode='nearest',
                              data_format='channels_last')

valGenerator = valDataGen.flow_from_directory(valDataPath,
                                      target_size=targetImageSize,
                                       color_mode='rgb',
                            classes=None, class_mode='categorical',
                            batch_size=batchSize, shuffle=True, seed=None,
                            save_to_dir=None,
                            save_prefix='resnet',
                            save_format='jpeg',
                            follow_links=False)


# In[3]:

# 学习率衰减策略
def lr_scheduler(epoch, mode='power_decay'):
    lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    print('lr: %f' % lr)
    return lr


# In[ ]:


with tf.device('/gpu:0'):
    baseModel = ResNet50(weights='imagenet', include_top=False)
    x = baseModel.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    predictions = Dense(6, activation='softmax')(x)
    model = Model(inputs=baseModel.input, outputs=predictions)
    
    early_stopping = EarlyStopping(monitor='val_acc', patience=200, verbose=1)
    checkpointer = ModelCheckpoint(filepath=os.path.join(modelSaveDir,"/checkpoint-{epoch:02d}e-val_acc_{val_acc:.6f}.hdf5"), 
					    monitor='val_acc',
                                   save_best_only=True, 
                                   verbose=1,  
                                   period=saveStep)
    scheduler = ReduceLROnPlateau(monitor='val_acc', factor=0.6, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.000001) 
    #scheduler = LearningRateScheduler(lr_scheduler)
    
    for layer in baseModel.layers:
        layer.trainable = True
        
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    history_tl = model.fit_generator(generator=trainGenerator, 
                                     steps_per_epoch=len(trainGenerator),
                                     epochs=epochs,
                                     validation_data=valGenerator, 
                                     validation_steps=len(valGenerator), 
                                     class_weight='auto' ,
                                    callbacks=[early_stopping, checkpointer, scheduler])

    
    

