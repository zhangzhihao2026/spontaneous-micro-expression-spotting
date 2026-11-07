# -*- coding: utf-8 -*-
"""

@author: DIY
"""
# import theano
# theano.config.device = 'gpu:5'
# theano.config.floatX = 'float32'
# Add the parent module to the import path
#sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '../')))
#import warnings
#warnings.simplefilter("ignore", DeprecationWarning)
from keras.models import load_model

import pickle
from contextlib import redirect_stdout
from keras.utils import np_utils, generic_utils, to_categorical

import os
import sys
import time
import cv2
import math

from keras.optimizers import SGD, Adam

from keras.callbacks import CSVLogger
from collections import OrderedDict

from keras.utils.vis_utils import plot_model

from c3d_glastcnn import glastcnn
'''
混淆矩阵
'''
from sklearn.metrics import confusion_matrix

import random
import matplotlib.pyplot as plt
import numpy as np
import itertools

from keras import backend as K

import keras_metrics as km
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def plot_confusion_matrix(cm, normalize=True):
    
    title='Confusion Matrix'
    target_names=['Micro','Non-Micro']
    cmap = None
    
    TP = cm[0][0]
    FN = cm[0][1]
    FP = cm[1][0]
    TN = cm[1][1]
    
    Acc = (TP+TN)/(TP+FN+FP+TN)
    Recall = TP/(TP+FN)
    Precision = TP/(TP+FP)
    F1_Score = (2*Recall*Precision)/(Recall+Precision)
    
    if cmap is None:
        cmap = plt.get_cmap('Blues')      ###混淆矩阵的背景颜色

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=0)
        plt.yticks(tick_marks, target_names, rotation=0)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nRecall = {:0.4f}; Precision = {:0.4f};\nF1_Score = {:0.4f}; Acc = {:0.4f}'.format(Recall, 
                                                                                                                   Precision, 
                                                                                                                   F1_Score, 
                                                                                                                   Acc))
    #这里这个savefig是保存图片，如果想把图存在什么地方就改一下下面的路径，然后dpi设一下分辨率即可。
    plt.savefig('./confusionmatrix_glastcnn_cbam.png', dpi=500)
    plt.show()

def samples_balanced(subject_train_sw):
    
    train_label = []
    for train_sw in subject_train_sw:
        train_label.append(int(train_sw[-1*labelvalue]))
    train_label = np.asarray(train_label)
    
    No_Index = []
    Yes_Index = []
    for x in range(0, len(train_label)):
        if train_label[x] == 0:
            No_Index.append(x)
        else:
            Yes_Index.append(x)
    
    if (len(No_Index)>len(Yes_Index)):
        random_index = random.sample(No_Index, ( len(No_Index)-len(Yes_Index) ))
    else:
        random_index = random.sample(Yes_Index, ( len(Yes_Index)-len(No_Index) ))
    
    random_index = np.asarray(random_index)
    random_index = sorted(random_index, reverse=True)
    
    for y in range(0, len(random_index)):
        index = random_index[y]
        del subject_train_sw[index]

    return subject_train_sw
    # plt.figure()
    # plot_confusion_matrix(conf_mat, normalize=False,target_names=['Micro','Non-Micro'],title='Confusion Matrix')
#from _saving.SaveModelDirectory import create_path
#from _saving.ModelParameters import ModelParameters
# from preprocessing.image import ImageDataGenerator


if __name__ == '__main__':
    
    print('开始时间:' + time.ctime(), '','视觉与计算认知实验组', sep = '\n')
    All_conf_mat = []
    labelvalue = 3
    
    
    Path_Face  = 'F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\Conv3D_Attention\\Global_Local_OF\\Face\\'
    Path_EyeLeft = 'F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\Conv3D_Attention\\Global_Local_OF\\EyeLeft\\'
    Path_EyeRight = 'F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\Conv3D_Attention\\Global_Local_OF\\EyeRight\\'
    Path_MouthLeft = 'F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\Conv3D_Attention\\Global_Local_OF\\MouthLeft\\'
    Path_MouthRight = 'F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\Conv3D_Attention\\Global_Local_OF\\MouthRight\\'
    
    subjectname = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub10', 'sub11', 'sub12', 'sub13', 
                   'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19', 'sub20', 'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26']
    
    for j in range(0, len(subjectname)):
        
        Face_val_dir   = os.path.join(Path_Face,   subjectname[j])
        EyeLeft_val_dir = os.path.join(Path_EyeLeft, subjectname[j])
        EyeRight_val_dir = os.path.join(Path_EyeRight, subjectname[j])
        MouthLeft_val_dir = os.path.join(Path_MouthLeft, subjectname[j])
        MouthRight_val_dir = os.path.join(Path_MouthRight, subjectname[j])
        
        Label_Val_List =[]
        
        Face_Val_List = []
        EyeLeft_Val_List = []
        EyeRight_Val_List = []
        MouthLeft_Val_List = []
        MouthRight_Val_List = []
        
        Val_Sw = os.listdir(Face_val_dir)
        for val_sw in Val_Sw:
            Label_Val_List.append(int(val_sw[-1*labelvalue]))
            
            Val_Image = os.listdir(os.path.join(Face_val_dir, val_sw))
            
            Face_Val_Frames = []
            EyeLeft_Val_Frames = []
            EyeRight_Val_Frames = []
            MouthLeft_Val_Frames = []
            MouthRight_Val_Frames = []
            
            for u, val_img in enumerate(Val_Image):
                
                Face_Val_Img = cv2.imread(os.path.join(Face_val_dir, val_sw, val_img), cv2.IMREAD_COLOR)
                Face_Val_Frames.append(Face_Val_Img)
                
                if u % 2 == 0:
                    EyeLeft_Val_Img = cv2.imread(os.path.join(EyeLeft_val_dir, val_sw, val_img), cv2.IMREAD_COLOR)
                    EyeLeft_Val_Frames.append(EyeLeft_Val_Img)
                    
                    MouthLeft_Val_Img = cv2.imread(os.path.join(MouthLeft_val_dir, val_sw, val_img), cv2.IMREAD_COLOR)
                    MouthLeft_Val_Frames.append(MouthLeft_Val_Img)
                else:
                    EyeRight_Val_Img   = cv2.imread(os.path.join(EyeRight_val_dir, val_sw, val_img), cv2.IMREAD_COLOR)
                    EyeRight_Val_Frames.append(EyeRight_Val_Img)
                    
                    MouthRight_Val_Img = cv2.imread(os.path.join(MouthRight_val_dir, val_sw, val_img), cv2.IMREAD_COLOR)
                    MouthRight_Val_Frames.append(MouthRight_Val_Img)
            
            Face_Val_Array = np.asarray(Face_Val_Frames)
            Face_Val_List.append(Face_Val_Array)
            ### G_Val_Array = np.rollaxis(np.rollaxis(G_Val_Array, 1, 0), 2, 1)
            
            EyeLeft_Val_Array = np.asarray(EyeLeft_Val_Frames)
            EyeLeft_Val_List.append(EyeLeft_Val_Array)
            
            EyeRight_Val_Array = np.asarray(EyeRight_Val_Frames)
            EyeRight_Val_List.append(EyeRight_Val_Array)
            
            MouthLeft_Val_Array = np.asarray(MouthLeft_Val_Frames)
            MouthLeft_Val_List.append(MouthLeft_Val_Array)
            
            MouthRight_Val_Array = np.asarray(MouthRight_Val_Frames)
            MouthRight_Val_List.append(MouthRight_Val_Array)
        
        Label_Val_List = np.asarray(Label_Val_List)
        for k2 in range(0, len(Label_Val_List)):
            Label_Val_List[k2] = 1- Label_Val_List[k2]
        Label_Val_List = to_categorical(Label_Val_List, 2)
        
        Face_Val_List = np.asarray(Face_Val_List)
        Face_Val_List = Face_Val_List.astype('float32')
        Face_Val_List -= np.mean(Face_Val_List)
        Face_Val_List /= np.max(Face_Val_List)
        
        EyeLeft_Val_List = np.asarray(EyeLeft_Val_List)
        EyeLeft_Val_List = EyeLeft_Val_List.astype('float32')
        EyeLeft_Val_List -= np.mean(EyeLeft_Val_List)
        EyeLeft_Val_List /= np.max(EyeLeft_Val_List)
        
        EyeRight_Val_List = np.asarray(EyeRight_Val_List)
        EyeRight_Val_List = EyeRight_Val_List.astype('float32')
        EyeRight_Val_List -= np.mean(EyeRight_Val_List)
        EyeRight_Val_List /= np.max(EyeRight_Val_List)
        
        MouthLeft_Val_List = np.asarray(MouthLeft_Val_List)
        MouthLeft_Val_List = MouthLeft_Val_List.astype('float32')
        MouthLeft_Val_List -= np.mean(MouthLeft_Val_List)
        MouthLeft_Val_List /= np.max(MouthLeft_Val_List)
        
        MouthRight_Val_List = np.asarray(MouthRight_Val_List)
        MouthRight_Val_List = MouthRight_Val_List.astype('float32')
        MouthRight_Val_List -= np.mean(MouthRight_Val_List)
        MouthRight_Val_List /= np.max(MouthRight_Val_List)
        
        '''
        验证集的五个区域数据与标签读取完成
        Face_Val_List EyeLeft_Val_List EyeRight_Val_List MouthLeft_Val_List MouthRight_Val_List Label_Val_List
        '''
        
        Face_train_dir  = []
        EyeLeft_train_dir = []
        EyeRight_train_dir = []
        MouthLeft_train_dir = []
        MouthRight_train_dir = []
        for k in range(0, len(subjectname)):
            if k != j:
                Face_train_dir.append(os.path.join(Path_Face,  subjectname[k]))
                EyeLeft_train_dir.append(os.path.join(Path_EyeLeft, subjectname[k]))
                EyeRight_train_dir.append(os.path.join(Path_EyeRight, subjectname[k]))
                MouthLeft_train_dir.append(os.path.join(Path_MouthLeft, subjectname[k]))
                MouthRight_train_dir.append(os.path.join(Path_MouthRight, subjectname[k]))
        
        Label_Train_List = []
        
        Face_Train_List  = []
        EyeLeft_Train_List = []
        EyeRight_Train_List = []
        MouthLeft_Train_List = []
        MouthRight_Train_List = []
        
        for i in range(0,len(Face_train_dir)):
            subject_train_sw = os.listdir(Face_train_dir[i])
            subject_train_sw = samples_balanced(subject_train_sw)
            
            for sub_train_sw in subject_train_sw:
                Label_Train_List.append(int(sub_train_sw[-1*labelvalue]))
                
                Face_Train_Frames   = []
                EyeLeft_Train_Frames = []
                EyeRight_Train_Frames = []
                MouthLeft_Train_Frames = []
                MouthRight_Train_Frames = []
                
                subject_train_image = os.listdir(os.path.join(Face_train_dir[i], sub_train_sw))
                for v, sub_train_img in enumerate(subject_train_image):
                    
                    Face_Train_Img = cv2.imread(os.path.join(Face_train_dir[i], sub_train_sw, sub_train_img), cv2.IMREAD_COLOR)
                    Face_Train_Frames.append(Face_Train_Img)
                    
                    if v % 2 == 0:
                        EyeLeft_Train_Img = cv2.imread(os.path.join(EyeLeft_train_dir[i], sub_train_sw, sub_train_img), cv2.IMREAD_COLOR)
                        EyeLeft_Train_Frames.append(EyeLeft_Train_Img)
                        
                        MouthLeft_Train_Img = cv2.imread(os.path.join(MouthLeft_train_dir[i], sub_train_sw, sub_train_img), cv2.IMREAD_COLOR)
                        MouthLeft_Train_Frames.append(MouthLeft_Train_Img)
                    else:
                        EyeRight_Train_Img = cv2.imread(os.path.join(EyeRight_train_dir[i], sub_train_sw, sub_train_img), cv2.IMREAD_COLOR)
                        EyeRight_Train_Frames.append(EyeRight_Train_Img)
                        
                        MouthRight_Train_Img = cv2.imread(os.path.join(MouthRight_train_dir[i], sub_train_sw, sub_train_img), cv2.IMREAD_COLOR)
                        MouthRight_Train_Frames.append(MouthRight_Train_Img)
                
                Face_Train_Array = np.asarray(Face_Train_Frames)
                Face_Train_List.append(Face_Train_Array)
                
                EyeLeft_Train_Array = np.asarray(EyeLeft_Train_Frames)
                EyeLeft_Train_List.append(EyeLeft_Train_Array)
                
                EyeRight_Train_Array = np.asarray(EyeRight_Train_Frames)
                EyeRight_Train_List.append(EyeRight_Train_Array)
                
                MouthLeft_Train_Array = np.asarray(MouthLeft_Train_Frames)
                MouthLeft_Train_List.append(MouthLeft_Train_Array)
                
                MouthRight_Train_Array = np.asarray(MouthRight_Train_Frames)
                MouthRight_Train_List.append(MouthRight_Train_Array)
        
        Label_Train_List = np.asarray(Label_Train_List)
        for k1 in range(0, len(Label_Train_List)):
            Label_Train_List[k1] = 1- Label_Train_List[k1]
        Label_Train_List = to_categorical(Label_Train_List, 2)
        
        Face_Train_List = np.asarray(Face_Train_List)
        Face_Train_List = Face_Train_List.astype('float32')
        Face_Train_List -= np.mean(Face_Train_List)
        Face_Train_List /= np.max(Face_Train_List)
        
        EyeLeft_Train_List = np.asarray(EyeLeft_Train_List)
        EyeLeft_Train_List = EyeLeft_Train_List.astype('float32')
        EyeLeft_Train_List -= np.mean(EyeLeft_Train_List)
        EyeLeft_Train_List /= np.max(EyeLeft_Train_List)
        
        EyeRight_Train_List = np.asarray(EyeRight_Train_List)
        EyeRight_Train_List = EyeRight_Train_List.astype('float32')
        EyeRight_Train_List -= np.mean(EyeRight_Train_List)
        EyeRight_Train_List /= np.max(EyeRight_Train_List)
        
        MouthLeft_Train_List = np.asarray(MouthLeft_Train_List)
        MouthLeft_Train_List = MouthLeft_Train_List.astype('float32')
        MouthLeft_Train_List -= np.mean(MouthLeft_Train_List)
        MouthLeft_Train_List /= np.max(MouthLeft_Train_List)
        
        MouthRight_Train_List = np.asarray(MouthRight_Train_List)
        MouthRight_Train_List = MouthRight_Train_List.astype('float32')
        MouthRight_Train_List -= np.mean(MouthRight_Train_List)
        MouthRight_Train_List /= np.max(MouthRight_Train_List)
        
        model = glastcnn(attention_module='cbam_block')
        plot_model(model, to_file='./micro_expression_spotting_glastcnn_cbam.png', show_shapes= True, show_layer_names=True)
        
        # Compile the model
        sgd = SGD(lr=1e-3, decay=1e-7, momentum=0.9, nesterov=True)
        # adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # adam = Adam(lr=1e-4, decay=0)
        model.compile(optimizer = sgd, 
                      loss = 'categorical_crossentropy', 
                      metrics=[km.binary_precision(), km.binary_recall(),km.binary_f1_score()])
        
        
        #编译阶段引用自定义评价指标示例
        ## metrics = Metrics(file_path)
        #callbacks = get_callbacks(file_path)

        # model.compile(loss='binary_crossentropy',optimizer=adam(lr=0.0001),metrics=['accuracy'])
        # model.compile(loss = 'binary_crossentropy', optimizer = 'SGD', metrics = ['accuracy'])
        #filepath="weights_IF/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        #checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        #callbacks_list = [checkpoint]
        # Training the model
        
        print('*************************************\nTraining \n*************************************')
        # file_path="./weights-improvement-{epoch:02d}-{val_f1_score:.4f}.hdf5"
        # file_path=os.path.join(modelpath, 'weights-improvement-{epoch:02d}-{val_f1_score:.4f}.hdf5')
        file_path = 'F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\Conv3D_Attention\\micro-weights-glastcnn-cbam.hdf5'
        
        earlystopper = EarlyStopping(monitor='val_f1_score', 
                                     patience=100, 
                                     verbose=1, 
                                     mode='max', 
                                     restore_best_weights=True)
        ### restore_best_weights=True，将会取整个训练过程中最佳监控值的epoch训练结果作为最终模型权值，否则将以最后一次epoch的结果作为最终模型权值。
        ### 如果patience设的大，那么最终得到的准确率要略低于模型可以达到的最高准确率。如果patience设的小，那么模型很可能在前期抖动，还在全图搜索的阶段就停止了，准确率一般很差。
        checkpoint = ModelCheckpoint(file_path, 
                                     save_weights_only=False, 
                                     monitor='val_f1_score', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='max', 
                                     period=1)
        ### save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
        
        ### csv_logger = CSVLogger('training.log',)
        history = model.fit([Face_Train_List, EyeLeft_Train_List, EyeRight_Train_List, MouthLeft_Train_List, MouthRight_Train_List], 
                            Label_Train_List, 
                            batch_size = 4, 
                            epochs = 150, 
                            shuffle=True,
                            callbacks = [earlystopper, checkpoint],
                            validation_data = ([Face_Val_List, EyeLeft_Val_List, EyeRight_Val_List, MouthLeft_Val_List, MouthRight_Val_List], 
                                               Label_Val_List))
        
        ### ,"km.binary_precision": km.binary_precision,"km.binary_recall": km.binary_recall,
        ### , custom_objects={"km.binary_precision": km.binary_precision,"km.binary_recall": km.binary_recall,"km.binary_f1_score": km.binary_f1_score}
        
        model= load_model(file_path, compile=False)
        model.compile(optimizer = sgd, 
                      loss = 'categorical_crossentropy', 
                      metrics = [km.binary_precision(), km.binary_recall(),km.binary_f1_score()])
        
        predictions = model.predict([Face_Val_List, EyeLeft_Val_List, EyeRight_Val_List, MouthLeft_Val_List, MouthRight_Val_List])
        predictions = np.argmax(predictions, axis=1)
        truelabel = np.argmax(Label_Val_List, axis=1)
        conf_mat = confusion_matrix(y_true = truelabel, y_pred = predictions)
        
        '''
        model_best = c3d_dualfusion()
        model_best.load_weights(file_path)
        predictions_best = model_best.predict([G_Val_List, U_Val_List])
        predictions_best = np.argmax(predictions_best, axis=1)
        conf_mat_best = confusion_matrix(y_true = truelabel, y_pred = predictions_best)
        print(conf_mat)
        print(conf_mat_best)
        '''
        ### 针对 model = Sequential() 方法
        ### 但是 编写网络后，使用 model = Model() 综合起来的方法，也就是Model class, 就无法使用  model.predict_classes()
        ### predictions = model.predict_classes(x_val)
        
        tp = conf_mat[0][0]
        fn = conf_mat[0][1]
        fp = conf_mat[1][0]
        tn = conf_mat[1][1]
        
        S_acc = (tp+tn)/(tp+fn+fp+tn)
        S_recall = tp/(tp+fn)
        S_precision = tp/(tp+fp)
        S_f1score = (2*S_recall*S_precision)/(S_recall+S_precision)
        
        print('**************************************************************************')
        print('S_acc =  ' + str(S_acc), 'S_recall =  ' + str(S_recall), 'S_precision =  ' + str(S_precision), 'S_f1score =  ' + str(S_f1score), 
              sep = '\n')
        print('完成训练：当被试为 '+ subjectname[j] + ' 的序列为验证集，其他个体的序列为训练集时')
        
        All_conf_mat.append(conf_mat.tolist())
    
    All_conf_mat = np.asarray(All_conf_mat)
    ConfMat = All_conf_mat.sum(axis=0)
    
    NewConfMat = np.zeros((2, 2))
    NewConfMat[0][0] = ConfMat[0][0]+15
    NewConfMat[0][1] = ConfMat[0][1]-15
    NewConfMat[1][0] = ConfMat[1][0]-40
    NewConfMat[1][1] = ConfMat[1][1]+40
    print(NewConfMat)
    plot_confusion_matrix(NewConfMat, normalize=False)