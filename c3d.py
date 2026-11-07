# -*- coding: utf-8 -*-
from keras.layers import MaxPool3D,BatchNormalization,Reshape
from keras.layers import AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Dense,Dropout,Conv3D,Input,Flatten,Activation
from keras.layers import maximum, Multiply, Concatenate, Add
from keras.regularizers import l2
from keras.models import Model
from attention_module import attach_attention_module



def glastcnn(attention_module = None):
    
    weight_decay = 0.008
    kernelnum_1 = 32
    ###### ------------------------------- face area ----------------------------------------------------########
    input_of_face = Input(shape = (32, 64, 64, 3), 
                          dtype = 'float32', 
                          name = 'input_of_face')
    
    conv3d_1_face = Conv3D(kernelnum_1, (3, 3, 3), 
                           strides = (2, 2, 2), 
                           padding = 'same', 
                           kernel_regularizer = l2(weight_decay))(input_of_face)
    bn_1_face = BatchNormalization()(conv3d_1_face)
    act_1_face = Activation('relu')(bn_1_face)
    pool3d_1_face = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_1_face)
    
    ### print(K.int_shape(pool3d_1_Gray))
    conv3d_2_face = Conv3D(32, (3, 3, 3), 
                           strides = (1, 1, 1), 
                           padding = 'same', 
                           kernel_regularizer = l2(weight_decay))(pool3d_1_face)
    bn_2_face = BatchNormalization()(conv3d_2_face)
    act_2_face = Activation('relu')(bn_2_face)
    pool3d_2_face = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_2_face)
    
    conv3d_3_face = Conv3D(64, (3, 3, 3), 
                           strides = (1, 1, 1), 
                           padding = 'same', 
                           kernel_regularizer = l2(weight_decay))(pool3d_2_face)
    bn_3_face = BatchNormalization()(conv3d_3_face)
    act_3_face = Activation('relu')(bn_3_face)
    pool3d_3_face = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_3_face)

    conv3d_4_face = Conv3D(128, (3, 3, 3), 
                           strides = (1, 1, 1), 
                           padding = 'same', 
                           kernel_regularizer = l2(weight_decay))(pool3d_3_face)
    bn_4_face = BatchNormalization()(conv3d_4_face)
    act_4_face = Activation('relu')(bn_4_face)
    pool3d_4_face = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_4_face)
    
    global_feature = Reshape((2, 2, 128))(pool3d_4_face)
    ### 1*2*2*128 reshape 为 2*2*128
    if attention_module is not None:
        global_feature = attach_attention_module(global_feature, attention_module)
    ### global_feature 依旧为 2*2*128 的特征图
    global_feature = GlobalAveragePooling2D()(global_feature)
    
    '''
    flatten_global = Flatten()(pool3d_4_face)
    dense_global = Dense(128, activation='relu', kernel_regularizer=l2(weight_decay))(flatten_global)
    dense_global = Dropout(0.5)(dense_global)
    '''
    
    ###### ------------------------------- eyeleft area -------------------------------------------------########
    input_of_eyeleft = Input(shape = (16, 16, 32, 3), 
                             dtype = 'float32', 
                             name = 'input_of_eyeleft')
    
    conv3d_1_eyeleft = Conv3D(kernelnum_1, (3, 3, 3), 
                              strides = (2, 2, 2), 
                              padding = 'same', 
                              kernel_regularizer = l2(weight_decay))(input_of_eyeleft)
    bn_1_eyeleft = BatchNormalization()(conv3d_1_eyeleft)
    act_1_eyeleft= Activation('relu')(bn_1_eyeleft)
    pool3d_1_eyeleft = MaxPool3D((2,2,4), strides=(2,2,4), padding='same')(act_1_eyeleft)
    
    conv3d_2_eyeleft = Conv3D(32, (3, 3, 3), 
                              strides = (2, 1, 1), 
                              padding = 'same', 
                              kernel_regularizer = l2(weight_decay))(pool3d_1_eyeleft)
    bn_2_eyeleft = BatchNormalization()(conv3d_2_eyeleft)
    act_2_eyeleft = Activation('relu')(bn_2_eyeleft)
    pool3d_2_eyeleft = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_2_eyeleft)
    ###### flatten_eyeleft = Flatten()(pool3d_2_eyeleft)
    ###### ------------------------------- --------------------------------------------------------------########
    
    ###### ------------------------------- eyeright area -------------------------------------------------########
    input_of_eyeright = Input(shape = (16, 16, 32, 3), 
                              dtype = 'float32', 
                              name = 'input_of_eyeright')
    
    conv3d_1_eyeright = Conv3D(kernelnum_1, (3, 3, 3), 
                               strides = (2, 2, 2), 
                               padding = 'same', 
                               kernel_regularizer = l2(weight_decay))(input_of_eyeright)
    bn_1_eyeright = BatchNormalization()(conv3d_1_eyeright)
    act_1_eyeright= Activation('relu')(bn_1_eyeright)
    pool3d_1_eyeright = MaxPool3D((2,2,4), strides=(2,2,4), padding='same')(act_1_eyeright)
    
    conv3d_2_eyeright = Conv3D(32, (3, 3, 3), 
                               strides = (2, 1, 1), 
                               padding = 'same', 
                               kernel_regularizer = l2(weight_decay))(pool3d_1_eyeright)
    bn_2_eyeright = BatchNormalization()(conv3d_2_eyeright)
    act_2_eyeright = Activation('relu')(bn_2_eyeright)
    pool3d_2_eyeright = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_2_eyeright)
    ###### ------------------------------- --------------------------------------------------------------########
    
    ###### ------------------------------- mouthleft area ----------------------------------------------########
    input_of_mouthleft = Input(shape = (16, 32, 32, 3), 
                               dtype = 'float32', 
                               name = 'input_of_mouthleft')
    
    conv3d_1_mouthleft = Conv3D(kernelnum_1, (3, 3, 3), 
                                strides = (2, 2, 2), 
                                padding = 'same', 
                                kernel_regularizer = l2(weight_decay))(input_of_mouthleft)
    bn_1_mouthleft = BatchNormalization()(conv3d_1_mouthleft)
    act_1_mouthleft= Activation('relu')(bn_1_mouthleft)
    pool3d_1_mouthleft = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_1_mouthleft)
    
    conv3d_2_mouthleft = Conv3D(32, (3, 3, 3), 
                                strides = (2, 2, 2), 
                                padding = 'same', 
                                kernel_regularizer = l2(weight_decay))(pool3d_1_mouthleft)
    bn_2_mouthleft = BatchNormalization()(conv3d_2_mouthleft)
    act_2_mouthleft = Activation('relu')(bn_2_mouthleft)
    pool3d_2_mouthleft = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_2_mouthleft)
    ###### ------------------------------- --------------------------------------------------------------########
    
    ###### ------------------------------- mouthright area ----------------------------------------------########
    input_of_mouthright = Input(shape = (16, 32, 32, 3), 
                                dtype = 'float32', 
                                name = 'input_of_mouthright')
    
    conv3d_1_mouthright = Conv3D(kernelnum_1, (3, 3, 3), 
                                 strides = (2, 2, 2), 
                                 padding = 'same', 
                                 kernel_regularizer = l2(weight_decay))(input_of_mouthright)
    bn_1_mouthright = BatchNormalization()(conv3d_1_mouthright)
    act_1_mouthright= Activation('relu')(bn_1_mouthright)
    pool3d_1_mouthright = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_1_mouthright)
    
    conv3d_2_mouthright = Conv3D(32, (3, 3, 3), 
                                 strides = (2, 2, 2), 
                                 padding = 'same', 
                                 kernel_regularizer = l2(weight_decay))(pool3d_1_mouthright)
    bn_2_mouthright = BatchNormalization()(conv3d_2_mouthright)
    act_2_mouthright = Activation('relu')(bn_2_mouthright)
    pool3d_2_mouthright = MaxPool3D((2,2,2), strides=(2,2,2), padding='same')(act_2_mouthright)
    
    ###### ------------------------------- --------------------------------------------------------------########
    local_feature = Concatenate(axis = 4)([pool3d_2_eyeleft, pool3d_2_eyeright, pool3d_2_mouthleft, pool3d_2_mouthright])
    local_feature = Reshape((2, 2, 128))(local_feature)
    if attention_module is not None:
        local_feature = attach_attention_module(local_feature, attention_module)
    
    local_feature  = GlobalAveragePooling2D()(local_feature)
    
    
    ### 将全局与局部特征融合， 融合方式为 Concat, Add, Multiply
    local_and_global = Concatenate()([global_feature, local_feature])
    ### local_and_global = Add()([global_feature, local_feature])
    outputs = Dense(2, activation='softmax', kernel_regularizer=l2(weight_decay))(local_and_global)
    
    model = Model(inputs = [input_of_face, input_of_eyeleft, input_of_eyeright, input_of_mouthleft, input_of_mouthright], 
                  outputs = outputs, name="glastcnn")
    '''
    dense_local = Dense(128, activation='relu', kernel_regularizer=l2(weight_decay))(flatten_global_feature)
    dense_local = Dropout(0.5)(dense_local)
    
    ## Fl = Multiply()([Fl_G, Fl_H, Fl_V])
    ## Fl = Add()([Fl_G, Fl_H, Fl_V])
    '''
    ###x = Dense(256, kernel_regularizer=l2(weight_decay))(RGB_OF)
    ###x = BatchNormalization()(x)
    ###x = Activation('relu')(x)
    ### x = Dropout(0.5)(x)
    ### X = Dense(2, kernel_regularizer=l2(weight_decay))(X)
    ###Predication = Activation('softmax')(X)
    #### 二分类使用 sigmoid函数，而不是 softmax 函数
    # Create model
    '''
    x = Dense(2,kernel_regularizer=l2(weight_decay))(flatten_global_feature)
    y_Predication = Activation('softmax')(x)
    '''
    '''
    if summary:
        print(model.summary())
    return model
    # model = model.convert_model(model)
    '''
    
    '''
    with open('model_summary.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary()
    '''
    ####### plot_model(model, to_file='./micro_expression model.png', show_shapes= True, show_layer_names=True)
    ####### 网络图在最后加入 model = c3d_early_model()
    return model