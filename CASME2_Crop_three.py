# -*- coding: utf-8 -*-
"""
Spyder Editor
数据库：CASME2-SW
目的：人脸配准与人脸裁剪
姓名：张志豪
单位：中国科学院心理研究所
"""
# _*_ coding:utf-8 _*_

import numpy as np
import cv2
import dlib
import os

SCALE_FACTOR = 1
'''
几个人脸关键区域的点坐标索引，即图像配准就与这些点相关
'''
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                               RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

detector = dlib.get_frontal_face_detector()
Predictor_File = 'F:\\Micro_Expression_Spotting_GLASTCNN\\predictor_68_face_landmarks\\shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(Predictor_File)

class TooManyFaces(Exception):
    pass

class NoFaces(Exception):
    pass

def get_landmarks(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    rects = detector(image_gray, 0)
    ### 可能 0就是灰度图像，1就是RGB图像
    ### rects.show()
    ### len(rects)=1 respresent just one face in image
    if len(rects) > 1:
        raise TooManyFaces("too many face")
    if len(rects) == 0:
        raise NoFaces("not discovered face")

    return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

'''
人脸配准函数
'''
def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])
'''
input_image 待配准图像
MATRIX 转换矩阵
dshape 模板图像的大小

函数 warpAffine
'''
def warp_im(input_image, MATRIX, dshape):
    align_image = np.zeros(dshape, dtype = input_image.dtype)
    cv2.warpAffine(input_image,
                   MATRIX[:2],
                   (dshape[1], dshape[0]),
                   dst = align_image,
                   borderMode = cv2.BORDER_TRANSPARENT,
                   flags = cv2.WARP_INVERSE_MAP)
    return align_image

def crop_face(image_align):
    img_gray = cv2.cvtColor(image_align, cv2.COLOR_BGR2GRAY)
    rects = detector(img_gray, 0)
    point = []
    for idex, face in enumerate(rects):
        #### predictor
        shape = predictor(image_align, face)
        for idx, pt in enumerate(shape.parts()):
            if (idx == 17) or (idx == 21) or (idx == 22) or (idx == 26) or (idx == 33) or (idx == 39) or (idx == 42) or (idx == 48) or (idx ==54):
                point.append((pt.x, pt.y))
    
    w_el1 = int(point[0][0])
    w_el2 = int(point[1][0])
    h_el1 = int(point[1][1] - 0.5*(w_el2-w_el1))
    h_el2 = int(point[1][1])
    
    w_er1 = int(point[2][0])
    w_er2 = int(point[3][0])
    h_er1 = int(point[2][1]-0.5*(w_er2-w_er1))
    h_er2 = int(point[2][1])
    
    w_fw1 = int(point[5][0] - (point[6][0] - point[5][0]))
    w_fw2 = int(point[6][0] + (point[6][0] - point[5][0]))
    h_fw1 = int(point[5][1] + point[6][1] - point[4][1])
    h_fw2 = int(2*point[4][1] - 0.5*(point[5][1] + point[6][1]))
    
    h_ml1 = int(point[4][1])
    h_ml2 = int(point[7][1]+(point[7][1]-point[4][1]))
    w_ml1 = int(point[7][0] - (point[4][0] - point[7][0]))
    w_ml2 = int(point[4][0])
    
    h_mr1 = int(point[4][1])
    h_mr2 = int(point[8][1]+(point[8][1] - point[4][1]))
    w_mr1 = int(point[4][0])
    w_mr2 = int(point[8][0]+(point[8][0]-point[4][0]))
    
    return [h_el1, h_el2, w_el1, w_el2, 
            h_er1, h_er2, w_er1, w_er2, 
            h_fw1, h_fw2, w_fw1, w_fw2, 
            h_ml1, h_ml2, w_ml1, w_ml2, 
            h_mr1, h_mr2, w_mr1, w_mr2]

'''
得到模板人脸的关键点
'''
image_templet = cv2.imread("F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\templateface.jpg",cv2.IMREAD_COLOR)
landmarks1 = get_landmarks(image_templet)
### image_templet = cv2.resize(image_templet, (image_templet.shape[1] * SCALE_FACTOR,
###                         image_templet.shape[0] * SCALE_FACTOR))
Path_Align   = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\Align\\"

Path_Crop_EL = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\Crop\\EyeLeft\\"
Path_Crop_ER = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\Crop\\EyeRight\\"
Path_Crop_FW = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\Crop\\Face\\"
Path_Crop_ML = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\Crop\\MouthLeft\\"
Path_Crop_MR = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\Crop\\MouthRight\\"

Path_GYH_EL  = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\GYH\\EyeLeft\\"
Path_GYH_ER  = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\GYH\\EyeRight\\"
Path_GYH_FW  = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\GYH\\Face\\"
Path_GYH_ML  = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\GYH\\MouthLeft\\"
Path_GYH_MR  = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\GYH\\MouthRight\\"

Path         = "F:\\Micro_Expression_Spotting_GLASTCNN\\CASME2_Related\\CASME2_Sliding_40\\CASME2_Long_Label\\"
Subject = os.listdir(Path)
for sub in Subject:
    os.mkdir(os.path.join(Path_Align, sub))
    
    os.mkdir(os.path.join(Path_Crop_EL, sub))
    os.mkdir(os.path.join(Path_Crop_ER, sub))
    os.mkdir(os.path.join(Path_Crop_FW, sub))
    os.mkdir(os.path.join(Path_Crop_ML, sub))
    os.mkdir(os.path.join(Path_Crop_MR, sub))
    
    os.mkdir(os.path.join(Path_GYH_EL, sub))
    os.mkdir(os.path.join(Path_GYH_ER, sub))
    os.mkdir(os.path.join(Path_GYH_FW, sub))
    os.mkdir(os.path.join(Path_GYH_ML, sub))
    os.mkdir(os.path.join(Path_GYH_MR, sub))
    
    Sw = os.listdir(os.path.join(Path, sub))
    for sw in Sw:
        
        os.mkdir(os.path.join(Path_Align, sub, sw))
        
        os.mkdir(os.path.join(Path_Crop_EL, sub, sw))
        os.mkdir(os.path.join(Path_Crop_ER, sub, sw))
        os.mkdir(os.path.join(Path_Crop_FW, sub, sw))
        os.mkdir(os.path.join(Path_Crop_ML, sub, sw))
        os.mkdir(os.path.join(Path_Crop_MR, sub, sw))
        
        os.mkdir(os.path.join(Path_GYH_EL, sub, sw))
        os.mkdir(os.path.join(Path_GYH_ER, sub, sw))
        os.mkdir(os.path.join(Path_GYH_FW, sub, sw))
        os.mkdir(os.path.join(Path_GYH_ML, sub, sw))
        os.mkdir(os.path.join(Path_GYH_MR, sub, sw))
        
        Image = os.listdir(os.path.join(Path, sub, sw))
        Image.sort(key = lambda x: int(x.split('.')[0]))
        image_first = cv2.imread(os.path.join(Path, sub, sw, Image[0]), cv2.IMREAD_COLOR)
        landmarks2 = get_landmarks(image_first)
        Matrix = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])
        
        for img in Image:
            image_raw = cv2.imread(os.path.join(Path, sub, sw, img), cv2.IMREAD_COLOR)
            image_warp = warp_im(image_raw, Matrix, image_templet.shape)
            cv2.imwrite(os.path.join(Path_Align, sub, sw, img), image_warp)
        
        
        Image_Align = os.listdir(os.path.join(Path_Align, sub, sw))
        Image_Align.sort(key = lambda x: int(x.split('.')[0]))
        image_align_first = cv2.imread(os.path.join(Path_Align, sub, sw, Image_Align[0]), cv2.IMREAD_COLOR)
        [h_el1, h_el2, w_el1, w_el2, h_er1, h_er2, w_er1, w_er2, 
         h_fw1, h_fw2, w_fw1, w_fw2, 
         h_ml1, h_ml2, w_ml1, w_ml2, h_mr1, h_mr2, w_mr1, w_mr2] = crop_face(image_align_first)
        
        for img_align in Image_Align:
            image_align_raw = cv2.imread(os.path.join(Path_Align, sub, sw, img_align), cv2.IMREAD_COLOR)
            
            crop_el = image_align_raw[h_el1:h_el2, w_el1:w_el2]
            cv2.imwrite(os.path.join(Path_Crop_EL, sub, sw, img_align), crop_el)
            gyh_el = cv2.resize(crop_el, (32, 16))
            cv2.imwrite(os.path.join(Path_GYH_EL, sub, sw, img_align), gyh_el)
            
            crop_er = image_align_raw[h_er1:h_er2, w_er1:w_er2]
            cv2.imwrite(os.path.join(Path_Crop_ER, sub, sw, img_align), crop_er)
            gyh_er = cv2.resize(crop_er, (32, 16))
            cv2.imwrite(os.path.join(Path_GYH_ER, sub, sw, img_align), gyh_er)
            
            crop_fw = image_align_raw[h_fw1:h_fw2, w_fw1:w_fw2]
            cv2.imwrite(os.path.join(Path_Crop_FW, sub, sw, img_align), crop_fw)
            gyh_fw = cv2.resize(crop_fw, (64, 64))
            cv2.imwrite(os.path.join(Path_GYH_FW, sub, sw, img_align), gyh_fw)
            
            crop_ml = image_align_raw[h_ml1:h_ml2, w_ml1:w_ml2]
            cv2.imwrite(os.path.join(Path_Crop_ML, sub, sw, img_align), crop_ml)
            gyh_ml = cv2.resize(crop_ml, (32, 32))
            cv2.imwrite(os.path.join(Path_GYH_ML, sub, sw, img_align), gyh_ml)
            
            crop_mr = image_align_raw[h_mr1:h_mr2, w_mr1:w_mr2]
            cv2.imwrite(os.path.join(Path_Crop_MR, sub, sw, img_align), crop_mr)
            gyh_mr = cv2.resize(crop_mr, (32, 32))
            cv2.imwrite(os.path.join(Path_GYH_MR, sub, sw, img_align), gyh_mr)
            