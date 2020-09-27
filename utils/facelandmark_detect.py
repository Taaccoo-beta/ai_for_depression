
# --------------------------------------------------------
# get face key points
# Licensed under The MIT License [see LICENSE for details]
# Written by YaoWendi
# --------------------------------------------------------
import face_alignment
import cv2
import numpy as np
import time
import sys

#This function can detect face landmarks from video
#videopath is the path of the video
#if LandmarksType is 3D then can detect 3D landmarks
#if LandmarksType is 2D then can detect 2D landmarks
def face_detect(videopath,LandmarksType):

    start = time.time()
    if LandmarksType == '3D' or LandmarksType == '3d':
        landmarkstype = face_alignment.LandmarksType._3D
        print('开始进行3D关键点检测')
    elif LandmarksType =='2D' or LandmarksType =='2d':
        landmarkstype = face_alignment.LandmarksType._2D
        print('开始进行2D关键点检测')
    else:
        print("Error!Please put '3D' or '2D' !")
        sys.exit(0)

    fa = face_alignment.FaceAlignment(landmarkstype, device='cuda', flip_input=True,face_detector='blazeface')
    cap = cv2.VideoCapture(videopath) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    landmarks = []
    nonDetectFr = []
    my_figsize, my_dpi = (20, 10), 80
    totalIndx = 0
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # create VideoWriter object
    totalFrame = np.int32(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames: ", totalFrame)

    while(cap.isOpened()):
        frameIndex = np.int32(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print("Processing frame ", frameIndex, "...")
        ret, frame = cap.read()
        if ret==True:
        # operations on the frame
            try:
                # generate face bounding box and track 2D landmarks for current frame
                frame_landmarks = fa.get_landmarks(frame)[-1]
            except:
                print("Landmarks in frame ", frameIndex, " (", frameIndex/fps, " s) could not be detected.")
                nonDetectFr.append(frameIndex/fps)
                continue
            landmarks.append(frame_landmarks)
            totalIndx = totalIndx + 1
        else:
            break  
    cap.release()
    end = time.time()
    print("processing time:" + str(end - start))
    return(landmarks)
