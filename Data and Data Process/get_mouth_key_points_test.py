import csv
import glob
import os
import os.path
import sys
from imutils import face_utils
import json
import numpy as np
import imutils
import dlib
import cv2
import codecs
from numpyencoder import NumpyEncoder

data = {
  "w1": [],
  "w2": [],
  "w3": [],
  "w4": [],
  "w5": [],
  "w6": [],
  "w7": [],
  "w8": [],
  "w9": [],
  "w10": []
}
#main code start here

#main code
# folder_name contain the folders under test folder. For e.g. w1, w2 etc
for folder_name in os.listdir('test'):
    
    # image_folder_src contain the path test/folder_under_test_folder. For e.g. test/w1, test/w2 etc
    image_folder_src ='test/{}'.format(folder_name)
    # old_video_number contain the video number of previous image. 
    # As video number always start with 1 so here initial value is 1
    old_video_number = '1'
    # img contain the files under the path test/folder_under_test_folder. For e.g. w1_test_2, w1_test_2-0003 etc
    for img in os.listdir(image_folder_src):
        root_ext = os.path.splitext(img)
        if root_ext[1]!=".mp4":
            # image contain the path test/folder_under_test_folder/image_file_under_these_folders. 
            # For e.g. test/w1/w1_test_2-0003, test/w2/w2_test_2-0003 etc
            image = image_folder_src+"/"+root_ext[0]
            # For e.g. test/w1/w1_test_2-0003, test/w2/w2_test_2-0003 etc
            image = image_folder_src+"/"+root_ext[0]
            #image_name contain the image name. For e.g. w9_test_3-0001, w9_test_3-0005 etc
            image_name = str(root_ext[0])
            # image_name_split split the image_name by _. For e.g. ['w9', 'test', '3-0002'], ['w9', 'test', '3-0003'] etc
            image_name_split = image_name.split("_")
            # video_number_and_image_number contain the video number under each class folder 
            # and image number under each video. For e.g. ['3', '0003'], ['1', '0003'] etc
            video_number_and_image_number = image_name_split[2].split("-")
            # video_number contain the video number under each class folder. For e.g. 1, 2 etc
            current_video_number = str(video_number_and_image_number[0])
            # to check the video number of the images are same or not
            if old_video_number != current_video_number:
                # if the videos number is not equal append * to indicate the end of all the images key points under the same video
                data[folder_name].append('*')
            image_src = image + root_ext[1]
            #print(image_src)
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            image_src = cv2.imread(image_src)
            gray = cv2.cvtColor(image_src,cv2.COLOR_BGR2GRAY)
            rects = detector(gray,1)
            mouth_landmarks = []
            for (i,rects) in enumerate(rects):
                shape = predictor(gray,rects)
                shape = face_utils.shape_to_np(shape)
                for (name, (i,j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
                    #print(i,j
                    if name=="mouth":
                        clone = image_src.copy()
                        for (x, y) in shape[i:j]:
                            #if x>130 and y>97 and y<412 and x<940:
                            position =[]
                            cv2.circle(clone, (x,y), 3, (0, 0, 255), -1)
                            position.append(x)
                            position.append(y)
                            print(x,y)
                            mouth_landmarks.append(position)
            
                        #if x>130 and y>97 and y<412 and x<940:
                        mouth_landmarks = np.array(mouth_landmarks)
                        #print(mouth_landmarks)
                        data[folder_name].append(mouth_landmarks)
                        old_video_number = current_video_number
                        print(image)
                        cv2.imshow("Image",clone)
                        cv2.waitKey(0)
                        
    data[folder_name].append('*')
            #detector = dlib.get_frontal_face_detector()
            #predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
            #image = cv2.imread(img_src)
            #gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            #rects = detector(gray,1)
            #mouth_landmarks = []
#print(data)
with open('array_data_test.json', 'w') as file:
    json.dump(data, file, indent=4, sort_keys=True,
              separators=(', ', ': '), ensure_ascii=False,
              cls=NumpyEncoder)