import face_recognition as fr
import os
import cv2
import face_recognition
import numpy as np
from time import sleep
import imutils


def get_encoded_faces():
    """
    looks through the faces folder and encodes all
    the faces

    :return: dict of (name, image encoded)
    """
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)
                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding

    return encoded


def unknown_image_encoded(img):
    """
    encode a face given the file name
    """
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]

    return encoding


def classify_face(im):
    """
    will find all of the faces in a given image and label
    them if it knows what they are

    :param im: str of file path
    :return: list of face names
    """
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)
    #img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    #img = img[:,:,::-1]
 
    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)
        name = "Unknown"

        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(faces_encoded, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(img, (left-20, top-20), (right+20, bottom+20), (255, 0, 0), 2)
            (x, y, w, h) = (left, top, right, bottom)
            print(x,y,w,h)
            roi = img[y:h, x:w]
            roi = imutils.resize(roi,width=500, inter=cv2.INTER_CUBIC)
            #cv2.imshow("ROI", roi)
            if x>98 and x<491 and y<377 :
              cv2.imshow("ROI", roi)
              cv2.imwrite(image_src,roi)
              cv2.waitKey(0)

for folder_name in os.listdir('train'):
    
    # image_folder_src contain the path test/folder_under_test_folder. For e.g. test/w1, test/w2 etc
    image_folder_src ='train/{}'.format(folder_name)
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
            image_src = image + root_ext[1]
            print(classify_face(image_src))


