'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    for idx in range(1, 104):

        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        locs = cascade.detectMultiScale(gray,1.1,5)
        for dims in locs:
            x = dims[0]
            y = dims[1]
            w = dims[2]
            h = dims[3]
            bbox = [float(x), float(y), float(w), float(h)]
            # while True :
            #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
               
            #     cv2.imshow("_img",img)
                
            #     if cv2.waitKey(1)& 0xFF==ord('q'):
            #             break
           
            cv2.destroyAllWindows()
           
            detection_results.append(bbox)

    # Add your code here. Do not modify the return and input arguments.
    
    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    #start here
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    nimages = len(imgs)
    im = []
    i_dim = []
    np.random.seed(42)
    encode = []

    for i in range(1, nimages+1):
        img = imgs[f'{i}.jpg']

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(gray, 1.5, 8)

        for (x, y, w, h) in faces:

            encodings = face_recognition.face_encodings(img, [(y, x, y+h, x+w)])
            encode.append(encodings)
            im.append(i)
            i_dim.append([int(x), int(y), int(w), int(h)]) 

    encode = np.array(encode)
    centroids = Centroids_func(encode, K)

    kmean = kmeans(encode, K, centroids, 5)

    objects = list(imgs.keys())

    for i in range(K):

        element = []
        holc = []
        for j, val in enumerate(kmean):
            if val == i:
                element.append(objects[j])
                i_j = imgs[objects[j]]

                x, y, w, h = int(i_dim[j][0]), int(i_dim[j][1]), int(i_dim[j][2]), int(i_dim[j][3])
                i_j = i_j[y:y+h, x:x+w]

                i_j = cv2.resize(i_j, (80, 80), interpolation=cv2.INTER_NEAREST)
                holc.append(i_j)

        h_img = cv2.hconcat(holc)
        # cv2.imwrite('cluster_'+str(i)+'.jpg', h_img)
        
        # cv2.imshow('Horizontal', h_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        elem = {"cluster_no": i, "elements": element}



        for i in range(K):
            for x,cun in enumerate(kmean):
                if cun == i:
                    cluster_results[i].append(objects[x])

    # Add your code here. Do not modify the return and input arguments.
    
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)

def Euclidean_distance(a, b):
    euc = math.sqrt(np.sum((a - b)**2))
    return euc

def Centroids_func(data, k):
    cent = []
    cent.append(data[np.random.randint(data.shape[0]
                
                ), :]
                )

    for ind in range(k - 1):
        dist = []
        for i in range(data.shape[0]):
            point = data[i, :]
            d = sys.maxsize

            for j in range(len(cent)):
                temp_dist = Euclidean_distance(point, cent[j])
                d = min(d, temp_dist)
            dist.append(d)

        dist = np.array(dist)
        next_centroid = data[np.argmax(dist), :]
        cent.append(next_centroid)

    return cent

def Centroids_distance(x, y):
    g_p = []

    for i in range(len(x)):
        for j in range(len(y)):
            d = x[i][0]-y[j][0]
            
            d = np.sum(np.power(d, 2)
                        )
            g_p.append(d)

    g_p = np.array(g_p)
    g_p = np.reshape(g_p, (len(x), len(y))
                    )
    return g_p

def kmeans(x, k, cent, iter):
    dist_matrix = Centroids_distance(x, cent
                    )
    image_class = np.array([np.argmin(d) for d in dist_matrix]
                )

    for i in range(iter):
        cent = []
        for j in range(k):
            ncent = x[image_class == j]
            m = 0
            for l in range(len(ncent)):
                m += ncent[l]
            ncent = np.divide(m, len(ncent)
            )
            cent.append(ncent)

    return image_class