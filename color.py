import numpy as np

def mean_rgb(upleft, downright, image):
    rgbs = []
    for i in range(upleft[0], downright[0]+1):
        for j in range(upleft[1], downright[1]+1):
            rgbs.append(image[i,j])
    rgbs = np.array(rgbs)
    return np.mean(rgbs, axis=0)

def mean_rgb_face(landmarks, image):
    face_upleft = np.array([landmarks[1,0], landmarks[11,1]], dtype=int)
    face_downright = np.array([landmarks[4,0], landmarks[16,1]], dtype=int)
    return mean_rgb(face_upleft, face_downright, image)

def mean_rgb_hair(landmarks, image):
    delta_y = landmarks[41,1] - landmarks[0,1]
    hair_upleft = np.array([landmarks[1,0], landmarks[1,1]-delta_y], dtype=int)
    hair_downright = np.array([landmarks[9,0], landmarks[1,1]-delta_y+3], dtype=int)
    return mean_rgb(hair_upleft, hair_downright, image)