from sklearn.cluster import KMeans
import cv2
import numpy as np
from scipy import ndimage
from . import util


def recreate_image(codebook, labels, w, h, original):
    labels = labels.reshape((w, h))
    for i in range(w):
        for j in range(h):
            original[i][j]=original[i][j]*labels[i][j]
    return original

def f(x, lab):
    return np.int(x==lab)

def flip(x):
    return 1-x

def kmeans(image_orig, image_space_representation, verbose):
    w, h, d = original_shape = tuple(image_orig.shape)
    image = cv2.cvtColor(image_orig, image_space_representation) / 255
    image = image.reshape((image.shape[0] * image.shape[1], d))
    clt = KMeans(n_clusters = 4).fit(image)
    labels = clt.predict(image)
    if verbose:
        util.save_photo('build/labels_of_kmeans.jpg',100 * labels.reshape((w, h)), True)
    court_label = np.bincount(labels).argmax()
    
    f2 = np.vectorize(f)
    labels = f2(labels, court_label)
    
    labels = labels.reshape((w, h))
    labels = ndimage.binary_fill_holes(labels, structure=np.ones((11,11))).astype(int)
    # labels = 1-ndimage.binary_fill_holes(1-labels, structure=np.ones((2,2))).astype(int)
    labels = labels.reshape((w * h))
    img_no_backgroung = recreate_image(clt.cluster_centers_, labels, w, h, image_orig)
    return img_no_backgroung
