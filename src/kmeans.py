from sklearn.cluster import KMeans
import cv2
import numpy as np
from scipy import ndimage
from . import util


def recreate_image(codebook, labels, w, h, original):
    labels = labels.reshape((w, h))
    for i in range(w):
        for j in range(h):
            original[i][j] = original[i][j] * labels[i][j]
    return original


def f(x, lab):
    return np.int(x == lab)

def kmeans(blurred_image, image_space_representation, n_clusters, min_max, original_image, verbose):
    w, h, d = tuple(blurred_image.shape)

    image = cv2.cvtColor(blurred_image, image_space_representation) / 255
    image = cv2.bilateralFilter(np.float32(image), 9, 200, 200)
    image = image.reshape((image.shape[0] * image.shape[1], d))
    # maybe blur here
    clt = KMeans(n_clusters=n_clusters).fit(image)
    labels = clt.predict(image)
    if verbose:
        util.save_photo('build/labels_of_kmeans.jpg', 100 * labels.reshape((w, h)), True)
    if min_max == "max":
        court_label = np.bincount(labels).argmax()
    else:
        court_label = np.bincount(labels).argmin()

    f2 = np.vectorize(f)
    labels = f2(labels, court_label)

    labels = labels.reshape((w, h))
    labels = ndimage.binary_fill_holes(labels, structure=np.ones((11, 11))).astype(int)
    labels = 1 - ndimage.binary_fill_holes(1 - labels, structure=np.ones((2, 2))).astype(int)
    labels = labels.reshape((w * h))
    img_no_backgroung = recreate_image(clt.cluster_centers_, labels, w, h, original_image)
    return img_no_backgroung
