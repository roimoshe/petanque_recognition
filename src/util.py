#!/usr/bin/env python3
import sys, os
import argparse
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import io, color
import matplotlib.image as mpimg
from sklearn.cluster import KMeans


def burn_blob_frame_step(img, num_of_pixels, verbose):
    mask = np.ones(img.shape[:2],np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if sum(img[i][j]) == 0:
                x_l = max(0, j-num_of_pixels)
                x_r = min(img.shape[1], j+num_of_pixels)
                y_u = max(0, i-num_of_pixels)
                y_d = min(img.shape[0], i+num_of_pixels)
                mask[y_u:y_d,x_l:x_r]=0
    return cv2.bitwise_and(img,img,mask = mask)

def edgeDetector(image, blur_size, median_blur_size, verbose):
    ksize = (blur_size, blur_size)
    image = cv2.blur(image, ksize, cv2.BORDER_DEFAULT)

    if 0:
        save_photo('build/before_bilateral.jpg', image, True)
    image = cv2.medianBlur(image, median_blur_size)
    # image = cv2.bilateralFilter(np.float32(image), 20, 200, 200)
    if 0:
        save_photo('build/bilateral.jpg', image, True)
    image_uint8 = image.astype(np.uint8)
    edge_map = cv2.Canny(image_uint8, 60, 120)
    return edge_map


def hough2(img, verbose):
    img = cv2.imread('images/day1/image39.jpg', 0)
    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cimg = cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cimg = cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    if verbose:
        util.save_photo('build/hough2.jpg', cimg, True)


def histogram(img_path):
    img = cv2.imread(img_path)
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        curr_histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(curr_histr, color=col)
        plt.xlim([0, 256])
    (height, width, _) = img.shape
    avg = np.array([0, 0, 0])
    for i in range(10):
        avg += np.array(img[int(height / 2) + i][int(width / 2) + i])
    # print(avg/10)
    plt.show()
    # cv2.imshow('hist',hist)
    # cv2.waitKey(0)


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist


def kmeans(img_path, image_space_representation):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, image_space_representation)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    blur = cv2.blur(image, (10, 10))
    clt = KMeans(n_clusters=2)
    clt.fit(blur)

    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)
    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()


def save_photo(path, img, verbosity):
    cv2.imwrite(path, img)
    if verbosity:
        print("image saved in: {}".format(path))


def extract_frames():
    count = 0
    vidcap = cv2.VideoCapture("images/day1/video4.mov")
    success, image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 8000))  # added this line
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        if not success:
            break
        cv2.imwrite("images/video4/frame%d.jpg" % count, image)  # save frame as JPEG file
        count = count + 1


def pca():
    cap = cv2.VideoCapture("images/day1/video4.mov")
    subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True)
    # blur attempt
    ksize = (25, 25)
    flag = False
    while True:
        _, frame = cap.read()
        frame_blur = cv2.blur(frame, ksize, cv2.BORDER_DEFAULT)
        mask = subtractor.apply(frame_blur)
        cv2.imshow("Frame", frame)
        if flag and pre_sum > 0 and mask.sum() == 0:
            avg_i = 0
            avg_j = 0
            cnt = 0
            for i in range(pre_mask.shape[0]):
                for j in range(pre_mask.shape[0]):
                    is_ball = int(pre_mask[i][j] > 0)
                    avg_i += is_ball * i
                    avg_j += is_ball * j
                    cnt += is_ball
            print("i=", int(avg_i / cnt), " j=", int(avg_j / cnt))
        pre_sum = mask.sum()
        pre_mask = mask.copy()
        cv2.imshow("mask", mask)
        # print(mask.sum())
        flag = True
        key = cv2.waitKey(30)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

# def find_y_position_below_half(h, theta, dy1, f):
#     beta = np.arctan(dy1/f)
#     y1 = (dy1*h*np.sin(90-beta)) / (f*np.sin(theta)*np.sin(theta+beta))
#     return y1

def find_y_position(h, theta, dy2, f, is_below_half):
    factor = 1
    if is_below_half:
        factor = -1
    alpha = np.arctan(dy2/f) * factor
    y2 = (dy2*h*np.sin((np.pi/2)+alpha)) / (f*np.sin(theta)*np.sin(theta-alpha))
    return y2

def position():
    # # image1
    # up_y = 560
    # middle_y = 965
    # down_y = 1345
    # h = 54
    # theta = 90-58 #degrees
    # ratio_world = 2

    # image2
    up_y = 620
    middle_y = 695
    down_y = 1300
    h = 50
    theta = 90-73 #degrees
    ratio_world = 0.25

    theta = (theta/180)*np.pi # radians
    f=np.array([0.1 * i for i in range(15000,30000)])
    # f=1766.1000000000001 # image1
    # f=555.2 # image2

    # # find pixels
    img = cv2.imread("images/day1/ball_posotion/bp3_h50_theta73_x4.jpeg")
    # mask = np.ones(img.shape[:2],np.uint8)
    # x = 1000
    # mask[up_y-10:up_y+10,x:x+50] = 0
    # img_masked = cv2.bitwise_and(img,img,mask = mask)
    # save_photo('build/ball_position.jpg',img_masked, True)
    half_y = img.shape[0]/2
    up_world_y     = (-1 * int((up_y     < half_y)) + 1 * int((up_y     > half_y))) * find_y_position(h, theta, abs(half_y-up_y),     f, (up_y     > half_y) )
    middle_world_y = (-1 * int((middle_y < half_y)) + 1 * int((middle_y > half_y))) * find_y_position(h, theta, abs(half_y-middle_y), f, (middle_y > half_y) )
    down_world_y   = (-1 * int((down_y   < half_y)) + 1 * int((down_y   > half_y))) * find_y_position(h, theta, abs(half_y-down_y),   f, (down_y   > half_y) )
    ratio = (up_world_y-middle_world_y)/(middle_world_y-down_world_y)
    print(min(np.abs(ratio/ratio_world - 1)), f[np.argmin(np.abs(ratio/ratio_world - 1))])
    plt.plot(f,np.abs(ratio/ratio_world - 1), 'r')
    # show the plot
    plt.show()
    print("up ", up_world_y)
    print("middle ", middle_world_y)
    print("down ", down_world_y)
    print("ratio ", ratio)

def undistort():
    img = cv2.imread('images/day2/photos/setup2/image0.jpeg')
    h,  w = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite('calibresult.png',dst)