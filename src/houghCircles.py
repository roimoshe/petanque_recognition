#!/usr/bin/env python3
import sys, os
import argparse
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy import unravel_index
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import io, color
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from . import util


def createKernel(r):
    color = (255, 255, 255)
    thickness = 1
    image = np.zeros(shape=[2 * r + 5, 2 * r + 5, 3], dtype=np.uint8)
    center_coordinates = (r + 2, r + 2)
    image = cv2.circle(image, center_coordinates, r, color, thickness)
    kernel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = kernel / 255
    kernel = np.array(kernel).astype(int)
    return kernel


def filter_close_circles(circles_center1, verbose):
    circles_center = []
    circles_radius = []
    for [x, y, z] in circles_center1:
        is_new_circle = True
        for i, center in enumerate(circles_center):
            if abs(center[0] - x + center[1] - y) < z / 2:
                if circles_radius[i] < z:
                    circles_center.pop(i)
                    circles_radius.pop(i)
                else:
                    is_new_circle = False
                break
        if is_new_circle:
            circles_center.append([x, y])
            circles_radius.append(z)
    if verbose:
        for center, z in zip(circles_center, circles_radius):
            print("x = ", center[0], " y = ", center[1], " r = ", z)
    return circles_center, circles_radius


def HoughCircles(edge_map, threshold_arg, hough_radius_range, verbose):
    new_edge_map = []
    for number in edge_map:
        new_edge_map.append(number / 255)  # making the edge_map contain values of only [1,0]
    new_edge_map = np.array(new_edge_map, dtype=np.int)

    circles_center1 = []

    # for each radios
    # create circle for convolution
    # run the convolution on every pixle and save the result
    # find the resualts local maxima over a threshold if there are, and append them in the answer

    for r in range(hough_radius_range[0], hough_radius_range[1]):
        kernel = createKernel(r)  # creating the kernel for the matching radius
        accumulator = cv2.filter2D(new_edge_map, ddepth=cv2.CV_32S,
                                   kernel=kernel)  # convoluting the picture borders with the kernel
        # finding local maxima
        image_max = ndi.maximum_filter(accumulator, size=20,
                                       mode='constant')  # finding local maximums in the convolution result (centers of circles)
        coordinates = peak_local_max(accumulator, min_distance=20, num_peaks=10)
        threshold = threshold_arg * (2 * np.pi * r)
        answers = list(filter(lambda x: (accumulator[x[0]][x[1]] > threshold),
                              coordinates))  # checking only for local maximas over certain threshold
        for ans in answers:  # appending the results while removing too close circles
            flag = False
            for point in circles_center1:
                if abs(ans[0] - point[1]) + abs(ans[1] - point[0]) < 20:
                    if accumulator[ans[0]][ans[1]] > accumulator[point[1]][point[0]]:
                        circles_center1.remove(point)  # remove the lower value point
                        circles_center1.append([ans[1], ans[0], r])
                    else:
                        flag = True
            if flag == False:
                circles_center1.append([ans[1], ans[0], r])

    circles_center, circles_radius = filter_close_circles(circles_center1, verbose)

    return circles_center, circles_radius


def plotCircles(image, circles_center, circles_radius):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.title.set_text("Detected Circles")

    # Replace with code to mark the center of the circles by a yellow point
    ax.plot([tpl[0] for tpl in circles_center], [tpl[1] for tpl in circles_center], 'o', color='yellow')

    # Replace with code to draw the circumference of the circles in blue
    for center_coordinates, radius in zip(circles_center, circles_radius):
        circ = plt.Circle(center_coordinates, radius, color='blue', fill=False)
        ax.add_artist(circ)
    # plt.imsave("output.png", image)
    return fig


def couchonnet_finder(kmeans_img, verbose):
    hsv = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2HSV)
    kernel = createKernel(2)

    RGB_green_lower = np.array([0, 200, 0], dtype="uint8")
    RGB_green_upper = np.array([100, 255, 100], dtype="uint8")

    blue_lower = np.array([94, 80, 2], dtype="uint8")
    blue_upper = np.array([126, 255, 255], dtype="uint8")

    RGB_purple_lower = np.array([20, 20, 280], dtype="uint8")
    RGB_purple_upper = np.array([100, 100, 310], dtype="uint8")

    orange_lower = np.array([35, 30, 97], dtype="uint8")
    orange_upper = np.array([50, 70, 100], dtype="uint8")

    HSV_purple_lower = np.array([130 ,133, 141])
    HSV_purple_upper = np.array([132, 170, 241])

    lower = HSV_purple_lower
    upper = HSV_purple_upper

    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(kmeans_img, kmeans_img, mask=mask)

    accumulator = cv2.filter2D(output, ddepth=cv2.CV_32S,
                               kernel=kernel)
    if verbose:
        util.save_photo('build/hsv_color_detection_part.jpg', hsv, True)
        util.save_photo('build/color_detection_output.jpg', output, True)
        util.save_photo('build/color_detection_mask.jpg', mask, True)
        util.save_photo('build/color_detection_acc.jpg', accumulator, True)
    coordinate = [0, 0]
    max = 0
    for count, x in enumerate(accumulator):
        for count2, y in enumerate(x):
            sum = y[0] + y[1] + y[2]
            if sum > max:
                max = sum
                coordinate = [count, count2]

    return coordinate


class Ball:
  def __init__(self, img_ball_only, radius, center, edge_map, sum_edges, team_num):
    self.img_ball_only = img_ball_only
    self.radius = radius
    self.center = center
    self.edge_map = edge_map
    self.sum_edges = sum_edges
    self.team_num = team_num

def teams_detection(original_image, circles_center, circles_radius, verbose):
    h, w = original_image.shape[:2]
    ball_masked_imgs = []
    for center, radius in zip(circles_center, circles_radius):
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = np.zeros(original_image.shape[:2],np.uint8)
        mask[dist_from_center <= 0.8*radius] = 1
        img_ball_only = cv2.bitwise_and(original_image,original_image,mask = mask)
        ball_masked_imgs.append(Ball(img_ball_only, radius, center, None, None, None))
    if verbose:
        for i in range(len(ball_masked_imgs)):
            util.save_photo('build/ball_masked_img{}.jpg'.format(i), ball_masked_imgs[i].img_ball_only, True)
    
    for i in range(len(ball_masked_imgs)):
        curr_ball = ball_masked_imgs[i]
        edge_map = cv2.Canny(curr_ball.img_ball_only, 200, 500) / 255
        curr_ball.edge_map = edge_map
        curr_ball.sum_edges = (np.sum(edge_map) - 2*curr_ball.radius*np.pi)/ (curr_ball.radius*np.pi**2)
        if verbose:
            util.save_photo('build/edge_map{}.jpg'.format(i), edge_map*255, True)
            print("ball", i, "sum", curr_ball.sum_edges)
        curr_ball.team_num = int(curr_ball.sum_edges > 0.2)
    return ball_masked_imgs
        



def hough_circles(original_image, kmeans, edges, threshold_arg, hough_radius_range, verbose):
    # Detect circles in the image using Hough transform
    circles_center, circles_radius = HoughCircles(edges, threshold_arg, hough_radius_range, verbose)
    # Teams detection
    balls = teams_detection(original_image, circles_center, circles_radius, verbose)

    # Plot the detected circles on top of the original coins image
    cimg = original_image.copy()
    for curr_ball in balls:
        # draw the outer circle
        if curr_ball.team_num:
            color = (0,255,0)
        else:
            color = (255, 0 ,0)
        cimg = cv2.circle(cimg, (curr_ball.center[0], curr_ball.center[1]), curr_ball.radius, color, 2)
        # draw the center of the circle
        cimg = cv2.circle(cimg, (curr_ball.center[0], curr_ball.center[1]), 2, (0, 0, 255), 3)

    # cochonnet detection--
    co_center = couchonnet_finder(kmeans, verbose) #TODO to change the original image for the after kmeans image
    cimg = cv2.circle(cimg, (co_center[1], co_center[0]), 10, (255, 0, 0), 10)


    if verbose:
        fig = plotCircles(edges, circles_center, circles_radius)
        fig.savefig("build/circles_on_edge_map.png", dpi=400)
    if len(circles_center) == 0:
        cimg = original_image.copy()


    return cimg
