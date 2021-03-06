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

class Ball:
  def __init__(self, img_ball_only, radius, center, edge_map, sum_edges, team_num, center_world_position, distance):
    self.img_ball_only = img_ball_only
    self.radius = radius
    self.center = center
    self.center_world_position = center_world_position
    self.edge_map = edge_map
    self.sum_edges = sum_edges
    self.team_num = team_num
    self.distance = distance
class Cochonnet:
  def __init__(self, center_image_pos, center_world_pos):
    self.center_image_pos = center_image_pos
    self.center_world_pos = center_world_pos

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
            if abs(center[0] - x + center[1] - y) < z:
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

def teams_detection(original_image, circles_center, circles_radius, verbose, balls, image_num):
    h, w = original_image.shape[:2]
    # mask ball only
    for center, radius in zip(circles_center, circles_radius):
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        mask = np.zeros(original_image.shape[:2],np.uint8)
        mask[dist_from_center <= 0.8*radius] = 1
        img_ball_only = cv2.bitwise_and(original_image,original_image,mask = mask)
        balls.append(Ball(img_ball_only, radius, center, None, None, None, None, None))
    if verbose:
        for i in range(len(balls)):
            util.save_photo('build/ball_masked_ball{}_img{}.jpg'.format(i, image_num), balls[i].img_ball_only, True)
    
    for i,curr_ball in enumerate(balls):
        edge_map = cv2.Canny(curr_ball.img_ball_only, 200, 500) / 255
        if verbose:
            util.save_photo('build/ball_masked_edge_ball{}_img{}.jpg'.format(i, image_num), edge_map*255, True)
        curr_ball.edge_map = edge_map
        curr_ball.sum_edges = (np.sum(edge_map) - 2*curr_ball.radius*np.pi)/ (curr_ball.radius*np.pi**2)
        curr_ball.team_num = int(curr_ball.sum_edges > 0.35) # all the range 0.41-0.31 is good setup 2 image25
        if verbose:
            util.save_photo('build/edge_map{}.jpg'.format(i), edge_map*255, True)
            print("ball", i, "sum", curr_ball.sum_edges)
            print("curr_ball.sum_edges = ", curr_ball.sum_edges, "curr_ball.center", curr_ball.center)

def hough_circles(original_image, kmeans, edges, threshold_arg, hough_radius_range, verbose, balls, cochonnet, image_num):
    # Detect circles in the image using Hough transform
    circles_center, circles_radius = HoughCircles(edges, threshold_arg, hough_radius_range, verbose)
    # Teams detection
    teams_detection(original_image, circles_center, circles_radius, verbose, balls, image_num)

    # Plot the detected circles on top of the original image
    cimg = original_image.copy()
    for curr_ball in balls:
        # draw the outer circle
        if curr_ball.team_num:
            color = (0,255,0)
        else:
            color = (255, 0 ,0)
        cimg = cv2.circle(cimg, (curr_ball.center[0], curr_ball.center[1]), int(0.85*curr_ball.radius), color, 6)
        # draw the center of the circle
        cimg = cv2.circle(cimg, (curr_ball.center[0], curr_ball.center[1]), 2, (0, 0, 0), 3)

    # cochonnet detection--
    co_center = couchonnet_finder(kmeans, verbose) #TODO to change the original image for the after kmeans image
    cimg = cv2.circle(cimg, (co_center[1], co_center[0]), 10, (0, 0, 255), 10)
    cochonnet.append(co_center[0])
    cochonnet.append(co_center[1])

    if len(circles_center) == 0:
        cimg = original_image.copy()


    return cimg
