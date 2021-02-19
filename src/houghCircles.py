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
        for i,center in enumerate(circles_center):
            if abs(center[0]-x+center[1]-y) < z/2:
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
        for center,z in zip(circles_center, circles_radius):
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
        for ans in answers:  # appending the results
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


def hough_circles(img, threshold_arg, hough_radius_range, original_image, verbose):
    edges = img
    # Step 2: Detect circles in the image using Hough transform
    circles_center, circles_radius = HoughCircles(edges, threshold_arg, hough_radius_range, verbose)
    # Step 3: Plot the detected circles on top of the original coins image
    for i in range(len(circles_center)):
        # draw the outer circle
        cimg = cv2.circle(original_image, (circles_center[i][0], circles_center[i][1]), circles_radius[i], (0, 255, 0), 2)
        # draw the center of the circle
        cimg = cv2.circle(cimg, (circles_center[i][0], circles_center[i][1]), 2, (0, 0, 255), 3)
    
    if verbose:
        fig = plotCircles(img, circles_center, circles_radius)
        fig.savefig("build/circles_on_edge_map.png", dpi=400)
    if len(circles_center)==0:
        cimg=original_image.copy()
    return cimg