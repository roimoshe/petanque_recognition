#!/usr/bin/env python
import sys, os
import argparse
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib.image as mpimg


def edgeDetector(image):
    image = cv2.bilateralFilter(image, 9, 75, 75)
    cv2.waitKey(0)
    image_uint8 = image.astype(np.uint8)
    edge_map = cv2.Canny(image_uint8, 100, 240)
    return edge_map



def houghLines(img):

    # Convert the img to grayscale

    # Apply edge detection method on the image
    edges = edgeDetector(img)
    cv2.imshow("edges", edges)
    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    print(lines)
    # The below for loop runs till r and theta values
    # are in the range of the 2d array
    for line in lines:
        for r, theta in line:
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a * r

            # y0 stores the value rsin(theta)
            y0 = b * r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).

            cv2.line(img, (x1, y1), (x2, y2), ((0,0,255)), 2)

    cv2.imshow("Result Image", img)
    cv2.imwrite('linesDetected.jpg', img)
    cv2.waitKey(0)


def main(arguments):
    img = cv2.imread('images/image5.jpg', cv2.IMREAD_GRAYSCALE)
    # returning_circles(img)
    houghLines(img)
    return 0


if __name__ == '__main__':
    result = main(sys.argv[1:])
    sys.exit(result)