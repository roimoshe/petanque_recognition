#!/usr/bin/env python
import sys,os
import argparse
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import matplotlib.image as mpimg

def edgeDetector(image):
    image = cv2.bilateralFilter(image,9,75,75)
    cv2.imshow('edges',image)
    cv2.waitKey(0)
    image_uint8 = image.astype(np.uint8) 
    edge_map = cv2.Canny(image_uint8,100,240)
    return edge_map


def createKernel(r):
  color = (255, 255, 255) 
  thickness = 1
  image = np.zeros(shape=[2*r+5, 2*r+5, 3], dtype=np.uint8)
  center_coordinates = (r+2, r+2)
  image = cv2.circle(image, center_coordinates, r, color, thickness)
  kernel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  kernel = kernel/255
  kernel = np.array(kernel).astype(int)
  return kernel

def HoughCircles(edge_map):
  new_edge_map = []
  for number in edge_map:
    new_edge_map.append(number / 255) #making the edge_map contain values of only [1,0]
  new_edge_map = np.array(new_edge_map, dtype=np.int)
 
  circles_center1 = []
  circles_center = []
  circles_radius = []

#for each radios
  #create circle for convolution
  #run the convolution on every pixle and save the result
  #find the resualts local maxima over a threshold if there are, and append them in the answer

  for r in range(31,41):
    kernel = createKernel(r)  #creating the kernel for the matching radius
    accumulator =  cv2.filter2D(new_edge_map ,ddepth=cv2.CV_32S,kernel=kernel) #convoluting the picture borders with the kernel 
    #finding local maxima
    image_max = ndi.maximum_filter(accumulator, size=20, mode='constant') #finding local maximums in the convolution result (centers of circles)
    coordinates = peak_local_max(accumulator, min_distance=20, num_peaks= 6) 
    threshold = 0.2*(2*np.pi*r)
    answers = list(filter(lambda x: (accumulator[x[0]][x[1]] > threshold), coordinates)) #checking only for local maximas over certain threshold
    for ans in answers: #appending the results 
      flag = False
      for point in circles_center1:
        if abs(ans[0]-point[1])+abs(ans[1]-point[0]) < 20:
          if accumulator[ans[0]][ans[1]] > accumulator[point[1]][point[0]]:
            circles_center1.remove(point)  #remove the lower value point
            circles_center1.append([ans[1], ans[0], r])
            print("ans[1]", ans[1], "ans[0]", ans[0]," r", r)
          else: 
            flag = True
      if flag == False: 
        circles_center1.append([ans[1], ans[0], r])
  
  for [x,y,z] in circles_center1:
    circles_center.append([x,y])
    circles_radius.append(z)

  return circles_center, circles_radius

def plotCircles(image,circles_center,circles_radius):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.title.set_text("Detected Circles")

    # Replace with code to mark the center of the circles by a yellow point
    ax.plot([tpl[0] for tpl in circles_center],[tpl[1] for tpl in circles_center], 'o', color='yellow')
    
    # Replace with code to draw the circumference of the circles in blue
    for center_coordinates, radius in zip(circles_center,circles_radius):
      circ = plt.Circle(center_coordinates, radius, color='blue', fill=False)
      ax.add_artist(circ)
    # plt.imsave("output.png", image)
    fig.savefig("output_savefig.png")

  
 
def hough(img):
    # fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True)
    plt.imshow(img, cmap='gray')
    # fig.savefig('out.png', dpi=400)
    # Step 1: Produce an edge map from the image using an edge detector
    edges = edgeDetector(img)
    cv2.imshow('edges',edges)
    cv2.waitKey(0)
    print("after edge show")
    # Step 2: Detect circles in the image using Hough transform
    circles_center, circles_radius = HoughCircles(edges)    
    # Step 3: Plot the detected circles on top of the original coins image
    plotCircles(img,circles_center,circles_radius)

def main(arguments):
    img = cv2.imread('images/image1.jpg', cv2.IMREAD_GRAYSCALE)
    hough(img)
    return 0

if __name__ == '__main__':
    result = main(sys.argv[1:])
    sys.exit(result)