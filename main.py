import cv2
import sys
from src import *
import argparse
import shutil
import os
import numpy as np

class Step:
  def __init__(self, function, name):
    self.function = function
    self.name = name

def bilateral_and_blur_step(img, verbose):
    img = cv2.bilateralFilter(np.float32(img),15,800,800)
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img,-1,kernel)
    return img

def blur_step(img, verbose):
    BLUR_WEIGHT = 500
    ksize = (BLUR_WEIGHT, BLUR_WEIGHT)
    return cv2.blur(img, ksize, cv2.BORDER_DEFAULT)

def kmeans_step(img, verbose):
    N_CLUSTERS = 3
    return kmeans.kmeans(img, cv2.COLOR_BGR2LAB, N_CLUSTERS, "max", original_image, verbose)

def edge_detector_step(img, verbose):
    return util.edgeDetector(img, 2, verbose)

def burn_blob_frame_step(img, verbose):
    return util.burn_blob_frame_step(img, 25, verbose)

def hough_lines_step(img, verbose):
    return lines.houghLines(img, original_image, verbose)

def hough_circles_step(img, verbose):
    THRESHOLD = 0.338
    return houghCircles.hough_circles(img, THRESHOLD, original_image, verbose)

def train():
    img = cv2.imread("images/day1/image38.png")
    ksize = (10, 10)
    img = cv2.blur(img, ksize, cv2.BORDER_DEFAULT)
    util.save_photo('build/blur_zoom_ball.jpg', img, True)
    # util.save_photo('build/hough_zoom_ball.jpg', util.hough(img, True), True)
    # util.pca()

main_plan = [Step(blur_step, "blur_step"), Step(kmeans_step, "kmeans_step"), Step(burn_blob_frame_step, "burn_blob_frame_step"), Step(edge_detector_step, "edge_detector_step"), Step(hough_circles_step, "hough_circles_step")]
plans     = [ main_plan ] 

def main(arguments):
    global original_image
    parser = argparse.ArgumentParser(description="Petanque recognition")
    parser.add_argument("-s","--step", type=int, help="Input step number to start from", default=1)
    parser.add_argument("-e","--end_step", type=int, help="Input step number to end in", default=len(main_plan))
    parser.add_argument("-i","--image_num", type=int, help="Input image number", default=1)
    parser.add_argument("-p","--plan_num", type=int, help="Input plan number", default=0)
    parser.add_argument("-q","--quick", action='store_true', help="quick run, without saving any step")
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity level")
    parser.add_argument("-t","--train", action='store_true', help="train mode")
    parser.add_argument("-c","--clean", action='store_true', help="clean build")
    parser.add_argument("-N","--no_previous_step", action='store_true', help="run from step 'step' with the original photo")   
    args = parser.parse_args(arguments)

    if args.clean:
        for root, dirs, files in os.walk('./build/'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))
        print("remove ./build content")
    
    if args.train:
        return train()
    if args.plan_num >= len(plans):
        print("plan_num: ", args.plan_num," not supported")
        return 1
    if args.image_num < 0 or args.step > 17:
        print("image_num: ", args.image_num," not supported")
        return 2
    if args.step < 1 or args.step > len(main_plan):
        print("step: ", args.step," not supported")
        return 3
    elif args.step  == 1 or args.no_previous_step:
        print("image_num: ", args.image_num,", start step: ", args.step)
        if args.no_previous_step:
            print("no_previous_step choosen!!")
        img_path = 'images/day1/image{}.jpg'.format(args.image_num)
        img = cv2.imread(img_path)
        original_image = img.copy()
        if not args.quick:
            util.save_photo('build/image{}.jpg'.format(args.image_num),img, True)
    else:
        img_path = 'images/day1/image{}.jpg'.format(args.image_num)
        original_image = cv2.imread(img_path)
        print("image_num: ", args.image_num,", start step: ", args.step)
        img = cv2.imread('build/image{}_{}_{}.jpg'.format(args.image_num, args.step-1, plans[args.plan_num][args.step-2].name))

    for i in range(args.step - 1, args.end_step):
        next_step = plans[args.plan_num][i]
        img = next_step.function(img, args.verbose)
        if not args.quick:
            util.save_photo('build/image{}_{}_{}.jpg'.format(args.image_num, i+1, next_step.name),img, True)
    if args.quick:
        util.save_photo('build/image{}_step{}.jpg'.format(args.image_num, i+1),img, True)
    return 0

if __name__ == '__main__':
    result = main(sys.argv[1:])
    sys.exit(result)
