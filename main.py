# commands:
# photo: python3 main.py -i 55 -c -F photo
# video: python3 main.py -i 4 -c -F video -f 4
import cv2
import sys
from src import *
import argparse
import shutil
import os
import numpy as np

# general params
BLUR_SIZE = 500
BURNING_SIZE = 100
N_CLUSTERS = 3
# format spesific params
HOUGH_THRESHOLD_VIDEO = 0.338
HOUGH_THRESHOLD_PHOTO = 0.33

HOUGH_RADIUS_RANGE_VIDEO = [12,18]
HOUGH_RADIUS_RANGE_PHOTO = [28,40]
class Step:
  def __init__(self, function, name):
    self.function = function
    self.name = name
class Parameters:
  def __init__(self, hough_threshold, hough_radius_range, burning_size, blur_size, n_clusters):
    self.hough_threshold    = hough_threshold
    self.hough_radius_range = hough_radius_range
    self.burning_size       = burning_size
    self.blur_size          = blur_size
    self.n_clusters         = n_clusters

def bilateral_and_blur_step(img, verbose):
    img = cv2.bilateralFilter(np.float32(img),15,800,800)
    kernel = np.ones((5,5),np.float32)/25
    return cv2.filter2D(img,-1,kernel)
    return img

def blur_step(img, verbose, params):
    BLUR_WEIGHT = params.blur_size
    ksize = (BLUR_WEIGHT, BLUR_WEIGHT)
    return cv2.blur(img, ksize, cv2.BORDER_DEFAULT)

def kmeans_step(img, verbose, params):
    return kmeans.kmeans(img, cv2.COLOR_BGR2LAB, params.n_clusters, "max", original_image, verbose)

def edge_detector_step(img, verbose, params):
    return util.edgeDetector(img, 2, verbose)

def burn_blob_frame_step(img, verbose, params):
    return util.burn_blob_frame_step(img, params.burning_size, verbose)

def hough_lines_step(img, verbose, params):
    return lines.houghLines(img, original_image, verbose)

def hough_circles_step(img, verbose, params):
    return houghCircles.hough_circles(img, params.hough_threshold, params.hough_radius_range, original_image, verbose)


def train():
    util.position()

main_plan = [Step(blur_step, "blur_step"), Step(kmeans_step, "kmeans_step"), Step(burn_blob_frame_step, "burn_blob_frame_step"), Step(edge_detector_step, "edge_detector_step"), Step(hough_circles_step, "hough_circles_step")]
plans     = [ main_plan ] 

def main(arguments):
    global original_image
    parser = argparse.ArgumentParser(description="Petanque recognition")
    parser.add_argument("-s","--step", type=int, help="Input step number to start from", default=1)
    parser.add_argument("-e","--end_step", type=int, help="Input step number to end in", default=len(main_plan))
    parser.add_argument("-i","--image_num", type=int, help="Input image number", default=1)
    parser.add_argument("-p","--plan_num", type=int, help="Input plan number", default=0)
    parser.add_argument("-f","--frame", type=int, help="Input frame number", default=-1)
    parser.add_argument("-q","--quick", action='store_true', help="quick run, without saving any step")
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity level")
    parser.add_argument("-t","--train", action='store_true', help="train mode")
    parser.add_argument("-c","--clean", action='store_true', help="clean build")
    parser.add_argument("-N","--no_previous_step", action='store_true', help="run from step 'step' with the original photo")
    parser.add_argument("-F","--image_format", type=str, help="image format - video/photo", default="photo")
    args = parser.parse_args(arguments)
    
    photo_params = Parameters(HOUGH_THRESHOLD_PHOTO, HOUGH_RADIUS_RANGE_PHOTO, BURNING_SIZE, BLUR_SIZE, N_CLUSTERS)
    video_params = Parameters(HOUGH_THRESHOLD_VIDEO, HOUGH_RADIUS_RANGE_VIDEO, BURNING_SIZE, BLUR_SIZE, N_CLUSTERS)

    if args.image_format == "photo":
        params = photo_params
        img_path = 'images/day1/photos/image{}.jpg'.format(args.image_num)
    elif args.image_format == "video":
        params = video_params
        img_path = 'images/day1/videos/video{}frame{}.jpg'.format(args.image_num, args.frame)
    else:
        print("image_format: ", args.image_format," not supported")
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
        img = next_step.function(img, args.verbose, params)
        if not args.quick:
            util.save_photo('build/{}{}_{}_{}.jpg'.format(args.image_format,args.image_num, i+1, next_step.name),img, True)
    if args.quick:
        util.save_photo('build/image{}_step{}.jpg'.format(args.image_num, i+1),img, True)
    return 0

if __name__ == '__main__':
    result = main(sys.argv[1:])
    sys.exit(result)
