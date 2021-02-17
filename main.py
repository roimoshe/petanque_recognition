import cv2
import sys
from src import *
import argparse
import shutil
import os
import numpy as np

def step1(img, verbose):
    # img = cv2.bilateralFilter(np.float32(img),15,800,800)
    # kernel = np.ones((5,5),np.float32)/25
    # return cv2.filter2D(img,-1,kernel)
    return img

def step2(img, verbose):
    if verbose:
        print("started step 2")
    return kmeans.kmeans(img, cv2.COLOR_BGR2LAB, verbose)

def step3(img, verbose):
    return util.edgeDetector(img)

def step4(img, verbose):
    return lines.houghLines(img, verbose)

def step5(img, verbose):
    return util.hough(img, verbose)

def train():
    util.extract_frames()

steps = [step1, step2, step3, step4, step5]

def main(arguments):
    parser = argparse.ArgumentParser(description="Petanque recognition")
    parser.add_argument("-s","--step", type=int, help="Input step number to start from", default=1)
    parser.add_argument("-e","--end_step", type=int, help="Input step number to end in", default=len(steps))
    parser.add_argument("-i","--image_num", type=int, help="Input image number", default=1)
    parser.add_argument("-q","--quick", action='store_true', help="Input image number")
    parser.add_argument("-v","--verbose", action='store_true', help="verbosity level", default=True)
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
        return 0
    
    if args.train:
        return train()

    if args.image_num < 0 or args.step > 17:
        print("image_num: ", args.image_num," not supported")
        return 1
    if args.step < 1 or args.step > len(steps):
        print("step: ", args.step," not supported")
        return 2
    elif args.step  == 1 or args.no_previous_step:
        print("image_num: ", args.image_num,", start step: ", args.step)
        if args.no_previous_step:
            print("no_previous_step choosen!!")
        img_path = 'images/day1/image{}.jpg'.format(args.image_num)
        img = cv2.imread(img_path)
        if not args.quick:
            util.save_photo('build/image{}.jpg'.format(args.image_num),img, True)
    else:
        print("image_num: ", args.image_num,", start step: ", args.step)
        img = cv2.imread('build/image{}_step{}.jpg'.format(args.image_num, args.step - 1))

    for i in range(args.step - 1, args.end_step):
        img = steps[i](img, args.verbose)
        if not args.quick:
            util.save_photo('build/image{}_step{}.jpg'.format(args.image_num, i+1),img, True)
    if args.quick:
        util.save_photo('build/image{}_step{}.jpg'.format(args.image_num, i+1),img, True)
    return 0

if __name__ == '__main__':
    result = main(sys.argv[1:])
    sys.exit(result)
