from . import util
import cv2
import numpy as np

def houghLines(edges, original_image, verbose):
    # This returns an array of r and theta values
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 400  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    # if verbose:
        # print(lines)

    for line in lines:
        x1,y1,x2,y2 = line[0]
        # if verbose:
            # print(x1,y1,x2,y2)
        edges = cv2.line(edges,(x1,y1),(x2,y2),(80,80,200),5)
    
    if verbose:
        util.save_photo('build/apply_frame_img.jpg', apply_frame(original_image, lines), True)
    return edges

def apply_frame(img, lines):
    MIN = 0
    MAX = 1
    min_max=[[np.inf,np.NINF]]*img.shape[1]
    for i in range(img.shape[1]):
        for line in lines:
            for x1,y1,x2,y2 in line:
                if x1==x2:
                    continue
                y_point = int(y1+(i-x1)*(y2-y1)/(x2-x1))
                if min(y_point,img.shape[0]) < min_max[i][MIN]:
                    min_max[i][MIN] = min(y_point,img.shape[0])
                    print(i, min_max[i][MIN])
                if max(y_point,0) > min_max[i][MAX]:
                    min_max[i][MAX] = max(y_point,img.shape[0])
    mask = np.zeros(img.shape[:2],np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mask[i][j] = int(i < min_max[j][MAX] and i > min_max[j][MIN]) * 255
    return cv2.bitwise_and(img,img,mask = mask)