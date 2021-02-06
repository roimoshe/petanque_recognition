from . import util
import cv2
import numpy as np

def houghLines(img):
    # Apply edge detection method on the image
    edges = util.edgeDetector(img)
    cv2.imshow("edges", edges)
    # This returns an array of r and theta values
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 400  # minimum number of pixels making up a line
    max_line_gap = 300  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    print(lines)
    
    print(img.shape)
    for line in lines:
        for x1,y1,x2,y2 in line:
            for i in range(img.shape[1]):
                if x1==x2:
                    continue
                for j in range(min(int(y1+(i-x1)*(y2-y1)/(x2-x1)),img.shape[0])) :
                    img[j][i] *= 0
            # cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
    
    # cv2.imwrite('tmp/lines.jpg',line_image)
    # # Draw the lines on the  image
    # final = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    # cv2.imwrite('tmp/houghlines5.jpg',final)
    return img
    
    # # The below for loop runs till r and theta values
    # # are in the range of the 2d array
    # for line in lines:
    #     for r, theta in line:
    #         # Stores the value of cos(theta) in a
    #         a = np.cos(theta)

    #         # Stores the value of sin(theta) in b
    #         b = np.sin(theta)

    #         # x0 stores the value rcos(theta)
    #         x0 = a * r

    #         # y0 stores the value rsin(theta)
    #         y0 = b * r

    #         # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
    #         x1 = int(x0 + 1000 * (-b))

    #         # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
    #         y1 = int(y0 + 1000 * (a))

    #         # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
    #         x2 = int(x0 - 1000 * (-b))

    #         # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
    #         y2 = int(y0 - 1000 * (a))

    #         # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).

    #         cv2.line(img, (x1, y1), (x2, y2), ((0,0,255)), 2)

    # cv2.imshow("Result Image", img)
    # cv2.waitKey(0)