# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from PIL import Image
import time
# from matplotlib import pyplot as plt
import cv2
import time

#FUCNTIONS
def red_filter(cropped):
    cv_im =np.array(cropped)
    img_hsv_r = cv2.cvtColor(cv_im, cv2.COLOR_RGB2HSV)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv_r, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv_r, lower_red, upper_red)

    # join masks
    mask = mask1
    output_hsv_r = img_hsv_r.copy()
    output_hsv_r[np.where(mask==0)] = 0

    return (output_hsv_r)

def yellow_filter(cropped):
    cv_im_yellow =np.array(cropped)
    img_hsv_y = cv2.cvtColor(cv_im_yellow, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([21,100,100])
    upper_yellow = np.array([30,255,255])
    mask = cv2.inRange(img_hsv_y, lower_yellow, upper_yellow)

    output_hsv_y = img_hsv_y.copy()
    output_hsv_y[np.where(mask==0)] = 0

    return (output_hsv_y)

def green_filter(cropped):
    cv_im_green =np.array(cropped)
    img_hsv_g = cv2.cvtColor(cv_im_green, cv2.COLOR_RGB2HSV)
    lower_green = np.array([40,40,40])
    upper_green = np.array([70,255,255])
    mask0 = cv2.inRange(img_hsv_g, lower_green, upper_green)

    # upper mask (170-180)
    lower_green = np.array([50,170,50])
    upper_green = np.array([255,180,255])
    mask1 = cv2.inRange(img_hsv_g, lower_green, upper_green)

    # join masks
    mask = mask1

    # or your HSV image, which I *believe* is what you want
    output_hsv_g = img_hsv_g.copy()
    output_hsv_g[np.where(mask==0)] = 0

    return (output_hsv_g)

def plotHist(bins, values, thickness, channel, yLbl, yLim = False):

    #second channel
    max_list2 = max_idx_rank(values[2])
    mean2 = max_list2[0]
    bimodal2 = is_bimodal(max_list2, values[2])

    #print (max_list2)
    summ = 0
    me = []
    for maxin in max_list2:
        summ += values[2][maxin]

        if (values[2][maxin] >0):
            me.append(maxin)

    return summ, np.mean(me)

def color_isolate(image, channel):
    if channel == "hsv":
        # Convert absimage to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        # Create color channels
        h = hsv[:,:,0]
        s = hsv[:,:,1]
        v = hsv[:,:,2]

        return h, s, v
    else:
        rgb = np.copy(image)
        # Create color channels
        r = rgb[:,:,0]
        g = rgb[:,:,1]
        b = rgb[:,:,2]

        return r, g, b

def matrix_scalar_mul(matrix, scalar):
    new = []
    for i in range(len(matrix)):
        new.append(matrix[i] * scalar)
    return new

def matrix_multiplication(matrixA, matrixB):
    product = []

    if len(matrixA) != len(matrixB):
        raise ValueError('list must be the same size, A:', len(matrixA), 'B:', len(matrixB))

    for i in range(len(matrixA)):
        product.append(matrixA[i] * matrixB[i])

    return product

# return list index which has a maximum value
def max_idx(yvals, ranges):
    mx = 0
    j = 0
    for i in ranges:
        if yvals[i] > mx:
            mx = yvals[i]
            j = i
    return j

# return list index which has a minimum value
def min_idx(yvals, ranges):
    mn = max(yvals)
    j = 0
    for i in ranges:
        if yvals[i] < mn and yvals[i] > 0:
            mn = yvals[i]
            j = i
    return j

# return the top x indicies from a list
def max_idx_rank(yvals):
    indicies = set(range(len(yvals)))
    # creat a list to append max bins
    max_list = []
    # create set to perform set operations
    max_set = set()
    intersect = indicies - max_set

    # rank first 8 bins
    for i in range(0, 8):
        # append next maximum value
        max_list.append(max_idx(yvals, intersect))
        # add value to set list
        max_set.add(max_list[-1])
        # remove bin from bin list
        intersect = indicies - max_set

    return max_list

# function to determine if the distribution is bimodal or normal
def is_bimodal(max_list, values):
    difference = []
    for i in range(len(max_list)-1):
        for j in range(max_list[i], max_list[i+1]):
            if values[j] == 0:
                return True

    return False

def yaxis_hists(rgb_image, channel):
    # seperate image out into different channels of color space
    c1, c2, c3 = color_isolate(rgb_image, 'hsv')
    # Sum components over all coloumns for each row (axis = 1)
    hist_sum = []
    c1_sum = np.sum(c1[:,:], axis=1)
    c2_sum = np.sum(c2[:,:], axis=1)
    c3_sum = np.sum(c3[:,:], axis=1)

    #get baselines
    base1 = np.median(c1_sum)
    base2 = np.median(c2_sum)
    base3 = np.median(c3_sum)

    # split histrogram around the median
    c1_norm = matrix_scalar_mul((c1_sum - base1).tolist(), -1)
    c2_norm = (c2_sum - base2).tolist()
    c3_norm = (c3_sum - base3).tolist()

    # get rid of negative values
    #np.nan
    c1_norm = [0 if x < 0 else x for x in c1_norm]
    c2_norm = [0 if x < 0 else x for x in c2_norm]
    c3_norm = [0 if x < 0 else x for x in c3_norm]

    # package as 2D list
    hist_vals = []
    hist_vals.append(c1_norm)
    hist_vals.append(c2_norm)
    hist_vals.append(c3_norm)

    # get bins
    bin_edges = range(rgb_image.shape[0])

    return bin_edges, hist_vals

def feature_value(rgb_image, plot = False):
    ## TODO: Convert absimage to HSV color space
    ## TODO: Create and return a feature value and/or vector

    # calculate HSVspace over the height on the traffic light
    bins, values = yaxis_hists(rgb_image, 'hsv')

    # for testing purposes
    if plot == True:
        a=plotHist(bins, values, 0.8, 'hsv', 'y-dist')

    return a

def analyze_color(dictionary, img):
    res = dict()
    a = []
    for i in range(0,len(dictionary[9])):
        a.append(dictionary[9][i][0])

    for j in range(len(a)):
        y1 = int(a[j][1])
        y2 = int(a[j][3])
        x1 = int(a[j][0])
        x2 = int(a[j][2])

        # cv2.rectangle(img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
        # cv2.imwrite('img_{}.jpg'.format(time.time()),img)

        # image = cv2.imread(img)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped = image[y1:y2, x1:x2]

        # cv2.rectangle(image,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),3)
        # cv2.imwrite('original_{}.jpg'.format(time.time()),image)
        # cv2.imwrite('{}.jpg'.format(time.time()),cropped)

        output_hsv_r = red_filter(cropped)
        output_hsv_y = yellow_filter(cropped)
        output_hsv_g = green_filter(cropped)

        res_r = feature_value(output_hsv_r, True)
        res_y = feature_value(output_hsv_y, True)
        res_g = feature_value(output_hsv_g, True)

        h = y2 - y1

        val = []
        position = []

        val.extend([int(res_r[0]),int(res_y[0]),int(res_g[0])])
        position.extend([res_r[1],res_y[1],res_g[1]])

        color = np.where(val == np.amax(val))

        indx = color[0][0]

        pos = position[indx]/h

        ret_col = None
        if(indx == 0 and pos <= 0.45):
            ret_col = 'RED'
        elif(indx == 2 and pos >= 0.55):
            ret_col = 'GREEN'
        elif(indx == 1 and pos > 0.45 and pos < 0.55):
            ret_col = 'YELLOW'
        else:
            # ret_col = 'BOX W/O LIGHTS'
            pass

        # coors = []
        # coors.extend([x1,y1,x2,y2])

        # dict_ind = ret_col + str(j)
        # res[ret_col + '_' + str(j)] = coors
        if ret_col:
            if ret_col in res.keys():
                res[ret_col] += 1
            else:
                res[ret_col] = 1
    return res
def state_predict(a):
    max=0
    current_state=None
    for key,num in a.items():
        if num > max:
            current_state = key
            max = num
    return current_state


