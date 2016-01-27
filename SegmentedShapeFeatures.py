# -*- coding: utf-8 -*-

import sys
import os
import math

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

import TreeOfShapes as tos
from attributeGraph import *


# ===============================================================
# blob features
# ===============================================================
# 1. normalized gray level
def computeCFH(img, pix):
    N = len(pix)
    sumValue = 0
    grayValues = []
    for p in pix:
        sumValue = sumValue + img[p[1], p[0]]
        grayValues.append(img[p[1], p[0]])

    meanValue = sumValue / N
    g_x = np.array(grayValues, dtype='float64')
    sd = math.sqrt( sum((g_x - meanValue) * (g_x - meanValue)) / N )
    CF = (g_x - meanValue) / sd
    return np.ndarray.tolist(CF)

# 2. skeltezation
def computeSkeletonization(img, pix):
    src = np.zeros(img.shape,np.uint8)
    for p in pix:
        src[p[1], p[0]] = 255
    size = np.size(src)
    skel = np.zeros(src.shape,np.uint8)
    ret,src = cv2.threshold(src,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    while(not done):
        eroded = cv2.erode(src,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(src,temp)
        skel = cv2.bitwise_or(skel,temp)
        src = eroded.copy()
        zeros = size - cv2.countNonZero(src)
        if zeros==size:
            done = True
    SK =  float(cv2.countNonZero(skel)) / math.sqrt(float(len(pix)))
    return SK

# 3. perimeter
def computePerimeter(img, pix):
    src = np.zeros(img.shape,np.uint8)
    for p in pix:
        src[p[1], p[0]] = 255
    th = cv2.threshold(src, 127, 255,0)[1]
    contours = cv2.findContours(th, 2, 1)[0]
    PE = float(len(contours)) / math.sqrt(float(len(pix)))
    return PE

# 4. convex hull
def computeConvexHull(img, pix):
    src = np.zeros(img.shape,np.uint8)
    for p in pix:
        src[p[1], p[0]] = 255
    th = cv2.threshold(src, 127, 255,0)[1]
    contours = cv2.findContours(th,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]
    hull = cv2.convexHull(contours[0])
    cv2.drawContours(src, [hull], 0, 255, -1)
    CH = float(cv2.countNonZero(src)) / float(len(pix))
    return CH
# ===============================================================
# ===============================================================


def showImage(img, windowName="img1"):
    src = np.array(img.data, dtype="uint8").reshape(img.size, order='F')
    cv2.imshow(windowName, np.transpose(src))

def drawBlobs(src, tree, g, nodePixMap, im):
    darkBlobs = src.copy()
    brightBlobs = src.copy()

    darkCF = []
    darkCH = []
    darkSK = []
    darkPE = []
    brightCF = []
    brightCH = []
    brightSK = []
    brightPE = []

    for i in tree.iterateFromLeavesToRoot(False):
        if g.node[i]['leaf'] != True:
            continue
        else:
            pix = g.node[i]['pixels']
            childIndex = i
            while True:
                parentIndex = g.node[childIndex]['parent'][0]
                if g.node[parentIndex]['startPoint'] == True:
                    pix.extend(g.node[parentIndex]['pixels'])
                    childIndex = parentIndex
                    break
                else:
                    pix.extend(g.node[parentIndex]['pixels'])
                    childIndex = parentIndex

            # if parent node is root or not
            if g.node[childIndex]['parent'] != None:
                parentIndex = g.node[childIndex]['parent'][0]

            if g.node[childIndex]['grayLevel'] <= g.node[parentIndex]['grayLevel']:
                if len(pix) >= 10:
                    for j in pix:
                        darkBlobs[j[1], j[0], :] = [0,255,0]
                    darkCF.extend(computeCFH(im, pix))
                    darkCH.append(computeConvexHull(im, pix))
                    darkSK.append(computeSkeletonization(im, pix))
                    darkPE.append(computePerimeter(im, pix))
            else:
                if len(pix) >= 10:
                    for j in pix:
                        brightBlobs[j[1], j[0], :] = [0,0,255]
                    brightCF.extend(computeCFH(im, pix))
                    brightCH.append(computeConvexHull(im, pix))
                    brightSK.append(computeSkeletonization(im, pix))
                    brightPE.append(computePerimeter(im, pix))

    # compute feature vector (histogram)
    dark_CFH   = plt.hist(darkCF,   bins=25, range=(-3.8, 2.08), normed=True)[0]
    dark_CHH   = plt.hist(darkCH,   bins=15, range=(1, 2), normed=True)[0]
    dark_SKH   = plt.hist(darkSK,   bins=15, range=(1.2, 4), normed=True)[0]
    dark_PEH   = plt.hist(darkPE,   bins=15, range=(0, 2), normed=True)[0]
    bright_CFH = plt.hist(brightCF, bins=25, range=(-2.26, 3.35), normed=True)[0]
    bright_CHH = plt.hist(brightCH, bins=15, range=(1, 2), normed=True)[0]
    bright_SKH = plt.hist(brightSK, bins=15, range=(1.2, 4), normed=True)[0]
    bright_PEH = plt.hist(brightPE, bins=15, range=(3, 7.5), normed=True)[0]

    featureVector = np.r_[dark_CFH,
                          dark_CHH,
                          dark_SKH,
                          dark_PEH,
                          bright_CFH,
                          bright_CHH,
                          bright_SKH,
                          bright_PEH]

    return darkBlobs, brightBlobs, featureVector


def computeSSF(img_name, filterSize=10, display=False, writeImage=False, verbose=False):

    # read image
    print "%s" % img_name
    src = cv2.imread(img_name, 0).astype(np.uint32)
    org = cv2.imread(img_name, 1)
    gray = cv2.imread(img_name, 0)

    # Area filter
    if verbose:
        print "Area Filter"
    AF_img = tos.areaFilter(src, size=filterSize)

    # Constructing tree of shapes
    if verbose:
        print "Constructing tree of shapes"
    padding_img = tos.imagePadding(AF_img, 0)
    tree = tos.constructTreeOfShapes(padding_img, None)

    # attribute
    if verbose:
        print "Adding area & depth attributes"
    tos.addAttributeArea(tree)
    tos.addAttributeDepth(tree)

    # generate graph and node position using NetworkX
    if verbose:
        print "Create Graph"
    g, nodePixMap = createAttributeGraph(tree, padding_img)

    # compute blobs and texture features
    if verbose:
        print "Compute blobs & features"
    dark, bright, fVector = drawBlobs(org, tree, g, nodePixMap, gray)

    # output
    name, ext = os.path.splitext(img_name)
    outName = name + ".csv"
    np.savetxt(outName, [fVector], delimiter=",")

    if verbose:
        print "%s; done." % img_name

    # write blob images
    if writeImage:
        cv2.imwrite("%s_dark.png" % name, dark)
        cv2.imwrite("%s_bright.png" % name, bright)

    # display
    if display:
        showImage(src, "src")
        showImage(AF_img, "area filter 1")
        cv2.imshow("dark_blobs", dark)
        cv2.imshow("bright_blobs", bright)
        cv2.waitKey()

    return tree, g, nodePixMap


# Main ===========================================================
if __name__ == '__main__':

    if len(sys.argv) < 2:
        img_name = "images/D2-1.bmp"
    else:
        img_name = sys.argv[1]

    tree, g, nodePixMap = computeSSF(img_name, filterSize=10, display=False, writeImage=False, verbose=False)
