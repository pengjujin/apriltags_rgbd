#!/usr/bin/env python3
# Author: Amy Phung

# Python Imports
import pandas as pd
import numpy as np
import math

# TODO: remove hardcode
WINDOW_SIZE = 20 # consider up to last 20 detections
MIN_PTS = 10 # minimum number of measurements before applying filters

# TODO: add time consideration - discard stored points if too long since last
# update

# TODO: put these in a utils file
def median(x):
    m,n = x.shape
    middle = np.arange((m-1)>>1,(m>>1)+1)
    x = np.partition(x,middle,axis=0)
    return x[middle].mean(axis=0)

def removeOutliers(data,thresh=2.0):
    m = median(data)
    s = np.abs(data-m)
    return data[(s<median(s)*thresh).all(axis=1)]



class CornerFilter():
    def __init__(self):
        self.tag_ids = []
        self.detected_corners = []

        # Using list for speed
        # Usage: self.detected_corners[tag_idx][corner_idx] - will provide list
        # of 3d points

    #
    #
    #     queue.pop(0)
    #     queue.append()
    #
    #     return corner_estimate

    def updateEstimate(self, tag_id, corners):
        if tag_id not in self.tag_ids:
            self.tag_ids.append(tag_id)
            self.detected_corners.append([])
            for i in range(4):
                self.detected_corners[-1].append([corners[i]])
            return None # We don't do anything with just one estimate

        idx = self.tag_ids.index(tag_id)
        for i in range(4):
            self.detected_corners[idx][i].append(corners[i])

        print("TAG ID" + str(tag_id))




        print(len(self.detected_corners[idx]))
        print(len(self.detected_corners[idx][0]))
        print(len(self.detected_corners[idx][3]))
        #     self.detected_corners.append([])
        #     self.detected_corners[-1].append(corners)
        #     return None # We don't do anything with just one estimate
        #
        # idx = self.tag_ids.index(tag_id)
        # self.detected_corners[idx].append(corners)
        #
        # # [Which tag][Which measurement][which corner][axis]
        # # print(tag_id)
        # print("here")
        # # print(np.array(self.detected_corners).shape)
        # print(self.detected_corners[0][0][0][0])
        # removeOutliers

        # Apply filter

                        # Restrict window size
