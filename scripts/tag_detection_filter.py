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

# TODO: debug why tool corner detections are all getting filtered out

# TODO: put these in a utils file
def median(x):
    m,n = x.shape
    middle = np.arange((m-1)>>1,(m>>1)+1)
    x = np.partition(x,middle,axis=0)
    return x[middle].mean(axis=0)

def removeOutliers(data,thresh=1.0):
    m = median(data)
    s = np.abs(data-m)
    return data[(s<abs(median(s)*thresh)).all(axis=1)]

class TagDetectionFilter():
    def __init__(self):
        self.tag_ids = []
        self.detected_corners = []
        self.normals = []

        # Using list for speed
        # Usage: self.detected_corners[tag_idx][corner_idx] - will provide list
        # of 3d points.

    #
    #
    #     queue.pop(0)
    #     queue.append()
    #
    #     return corner_estimate

    def updateEstimate(self, tag_id, corners, normal):
        if tag_id not in self.tag_ids:
            self.tag_ids.append(tag_id)
            self.detected_corners.append([])
            for i in range(4):
                self.detected_corners[-1].append([corners[i]])
            self.normals.append([normal])
            return None, None # We don't do anything with just one estimate

        idx = self.tag_ids.index(tag_id)
        for i in range(4):
            self.detected_corners[idx][i].append(corners[i])
        self.normals[idx].append(normal)

        if len(self.normals[idx]) > WINDOW_SIZE:
            self.normals[idx].pop(0)
            self.detected_corners[idx][0].pop(0)
            self.detected_corners[idx][1].pop(0)
            self.detected_corners[idx][2].pop(0)
            self.detected_corners[idx][3].pop(0)

        # TODO: Pre-allocate space for numpy array to avoid re-creating it each loop
        # TODO: improve this implementation
        filtered_corners = []
        c0 = removeOutliers(np.array(self.detected_corners[idx][0], dtype=np.float32))
        c1 = removeOutliers(np.array(self.detected_corners[idx][1], dtype=np.float32))
        c2 = removeOutliers(np.array(self.detected_corners[idx][2], dtype=np.float32))
        c3 = removeOutliers(np.array(self.detected_corners[idx][3], dtype=np.float32))

        if len(c0)==0 or len(c1)==0 or len(c2)==0 or len(c3)==0:
            return None, None # We can't do anything if we're missing a corner

        filtered_corners.append(np.mean(c0,axis=0))
        filtered_corners.append(np.mean(c1,axis=0))
        filtered_corners.append(np.mean(c2,axis=0))
        filtered_corners.append(np.mean(c3,axis=0))

        filtered_normals = removeOutliers(np.array(self.normals[idx], dtype=np.float32))
        filtered_normal = np.mean(filtered_normals,axis=0)

        return filtered_corners, filtered_normal
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
