import random
from annotator import *
import numpy as np
from generateBoundingBoxData import *

""" 
Generate artificial dataset, with each region having size >= R_low 
and <= R_high

Input:
N: number of annotators
M: number of regions
I_low: smallest number of 1s in the indicator vector
I_high: largest number of 1s in the indicator vector
R_low: smallest region size
R_high: largest region size

Output:
(N + 1) X M matrix, with first N rows indicator vectors for each annotator and
 the last row being region sizes
"""
def gen_data(N, M, I_low, I_high,  R_low, R_high):
    res = []
    indices = [i for i in range(M)]
    for i in range(N):
        I = random.randint(I_low, I_high)
        row = [0 for i in range(M)]
        random.shuffle(indices)
        for j in range(I):
            row[indices[j] - 1] = 1
        res.append(row)
	
    res.append([random.randint(R_low, R_high) for i in range(M)])

    return res

"""
converts the data matrix to Annotator list
"""
def gen_data2(A):
    N = len(A) - 1
    M = len(A[0])
    res = []
    regions = np.array(A[N])
    for i in range(N):
        indicators = np.array(A[i])
        res.append(Annotator(indicators, regions))

    return res

"""
Fills in the (N + 1) X M raw matrix, using already generated test data
"""
def generateRawData(testSet):
    rawDataArray = [[0 for _ in xrange(testSet.M)] for _ in xrange(testSet.N + 1)]
    for i,region in enumerate(testSet.SelectedRegions):
        annotators = regionToAnnotators(region)
        for annotator in annotators:
            rawDataArray[annotator - 1][i] = 1
        rawDataArray[-1][i] = testSet.RegionToAreaMapping[region]
    return rawDataArray