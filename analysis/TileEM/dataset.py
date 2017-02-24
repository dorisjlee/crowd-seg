from random import randint, shuffle
from math import log
import numpy as np
# Generate data for M regions (Area of region)
# Each region belongs to k annotators of a total of N

# Areas are normalized so that the Intersection of all Boxes has an area of IntersectionArea units

class Dataset:
    # Given a indicator matrix, create a TestSet
    def __init__(self, tiles,objIndicatorMat, IntersectionArea):

        Nworkers,Nregions = np.shape(objIndicatorMat)
        self.N = Nworkers-1
        self.M = Nregions
        self.MaximumNumberOfRegions = 2**self.N
        self.IntersectionArea = IntersectionArea
        # Dictionary that maps region number to its area
        # Admissible region numbers for the 5 annotators case
        # are 1 to 31
        regions = range(Nregions)
        self.RegionToAreaMapping = dict(zip(regions,objIndicatorMat[self.N]))
        self.SelectedRegions = sorted(regions)
        self.setOfRegions = set(self.SelectedRegions)
    def hasIntersectionRegion(self):
        if self.MaximumNumberOfRegions - 1 in self.setOfRegions:
            return True
        return False
