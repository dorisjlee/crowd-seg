from random import randint, shuffle
from math import log

# Generate data for M regions (Area of region)
# Each region belongs to k annotators of a total of N

# Areas are normalized so that the Intersection of all Boxes has an area of IntersectionArea units
# Other areas will be sampled from a uniform distribution over integers
# from 1 to MaxAreaOfOtherRegions
class SimulateSingleObject:
    def __init__(self, N, M, IntersectionArea, MaxAreaOfOtherRegions):
        self.N = N
        self.M = M
        self.MaximumNumberOfRegions = 2**N
        self.IntersectionArea = IntersectionArea
        self.MaxAreaOfOtherRegions = MaxAreaOfOtherRegions
        # Dictionary that maps region number to its area
        # Admissible region numbers for the 5 annotators case
        # are 1 to 31
        self.RegionToAreaMapping = {}
        self.SelectedRegions = []
        self.setOfRegions = set()

    def createNewMapping(self):
        self.RegionToAreaMapping = {}
        # Create a list of regions and randomly drop a DROP_FRACTION of regions
        Regions = range(1,self.MaximumNumberOfRegions)
        shuffle(Regions)
        self.SelectedRegions = sorted(Regions[:self.M])
        self.setOfRegions = set(self.SelectedRegions)

        # Loop over selected regions
        # For 5 total annotators valid region codes go from
        # 00001 to 11111 (00000 is outside the bounding boxes)
        for i in self.SelectedRegions:
            self.RegionToAreaMapping[i] = randint(1,self.MaxAreaOfOtherRegions)

        # If the intersection region has not been dropped, make its area = INTERSECTION_AREA
        if self.SelectedRegions[-1] == self.MaximumNumberOfRegions - 1:
            self.RegionToAreaMapping[self.MaximumNumberOfRegions - 1] = self.IntersectionArea

    def hasIntersectionRegion(self):
        if self.MaximumNumberOfRegions - 1 in self.setOfRegions:
            return True
        return False

# Given a set of annotators, what region did only those annotators propose?
# Eg. If there are 5 total annotators, and (1,4,5) is the given set,
# then the region would have a binary code of 11001 = 25
def annotatorsToRegion(annotatorSet):
    region = 0
    for annotator in annotatorSet:
        region += 2**(annotator - 1)
    return region

# Given a region, find the annotators that proposed that region
# Eg. If there are 5 total annotators, and 25 is the given region,
# it has a binary code of 11001 and the annotators would be (1,4,5)
def regionToAnnotators(region):
    annotators = []
    annotatorPointer = int(log(region,2)) + 1
    while (region > 0):
        subtractor = 2**annotatorPointer
        if (region >= subtractor):
            region -= subtractor
            annotators.append(annotatorPointer + 1)
        annotatorPointer -= 1
    annotators.reverse()
    return annotators

#print regionToAnnotators(25)