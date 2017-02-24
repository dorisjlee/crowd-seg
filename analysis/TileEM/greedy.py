__author__ = 'kgoel93'

from generateBoundingBoxData import *

class Solution:

    def __init__(self,n):
        self.areaOfIntersections = [1e-10]*n
        self.areaOfSolution = 0
        self.logLikelihood = -100000000
        self.frontier = set()
        self.solution = set()

    def printSolution(self):
        print "Solution list is " + str(list(self.solution)) + "\nFrontier is " + \
                str(self.frontier) + "\nLog-likelihood is " + str(self.logLikelihood) \
                    + "\nArea is " + str(self.areaOfSolution)

# Do Greedy Search on the set of Regions using the poset ordering
# Maintain the set of candidate nodes (frontier) that are available to test likelihood increases on
# Pick the one that increases the likelihood the most, and add it to the set of solution nodes
# Update the frontier
# Terminate when no node in the frontier improves the likelihood
def greedySearch(testSet):
    #Initialize a solution
    solution = Solution(testSet.N)
    initializeFrontier(testSet,solution)
    while True:
        maxPick = greedyPick(solution,testSet)
        if (maxPick[0] > solution.logLikelihood):
            updateSolution(testSet,solution,maxPick)
        else:
            break
    boundingBoxTerm = getBoundingBoxTerm(testSet)
    solution.logLikelihood = solution.logLikelihood - boundingBoxTerm
    return solution

def initializeFrontier(testSet,solution):
    if testSet.hasIntersectionRegion():
        solution.frontier = {testSet.MaximumNumberOfRegions - 1}
    else:
        #Checking what the highest available level in the poset is and its corresponding frontier
        checkList = {testSet.MaximumNumberOfRegions - 1}
        for _ in xrange(1,testSet.N):
            allChildren = set()
            for region in checkList:
                allChildren = allChildren.union(set(getChildrenRegionsSlow(region,set(range(1,testSet.MaximumNumberOfRegions)))))
            children = testSet.setOfRegions.intersection(allChildren)
            if (len(children) > 0):
                solution.frontier = children
                break
            else:
                checkList = allChildren

#Given a frontier, pick the region that gives the most increase in log-likelihood
def greedyPick(solution,testSet):
    # Max greedy pick with (maxLikelihood, corresponding region)
    maxPick = (-1000000000,-1)
    for region in solution.frontier:
        logLikelihood = computeLogLikelihood(region,solution,testSet)
        if (logLikelihood > maxPick[0]):
            maxPick = (logLikelihood,region)
    return maxPick

def updateSolution(testSet,solution,maxPick):
    logLikelihood,region = maxPick
    solution.solution = set(list(solution.solution)+ [region])
    solution.logLikelihood = logLikelihood
    solution.areaOfSolution += testSet.RegionToAreaMapping[region]
    annotators = set(regionToAnnotators(region))
    for i in xrange(1,testSet.N + 1):
        if i in annotators:
            solution.areaOfIntersections[i - 1] += testSet.RegionToAreaMapping[region]
    updateFrontier(testSet,solution,region)


def updateFrontier(testSet,solution,region):
    initialFrontier = list(solution.frontier)
    candidatesToAdd = getChildrenRegionsSlow(region,testSet.setOfRegions)
    for candidate in candidatesToAdd:
        #Check if all its parents are in the solution we have until now
        parentsOfCandidate = set(getParentRegionsSlow(candidate,testSet.setOfRegions,testSet.N))
        if parentsOfCandidate.issubset(solution.solution):
            #We can add this to the frontier
            initialFrontier.append(candidate)
    finalFrontier = set(initialFrontier).difference({region})
    solution.frontier = finalFrontier


#For a region, get its children in the poset/dag
def getChildrenRegionsSlow(region,regions):
    annotators = regionToAnnotators(region)
    children = []
    #Remove each annotator once to generate each child
    for annotator in annotators:
        child = region - 2**(annotator - 1)
        if child in regions:
            children.append(child)
    return children

#For a region, get its parents in the poset/dag
def getParentRegionsSlow(region,regions, n):
    annotators = set(regionToAnnotators(region))
    parents = []
    for annotator in xrange(1,n+1):
        if annotator not in annotators:
            parent = region + 2**(annotator - 1)
            if parent in regions:
                parents.append(parent)
    return parents

#Compute the log-likelihood for a candidate solution
def computeLogLikelihood(region,solution,testSet):
    annotators = set(regionToAnnotators(region))
    logLikelihood = - testSet.N * log(solution.areaOfSolution + testSet.RegionToAreaMapping[region])
    for i in xrange(1,testSet.N + 1):
        if i in annotators:
            logLikelihood += 2 * log(solution.areaOfIntersections[i - 1] + testSet.RegionToAreaMapping[region])
        else:
            solution.printSolution()
            print solution.areaOfIntersections
            print solution.areaOfIntersections[i - 1]
            logLikelihood += 2 * log(solution.areaOfIntersections[i - 1])
    return logLikelihood

#Returns the value of the log-sum of bounding box areas across all annotators
def getBoundingBoxTerm(testSet):
    boundingBoxes = [0] * testSet.N
    for region in testSet.SelectedRegions:
        annotators = regionToAnnotators(region)
        for annotator in annotators:
            boundingBoxes[annotator - 1] += testSet.RegionToAreaMapping[region]
    boundingBoxTerm = 0
    for boundingBox in boundingBoxes:
        boundingBoxTerm += log(boundingBox)
    return boundingBoxTerm