import time
from sys import argv
from greedy import *
from data import *
from experiment import *
import numpy as np
import matplotlib.pyplot as plt


def main(argv):
    N = int(argv[0]) # Number of annotators
    M = int(argv[1]) # Number of regions 
    IntersectionArea = int(argv[2])
    MaxAreaOfOtherRegions = int(argv[3])
    print "**************************"
    TestSet = SimulateSingleObject(N,M,IntersectionArea,MaxAreaOfOtherRegions)
    TestSet.createNewMapping()
    tinit = time.time()
    print "Starting Greedy Algorithm"
    print "--------------------------"
    solution = greedySearch(TestSet)
    print "Greedy Algorithm Completed"
    print "--------------------------"
    solution.printSolution()
    print "---------------------------*---------------*--------------------------------"
    tend = time.time()
    Tgreedy = tend-tinit
    print "Time for Greedy Algorithm: ", Tgreedy
    raw_data = generateRawData(TestSet)
    tinit = time.time()
    T, L, solutionList = experiment_exhaustive(1, raw_data)
    print "Exhaustive Search Completed"
    print "--------------------------"
    print "Solution list is " + str(solutionList)
    print "Log-likelihood is " + str(np.log(max(L)))
    print "Area is " + str(T[L.index(max(L))])
    print "---------------------------*---------------*--------------------------------"
    tend = time.time()
    Texhaust =  tend-tinit
    print "Time for Exhaustive Search: ",Texhaust
    #plt.scatter(T, np.log(L), s = 1)
    #plt.show()
    tinit = time.time()
    T, L, solutionList = experiment_local(1, raw_data)
    print "Local Search Completed"
    print "--------------------------"
    print "Solution list is " + str(solutionList)
    print "Log-likelihood is " + str(np.log(L))
    print "Area is " + str(T)
    tend = time.time()
    Tlocal = tend-tinit
    print "Time for Local Search: ", Tlocal
    print "---------------------------*---------------*--------------------------------"
	
    T, L, solutionList = experiment_median(raw_data)
    print "Median Solution Completed"
    print "--------------------------"
    print "Solution list is " + str(solutionList)
    print "Log-likelihood is " + str(np.log(L))
    print "Area is " + str(T)
    print "---------------------------*---------------*--------------------------------"

    T, L, solutionList = experiment_avg(raw_data)
    print "Average Solution Completed"
    print "--------------------------"
    print "Solution list is " + str(solutionList)
    print "Log-likelihood is " + str(np.log(L))
    print "Area is " + str(T)
    print "---------------------------*---------------*--------------------------------"

    print "**************************"
    f = open("timing.txt",'a')
    f.write("{0},{1},{2},{3},{4}\n ".format(N,M,Tgreedy,Texhaust,Tlocal))
    f.close()
if __name__ == '__main__':
    main(argv[1:])
