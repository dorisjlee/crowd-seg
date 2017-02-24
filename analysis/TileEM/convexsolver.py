from __future__ import division
import cvxpy as cvx
from annotator import *
import numpy as np
import data

# Returns tuple (L, gammas) where L is the MLE
def solve(annotators, T):
	N = len(annotators)
	regions = annotators[0].regions
	gammas = cvx.Variable(regions.size)
	true_area = gammas.T * regions
	constraints = [gammas >= 0, gammas <= 1, true_area == T]
	f = 0
	for i in range(N):
		indicators = annotators[i].indicators
		term = 2 * cvx.log(gammas.T * np.multiply(regions, indicators)) \
				- cvx.log(np.dot(regions, indicators)) - cvx.log(T)
		f = f + term 

	objective = cvx.Maximize(f)
	prob = cvx.Problem(objective, constraints)
	
	return np.exp(prob.solve(solver='SCS')), gammas.value


