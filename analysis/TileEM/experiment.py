import convexsolver as cvx
import data
import numpy as np
from tqdm import tqdm
def experiment_exhaustive(T_low, raw_data,step_size=1):
    annotators = data.gen_data2(raw_data)
    T_high = int(sum(annotators[0].regions))
    T_values = []
    L_values = []
    print "Starting Exhaustive Search with T values between " + str(T_low) + " and " + str(T_high)
    print "--------------------------"
    maxL = -1000000000
    gammas = None
    for T in tqdm(range(T_low, T_high + 1,step_size)):
        l, g = cvx.solve(annotators, T)
        if l is None:
            print "Failed to solve"
        else:
            if (l > maxL):
                maxL = l
                gammas = g
            T_values.append(T)
            L_values.append(l)
        #print "Done with T = " + str(T)
    return T_values, L_values, gammas, getSolution(gammas)

def experiment_local(T_low, raw_data):
	annotators = data.gen_data2(raw_data)
	T_high = sum(annotators[0].regions)
	print "Starting Local Search with T values between " + str(T_low) + " and " + str(T_high)
	print "--------------------------"
	
	step_size = T_high / 10
	curr_T = (T_low + T_high) / 2	
	max_L = -1000000000
	gammas = None
	while True:
		# left neighbor, step_size away from current T value
		T_left = curr_T - step_size
		# right neighbor, step_size away from current T value
		T_right = curr_T + step_size

		next_T = curr_T
		if T_left >= T_low:
			l, g = cvx.solve(annotators, T_left)
			if l > max_L:
				max_L = l
				gammas = g
				next_T = T_left
		
		if T_right <= T_high:
			l, g = cvx.solve(annotators, T_right)
			if l > max_L:
				max_L = l
				gammas = g
				next_T = T_right
		
		# only decrease step size when no update was made
		if step_size >= 1 and next_T == curr_T:
			step_size /=  2
		# if no update was made when step_size reached 1, we have
		# found a local minimum
		if step_size <= 1 and next_T == curr_T:
			break
		curr_T = next_T
	return curr_T, max_L, gammas, getSolution(gammas)

def experiment_median(raw_data):
	annotators = data.gen_data2(raw_data)
	N = len(annotators)
	areas = [np.dot(annotators[i].indicators, annotators[i].regions) \
				for i in range(N)]
	sorted(areas)
	T = areas[int(N / 2)]
	print "Starting Median Experiment with T value " + str(T)
	print "--------------------------"
	l, g = cvx.solve(annotators, T)
	return T, l, g, getSolution(g)

def experiment_avg(raw_data):
	annotators = data.gen_data2(raw_data)
	N = len(annotators)
	T = sum([np.dot(annotators[i].indicators, annotators[i].regions) \
				for i in range(N)]) / N
	
	print "Starting Average Experiment with T value " + str(T)
	print "--------------------------"
	l, g = cvx.solve(annotators, T)
	print l,g
	return T, l, g, getSolution(g)

def getSolution(gammas):
    if gammas is None:
    	# In the case when the CVX solver can not find a solution, it returns gamma as None and l as inf. 
    	# This happens for Median or Average case, where your T value is just very off, so you can't really find a good ML region corresponding to the T constraints.
    	# This means that our solution set should be empty.
    	return []
    solutionList = []
    solutionListPartial = []
    for i,gamma in enumerate(gammas):
        if gamma >= 0.99:
            solutionList.append(i+1)
        elif gamma < 0.99 and gamma > 0.01:
            solutionListPartial.append(i+1)
    solutionList.extend(solutionListPartial)
    return solutionList
