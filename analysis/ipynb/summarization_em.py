import numpy
import numpy as np
import matplotlib.pyplot as plt
from analysis_toolbox import *
from qualityBaseline import *
import pandas as pd 
from scipy.optimize import minimize
bb_info = pd.read_csv('../../crowd-seg/analysis/computed_my_COCO_BBvals.csv')
obj_sorted_tbl =  bb_info[bb_info['Jaccard [COCO]']!=-1][bb_info['Jaccard [COCO]']!=0][bb_info['Jaccard [Self]']!=0].sort('object_id')
object_id_lst  = list(set(obj_sorted_tbl.object_id))
img_info,object_tbl,bb_info,hit_info = load_info()
def abslogN(param):
    '''
    Drawing from normal distribution with prior (std) as param, with mean centered at 0
    log without nan; nan if value <0
    '''
    return log(abs(random.normal(0,param)))
def compute_phii(metric):
    '''
    phi value compared to ground truth 
    '''
    if metric in ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',"NME [COCO]",\
       'Precision [Self]','Recall [Self]','Jaccard [Self]',"NME [Self]"]:
        return 1
    elif metric in ["Num Points","Area Ratio"]:
        return 0

# T = total number of tasks
# n = number of points in bounding box
# m = number of metric functions of interest
# W = total number of workers 
#metric functions of interest 
#metrics_lst = ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',"NME [COCO]","Num Points",\
#               'Precision [Self]','Recall [Self]','Jaccard [Self]',"NME [Self]","Area Ratio"]
metrics_lst = ['Precision [COCO]','Recall [COCO]']
worker_lst = unique(bb_info.worker_id)
#Parameters: Priors
beta_vec = np.ones((m,1))# m x 1
gamma_vec = np.ones((m,1))# m x 1
theta_vec = np.ones((m,1))# m x 1
# Hidden variables 
m = len(metrics_lst)
T = len(img_info)
W = len(worker_lst)
z = np.ones((50,T))# n x T
D = np.zeros((m,T))# m x T
b = np.zeros((m,W))# m x W
c = np.zeros((m,W))# m x W
# equally weigh Di and cj
w1=0.5
w2=0.5

def Q_k(params,k):
    '''
    Objective function to maximize
    params : parameter beta, gamma, theta
    k : index for the metric component to compute Q_k for
    $$ Q_k(\mathbf{\Theta}|\mathbf{\Theta}^\prime)= \sum_j \Bigg[ log p(b_j|\beta_k)+  log p(c_j|\gamma_k)\Bigg] + \sum_i \Bigg[\sum_{j\in J_i}  log p(D_i|\theta_k) + log \mathcal{N}(\phi_{ij};\mu_{ij},\sigma_{ij}) \Bigg]$$
    '''
    beta,gamma,theta = params
    loglikelihood=[]
    #Loop through all task
    task_prob_lst =[]
    for i in range(T):
        logphiij_lst = []
        # Ji is the list of workers that annotated object i
        Ji = list(obj_sorted_tbl[obj_sorted_tbl["object_id"]==i].worker_id)
        Di = abslogN(theta)
        # Updating Di
        D[k][i]=Di
        for j in Ji : 
            j = int(np.where(worker_lst==j)[0])
            phii=compute_phii(metrics_lst[k])
            mu = phii-b[k][j]
            cj = abslogN(gamma)
            sig = sqrt(w1*Di**2+w2*cj**2)
            if sig==0: sig=1e-5
            logphiij_lst.append(log(abs(random.normal(mu,sig))))
        logphiij = sum(logphiij_lst)
        task_prob_lst.append(Di+logphiij)
    task_prob = sum(task_prob_lst)

    #Loop through all workers
    worker_prob_lst=[]
    for j in range(W):
        bj = abslogN(beta)
        cj = abslogN(gamma)
        worker_prob_lst.append(bj+cj)
        # Updating cj,bj
        b[k][j] = bj
        c[k][j] = cj
    worker_prob= sum(worker_prob_lst)
    loglikelihood.append(worker_prob+task_prob)
    # Updating parameter vectors 
    beta_vec[k] = beta
    gamma_vec[k] = gamma
    theta_vec[k] = theta

    # sum of likelihood over all the Phi_k functions 
    return -sum(loglikelihood)
# constrain optimization: parameters can only be postitive (since they are variance measures)
bnds= ((0, None), (0, None),(0, None))
N_iter = 50 #numebr of iterations 
negloglikelihood_lst = []
# Loop through all Phi metrics 
for k in range(m):
    negloglikelihood_klst=[]
    for i in range(N_iter): 
        # E step 
        negloglikelihood = Q_k([beta,gamma,theta],k)
        negloglikelihood_klst.append(negloglikelihood)
        # M step
        results = minimize(Q_k,[100,100,100],args=(k),method='tnc',bounds=bnds)#, options={'factr' : 1e7})
        beta,gamma,theta = results.x
        beta_vec[k]=beta
        gamma_vec[k]=gamma
        theta_vec[k]=theta
        print '------------'
        print "{0} iter".format(i)
        print "likelihood: {0}".format(negloglikelihood)
        print "[b,g,t]: {0},{1},{2}".format(beta,gamma,theta)
    negloglikelihood_lst.append(negloglikelihood_klst)
for negloglikelihood_klst in negloglikelihood_lst:
    plt.figure()
    plt.plot(negloglikelihood_klst)