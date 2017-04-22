import os 
import csv
import sqlite3
from glob import glob
from os.path import expanduser
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from qualityBaseline import *
import scipy
def save_db_as_csv(db="crowd-segment",connect=True,postgres=True):
	'''
	Create CSV file of each table from app.db
	db = "segment" (local) ,"crowd-segment" (heroku remote)
	'''
	path = "/Users/dorislee/Desktop/Research/seg/data/"
	table_names = ["bounding_box","image","object","object_location","worker","hit"]
	for table_name in table_names :
		if postgres:
			if db=="crowd-segment" and connect==True:
				# Connect onto the DB on Heroku 
				os.system("bash herokuDBupdate.sh")
			os.system("psql {2}  -F , --no-align  -c  'SELECT * FROM {0}' > {1}/{0}.csv".format(table_name,path,db))
		else:
			# sqlite
			conn = sqlite3.connect(glob(expanduser('../web-app/app.db'))[0])
			cursor = conn.cursor()
			cursor.execute("select * from {};".format(table_name))
			with open("{}.csv".format(table_name), "wb") as csv_file:
				csv_writer = csv.writer(csv_file)
				csv_writer.writerow([i[0] for i in cursor.description]) # write headers
				csv_writer.writerows(cursor)
def COCO_convert_png_to_jpg():
	#Convert .jpg to .png
	os.chdir("app/static")
	for fname in glob.glob("COCO*"):
		os.system("convert {0} {1}".format(fname, fname.split(".")[0]+".png"))
from config import path
def load_info(eliminate_self_intersection_bb=True):
    from shapely.validation import explain_validity
    old_path = os.getcwd()
    os.chdir(path)
    img_info = pd.read_csv("image.csv",skipfooter=1)
    object_info = pd.read_csv("object.csv",skipfooter=1)
    object_location = pd.read_csv("object_location.csv",skipfooter=1)
    object_tbl = object_info.merge(object_location,how="inner",left_on="id",right_on="object_id")
    bb_info = pd.read_csv("bounding_box.csv",skipfooter=1)
    if eliminate_self_intersection_bb:
        for bb in bb_info.iterrows():
            bb=bb[1]
            xloc,yloc =  process_raw_locs([bb["x_locs"],bb["y_locs"]]) 
            worker_BB_polygon=Polygon(zip(xloc,yloc))
            if explain_validity(worker_BB_polygon).split("[")[0]=='Self-intersection':
                bb_info.drop(bb.name, inplace=True)
    hit_info = pd.read_csv("hit.csv",skipfooter=1)
    os.chdir(old_path)
    return [img_info,object_tbl,bb_info,hit_info]

import matplotlib.image as mpimg
def visualize_bb_objects(object_id,img_bkgrnd=True,worker_id=-1,gtypes=['worker','self'],single=False,bb_info=""):
    '''
    Plot BB for the object corresponding to the given object_id
    #Still need to implement COCO later...
    gtypes: list specifying the types of BB to be plotted (worker=all worker's annotation, 'self'=self BBG)
    '''
    if not single:
        img_info,object_tbl,bb_info,hit_info=load_info()
    else:
        img_info,object_tbl,bb_info_bad,hit_info=load_info()
    plt.figure(figsize =(7,7))
    ground_truth = pd.read_csv("../../data/object_ground_truth.csv")
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    if img_bkgrnd:
        img_name = img_info[img_info.id==int(object_tbl[object_tbl.id==object_id]["image_id"])]["filename"].iloc[0]
        fname = "../web-app/app/static/"+img_name+".png"
        img=mpimg.imread(fname)
        width,height = get_size(fname)
        img_id = int(img_name.split('_')[-1])
        plt.imshow(img)
        plt.xlim(0,width)
        plt.ylim(height,0)
        plt.axis("off")   
    else:
        plt.gca().invert_yaxis()
    plt.title("Object {0} [{1}]".format(object_id,object_tbl[object_tbl.object_id==object_id]["name"].iloc[0]))
#         plt.fill_between(x_locs,y_locs,color='none',facecolor='#f442df', alpha=0.5)
    if 'worker' in gtypes:
        bb_objects = bb_info[bb_info["object_id"]==object_id]
        if worker_id!=-1:
            bb = bb_objects[bb_objects["worker_id"]==worker_id]
            xloc,yloc =  process_raw_locs([bb["x_locs"].iloc[0],bb["y_locs"].iloc[0]])    
        
            plt.plot(xloc,yloc,'-',color='cyan',linewidth=3)
            plt.fill_between(xloc,yloc,color='none',facecolor='#f442df', alpha=0.01)
        else:
            for x,y in zip(bb_objects["x_locs"],bb_objects["y_locs"]):
                xloc,yloc = process_raw_locs([x,y])
                if single:
                    plt.plot(xloc,yloc,'-',color='#f442df',linewidth=4)
                else:
                    plt.plot(xloc,yloc,'-',color='#f442df',linewidth=1)
                    plt.fill_between(xloc,yloc,color='none',facecolor='#f442df', alpha=0.01)
    if 'self' in gtypes:
        ground_truth_match = my_BBG[my_BBG.object_id==object_id]
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
        if single:
            plt.plot(x_locs,y_locs,'--',color='#0000ff',linewidth=2)
        else: 
            plt.plot(x_locs,y_locs,'-',color='#0000ff',linewidth=4)
    # elif gtype=='COCO':
    #     ground_truth_match = my_BBG[my_BBG.object_id==object_id]
    if not single:plt.savefig("bb_object_{}.pdf".format(object_id))
def visualize_bb_worker(worker_id,gtypes=['worker','self']):
    '''
    Plot BB for the object corresponding to the given object_id
    #Still need to implement COCO later...
    gtypes: list specifying the types of BB to be plotted (worker=all worker's annotation, 'self'=self BBG)
    '''
    img_info,object_tbl,bb_info,hit_info=load_info()
    ground_truth = pd.read_csv("../../data/object_ground_truth.csv")
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    filtered_bb_info=bb_info[bb_info["worker_id"]==worker_id]
    for object_id in list(filtered_bb_info.object_id):
        visualize_bb_objects(object_id,single=True,gtypes=gtypes,bb_info=filtered_bb_info)
    plt.savefig("bb_worker_{0}_object_{1}.pdf".format(worker_id,object_id))
def visualize_all_ground_truth_bb():
    '''
	Plot all Ground truth bounding box drawn by me
	'''
    ground_truth = pd.read_csv("../../data/object_ground_truth.csv")
    worker_info = pd.read_csv("../../data/worker.csv",skipfooter=1)
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    for i in np.arange(len(img_info)):
        img_name = img_info["filename"][i]
        if 'COCO' in img_name:
            fname = "../web-app/app/static/"+img_name+".png"
            img=mpimg.imread(fname)
            width,height = get_size(fname)
            img_id = int(img_name.split('_')[-1])
            plt.figure(figsize =(10,10))
            plt.imshow(img)
            plt.axis("off")

            filtered_object_tbl = object_tbl[object_tbl["image_id"]==i+1]

            #for oid,bbx_path,bby_path in zip(bb_info["object_id"],bb_info["x_locs"],bb_info["y_locs"]):
            for bb in bb_info.iterrows():
                oid = bb[1]["object_id"]
                bbx_path= bb[1]["x_locs"]
                bby_path= bb[1]["y_locs"]
                if int(object_tbl[object_tbl.object_id==oid].image_id) ==i+1:
    #                 worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
                    ground_truth_match = my_BBG[my_BBG.object_id==oid]
                    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
                    plt.plot(x_locs,y_locs,'-',color='#f442df',linewidth=0.5)
                    plt.fill_between(x_locs,y_locs,color='none',facecolor='#f442df', alpha=0.01)
############################################################
############################################################
############		DATA FITTING MODULES			########
############################################################
############################################################
from scipy import stats
def pcheck(p,null_hyp):
    '''
    if p>0.05 then reject null hypothesis
    '''
    if p>0.05:
        return  null_hyp
    else:
        return "NOT "+null_hyp
def one_way_kolmogorov_smirnov(data,name,distr_name):
	'''
	See if data come from the reference distribution
	'''
	result = stats.kstest(data,distr_name)
	print "{0} : D = {1} ; p ={2} ---> {3}".format(name,np.around(result[0],2),np.around(result[1],2),pcheck(result[1],"from {} distribution".format(distr_name)))

def basic_stat(data1,name,PRINT=False):
	if PRINT: print "{0}: mu = {1}; std = {2}".format(name,np.around(np.mean(data1),3),np.around(np.std(data1),3))
	return [name,np.around(np.mean(data1),3),np.around(np.std(data1),3)]

def basic_stats(data1,data2,mode="double"):
	print "Precision: mu = {0}; std = {1}".format(np.around(np.mean(data1),3),np.around(np.std(data1),3))
	if mode=="double": print "Recall: mu = {0}; std = {1}".format(np.around(np.mean(data2),3),np.around(np.std(data2),3))

def kolmogorov_smirnov(data1,data2,name,PRINT=False):
	'''
	Two-way KS test . See if data come from the same distribution
	'''
	result = stats.ks_2samp(data1,data2)
	if result[1]>0.05 and PRINT: print "{0} : D = {1} ; p ={2} ---> {3}".format(name,np.around(result[0],2),np.around(result[1],2),pcheck(result[1],"from same distribution"))
	return result
############################################################
############   OVERALL DISTRIBUTION ANALYSIS        ########
############################################################

def plot_fitted_worker_histo(fcn,FILTER_CRITERION=0):
    bb_info = pd.read_csv("computed_my_COCO_BBvals.csv")
    metrics_lst = ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',"NME [COCO]","Num Points",\
               'Precision [Self]','Recall [Self]','Jaccard [Self]',"NME [Self]","Area Ratio"]
    NUM_PLOTS = len(metrics_lst)
    NUM_ROW = 2
    NUM_COL = NUM_PLOTS/NUM_ROW

    fig, axs = plt.subplots(NUM_ROW,NUM_COL, figsize=(NUM_COL*2.5,NUM_ROW*3))#, sharex='col')
    stitle = fig.suptitle("Worker Metric Distribution [{}] ".format(fcn.name),fontsize=16,y=1.05)

    axs = axs.ravel()
    table_data = []
    for i,metric in zip(range(len(metrics_lst)),metrics_lst):
        metric_value = np.array(bb_info[metric][bb_info[metric]>FILTER_CRITERION][bb_info[metric]<=1]) 
        ax = axs[i]
        ax.set_title(metric)
        #ax.hist(metric_value,bins=30)
        if metric in ["Num Points"]:
            metric_value = np.array(bb_info[metric])
            ax.set_xlim(0,metric_value.max())
        else:
            #restrict range [0,1] for normalized measures
            ax.set_xlim(0,1.03)
            metric_value = np.array(bb_info[metric][bb_info[metric]>0][bb_info[metric]<=1]) 
        params = fcn.fit(metric_value)
        n, bins, patches = ax.hist(metric_value, 50, normed=1, facecolor='blue', alpha=0.75)
        y = fcn.pdf(bins, *params)
        l = ax.plot(bins, y, 'r--', linewidth=2) 
    fig.tight_layout()
    fig.savefig('{}_fitted_metric_histogram.pdf'.format(fcn.name), bbox_inches='tight',bbox_extra_artists=[stitle])
def compute_all_stats(FILTER_CRITERION=0.):
    '''
    Compute the basic stats of all metrics and store it in a table format (table_data)
    '''    
    bb_info = pd.read_csv("computed_my_COCO_BBvals.csv")
    table_data = []
    metrics_lst = ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',"NME [COCO]","Num Points",\
               'Precision [Self]','Recall [Self]','Jaccard [Self]',"NME [Self]","Area Ratio"]    
    for i,metric in zip(range(len(metrics_lst)),metrics_lst):
        if metric in ["Num Points","Area Ratio"]:
            metric_value = np.array(bb_info[metric])
        else:
            #restrict range [0,1] for normalized measures
            metric_value = np.array(bb_info[metric][bb_info[metric]>FILTER_CRITERION][bb_info[metric]<=1]) 
        table_data.append(basic_stat(metric_value,metric,PRINT=False))
    if FILTER_CRITERION==0:
        print tabulate(table_data,headers=["All","Mean","SD"],showindex="False",tablefmt='latex',floatfmt='.2g')
    else:
        print tabulate(table_data,headers=["Filter>{}".format(FILTER_CRITERION),"Mean","SD"],showindex="False",tablefmt='latex',floatfmt='.2g')
############################################################
############       J_I DISTRIBUTION ANALYSIS        ########
############################################################

import matplotlib.ticker as ticker
def plot_all_Ji_hist(fcn,SHOW_PLOT=10,NBINS=30):
    '''
    Plot all worker distributions for each object 
    compute fitting coefficients for each Ji distribution
    show SHOW_PLOT number of sample plots
    '''
    bb_info = pd.read_csv('computed_my_COCO_BBvals.csv')
    obj_sorted_tbl =  bb_info[bb_info['Jaccard [COCO]']!=-1][bb_info['Jaccard [COCO]']!=0][bb_info['Jaccard [Self]']!=0].sort('object_id')
    object_id_lst  = list(set(obj_sorted_tbl.object_id))
    metrics_lst = ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',"NME [COCO]","Num Points",\
               'Precision [Self]','Recall [Self]','Jaccard [Self]',"NME [Self]","Area Ratio"]
    NUM_PLOTS = len(metrics_lst)
    NUM_ROW = 2
    NUM_COL = NUM_PLOTS/NUM_ROW
    data_fit_stats=[]
    for objid in object_id_lst:
        if SHOW_PLOT>0:
            fig, axs = plt.subplots(NUM_ROW,NUM_COL, figsize=(NUM_COL*2.5,NUM_ROW*3))
            stitle = fig.suptitle("J{} Distribution ".format(objid),fontsize=16,y=1.05)
            axs = axs.ravel()

        # Ji_tbl (bb_info) is the set of all workers that annotated object i 
        bb  = obj_sorted_tbl[obj_sorted_tbl["object_id"]==objid]
        for i,metric in zip(range(len(metrics_lst)),metrics_lst):
            if metric in ["Num Points"]:
                metric_value = np.array(bb[metric])
            else:
                #restrict range [0,1] for normalized measures
                metric_value = np.array(bb[metric][bb[metric]>0][bb[metric]<=1]) 
            params = fcn.fit(metric_value)
            histo,bin_edges = np.histogram(metric_value, NBINS, normed=1)
            bins = ((bin_edges+np.roll(bin_edges,-1))/2)[:-1]
            y = fcn.pdf(bins, *params)
            RSS =sum((histo-y)**2)
            ks_result = kolmogorov_smirnov(bins,y,fcn.name) #D-value and p-value
            # object_id, Metric, mu, sd,RSS,D-value,p-value
            data_stats  = [objid,metric,params[0],params[1],RSS,ks_result[0],ks_result[1]] 
            #same as what you would get if you did basic_stats because in the MLE estimate for Gaussians, mu and sigma is equal to sample mean and sample sd
            data_fit_stats.append(data_stats)

            if SHOW_PLOT>0:
                ax = axs[i]
                ax.set_title(metric)
                start = metric_value.min()
                end = metric_value.max()
                logdx = np.log10(abs(end-start))
                if logdx<=-2:
                    ax.set_xticks(np.linspace(metric_value.min(),metric_value.max(),3))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
                elif logdx<=-1.5:
                    ax.set_xticks(np.linspace(metric_value.min(),metric_value.max(),4))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
                elif logdx<=-1:
                    ax.set_xticks(np.linspace(metric_value.min(),metric_value.max(),4))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
                elif logdx<=-0.5:
                    ax.set_xticks(np.linspace(metric_value.min(),metric_value.max(),4))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
                elif logdx<=0:
                    ax.set_xticks(np.linspace(metric_value.min(),metric_value.max(),5))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                elif logdx<=1:
                    ax.set_xticks(np.linspace(metric_value.min(),metric_value.max(),6))
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
                n, bins, patches = ax.hist(metric_value, NBINS, normed=1,facecolor='blue', alpha=0.75)
                y = fcn.pdf(bins, *params)
                l = ax.plot(bins, y, 'r--', linewidth=2) 

        if SHOW_PLOT>0: fig.tight_layout()
        SHOW_PLOT-=1
    fit_results =pd.DataFrame(data_fit_stats,columns=["object_id", "Metric", "Mean", "SD","RSS","D-value","p-value"])
    return fit_results
def compute_all_fittings():
    '''
    Compute all fitting coefficients 
    '''
    bb_info = pd.read_csv('computed_my_COCO_BBvals.csv')
    metrics_lst = ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',"NME [COCO]","Num Points",\
               'Precision [Self]','Recall [Self]','Jaccard [Self]',"NME [Self]","Area Ratio"]
    exclude= ['division', 'skellam', 'nbinom', 'logser', 'erlang','dlaplace', 'hypergeom', 'bernoulli', 'levy_stable', 'zipf', 'rv_discrete', 'rv_frozen', 'boltzmann', 'rv_continuous', 'entropy', 'randint', 'poisson', 'geom', 'binom', 'planck', 'print_function']
    data_fit_stats=[]
    for i,metric in tqdm(zip(range(len(metrics_lst)),metrics_lst)):
        if metric in ["Num Points"]:
            metric_value = np.array(bb_info[metric])
        else:
            #restrict range [0,1] for normalized measures
            metric_value = np.array(bb_info[metric][bb_info[metric]>0][bb_info[metric]<=1]) 
        #Testing against various distributions 
        for fcn_name in filter(lambda x: x not in exclude,dir(stats.distributions)[9:]):
            # Based on MLE estimate for fitting
            try:
                fcn = getattr(scipy.stats,fcn_name)
                params = fcn.fit(metric_value)
                histo,bin_edges = np.histogram(metric_value, 50, normed=1)
                bins = ((bin_edges+np.roll(bin_edges,-1))/2)[:-1]
                y = fcn.pdf(bins, *params)
                RSS =sum((histo-y)**2)
                ks_result = kolmogorov_smirnov(bins,y,fcn_name) #D-value and p-value
                data_stats  = [metric,fcn_name,params,RSS]
                data_stats.extend(ks_result)
                data_fit_stats.append(data_stats)
            except(AttributeError,NotImplementedError,TypeError):
                #function has no fitting
                print "Skipped", fcn_name
    df_stats_tbl = pd.DataFrame(data_fit_stats,columns=["metric","Function Name", "Parameters","RSS","D-value","p-value"])
    df_stats_tbl.to_csv("overall_fit_results.csv")
    return df_stats_tbl

def test_all_Ji_fit_fcn(fcns_to_test="all",NBINS=30,RAND_SAMPLING=5):
    '''
    Test all function form against all Ji distributions, then return the fitting coefficients table
    RAND_SAMPLING controls the number of objects that gets tested, running this on all 23 object will cause memory crash
    '''
    if fcns_to_test=="all":
        exclude= ['division', 'skellam', 'nbinom', 'logser', 'erlang','dlaplace', 'hypergeom', 'bernoulli', 'levy_stable', 'zipf', 'rv_discrete', 'rv_frozen', 'boltzmann', 'rv_continuous', 'entropy', 'randint', 'poisson', 'geom', 'binom', 'planck', 'print_function']
        fcns_to_test = filter(lambda x: x not in exclude,dir(stats.distributions)[9:])
    bb_info = pd.read_csv('computed_my_COCO_BBvals.csv')
    obj_sorted_tbl =  bb_info[bb_info['Jaccard [COCO]']!=-1][bb_info['Jaccard [COCO]']!=0][bb_info['Jaccard [Self]']!=0].sort('object_id')
    object_id_lst  = list(set(obj_sorted_tbl.object_id))
    metrics_lst = ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',"NME [COCO]","Num Points",\
               'Precision [Self]','Recall [Self]','Jaccard [Self]',"NME [Self]","Area Ratio"]
    if RAND_SAMPLING : object_id_lst = np.random.choice(object_id_lst,RAND_SAMPLING)
    NUM_PLOTS = len(metrics_lst)
    NUM_ROW = 2
    NUM_COL = NUM_PLOTS/NUM_ROW
    data_fit_stats=[]
    for objid in tqdm(object_id_lst):
        # Ji_tbl (bb_info) is the set of all workers that annotated object i 
        bb  = obj_sorted_tbl[obj_sorted_tbl["object_id"]==objid]
        for i,metric in zip(range(len(metrics_lst)),metrics_lst):
            if metric in ["Num Points"]:
                metric_value = np.array(bb[metric])
                pltmax=metric_value.max()
            else:
                #restrict range [0,1] for normalized measures
                metric_value = np.array(bb[metric][bb[metric]>0][bb[metric]<=1]) 
                pltmax=1

            #Testing against various distributions 
            for fcn_name in fcns_to_test:
                fcn = getattr(stats,fcn_name)
                params = fcn.fit(metric_value)
                histo,bin_edges = np.histogram(metric_value, NBINS, normed=1)
                bins = ((bin_edges+np.roll(bin_edges,-1))/2)[:-1]
                y = fcn.pdf(bins, *params)
                RSS =sum((histo-y)**2)
                ks_result = kolmogorov_smirnov(bins,y,fcn.name) #D-value and p-value
                # object_id, Metric, mu, sd,RSS,D-value,p-value
                data_stats  = [objid,fcn_name,metric,params[0],params[1],RSS,ks_result[0],ks_result[1]] 
                #same as what you would get if you did basic_stats because in the MLE estimate for Gaussians, mu and sigma is equal to sample mean and sample sd
                data_fit_stats.append(data_stats)
    fit_results =pd.DataFrame(data_fit_stats,columns=["object_id","Function", "metric", "Mean", "SD","RSS","D-value","p-value"])
    #Drop Unnamed columns (index from rewriting same file)
    fit_results = fit_results[fit_results.columns[~fit_results.columns.str.contains('Unnamed:')]]
    fit_results.to_csv("Ji_fit_results.csv")
    return fit_results
def RSSBoxplot(fit_results,fcn_name):
    data_lst = []
    for metric in metrics_lst:
        data = np.array(fit_results[fit_results["Metric"]==metric].RSS)
        data_lst.append(data)
    fig,ax = plt.subplots()
    ax.set_yscale('log')
    plt.boxplot(data_lst)
    plt.ylim(0,1e6)
    plt.ylabel("{} fit RSS".format(fcn_name),fontsize=12)
    p = ax.set_xticklabels([metrics_lst[i] for i in range(len(metrics_lst))], rotation=25,ha='right',fontsize=12)
    plt.tight_layout()
    plt.savefig("{}RSSBoxplot.pdf".format(fcn_name))    
import warnings
warnings.filterwarnings("ignore")
def fit_against_all_dist(data,binsize=10):
    '''
    Fitting given data array against all statistical distributions in scipy.stats
    '''
    exclude= ['division', 'skellam', 'nbinom', 'logser', 'erlang','dlaplace', 'hypergeom', 'bernoulli', 'levy_stable', 'zipf', 'rv_discrete', 'rv_frozen', 'boltzmann', 'rv_continuous', 'entropy', 'randint', 'poisson', 'geom', 'binom', 'planck', 'print_function']
    fcn_lst = filter(lambda x: x not in exclude,dir(stats.distributions)[9:])
    data_fit_stats=[]
    #Testing against various distributions 
    for fcn_name in tqdm(fcn_lst):
        # Based on MLE estimate for fitting
        try:
            fcn = getattr(scipy.stats,fcn_name)
            params = fcn.fit(data)
            histo,bin_edges = np.histogram(data, binsize, normed=1)
            bins = ((bin_edges+np.roll(bin_edges,-1))/2)[:-1]
            y = fcn.pdf(bins, *params)
            RSS =sum((histo-y)**2)
            #ks_result = kolmogorov_smirnov(bins,y,fcn_name) #D-value and p-value
            data_stats  = [fcn_name,params,RSS]
            #data_stats.extend(ks_result)
            data_fit_stats.append(data_stats)
        except(AttributeError,NotImplementedError,TypeError):
            #function has no fitting
            print "Skipped", fcn_name
    df_stats_tbl = pd.DataFrame(data_fit_stats,columns=["Function Name", "Parameters","RSS"])
    return df_stats_tbl
    
