import os 
import csv
import sqlite3
from glob import glob
from os.path import expanduser
import pandas as pd 
from PIL import Image
import numpy as np
def save_db_as_csv(db="crowd-segment",connect=True,postgres=True):
	'''
	Create CSV file of each table from app.db
	db = "segment" (local) ,"crowd-segment" (heroku remote)
	'''
	path = "/Users/dorislee/Desktop/Fall2016/Research/seg/data/"
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

def load_info():
	path = "/Users/dorislee/Desktop/Fall2016/Research/seg/data/"
	old_path = os.getcwd()
	os.chdir(path)
	img_info = pd.read_csv("image.csv",skipfooter=1)
	object_info = pd.read_csv("object.csv",skipfooter=1)
	object_location = pd.read_csv("object_location.csv",skipfooter=1)
	object_tbl = object_info.merge(object_location,how="inner",left_on="id",right_on="object_id")
	bb_info = pd.read_csv("bounding_box.csv",skipfooter=1)
	hit_info = pd.read_csv("hit.csv",skipfooter=1)
	os.chdir(old_path)
	return [img_info,object_tbl,bb_info,hit_info]

def get_size(fname):
	#Open image for computing width and height of image 
	im = Image.open(fname)
	width = im.size[0]
	height = im.size[1]
	return width, height
def visualize_my_ground_truth_bb():
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

def basic_stat(data1,name):
	print "{0}: mu = {1}; std = {2}".format(name,np.around(mean(data1),3),np.around(std(data1),3))
	return [name,np.around(mean(data1),3),np.around(std(data1),3)]

def basic_stats(data1,data2,mode="double"):
	print "Precision: mu = {0}; std = {1}".format(np.around(mean(data1),3),np.around(std(data1),3))
	if mode=="double": print "Recall: mu = {0}; std = {1}".format(np.around(mean(data2),3),np.around(std(data2),3))

def kolmogorov_smirnov(data1,data2,name):
	'''
	Two-way KS test . See if data come from the same distribution
	'''
	result = stats.ks_2samp(data1,data2)
	if result[1]>0.05: print "{0} : D = {1} ; p ={2} ---> {3}".format(name,np.around(result[0],2),np.around(result[1],2),pcheck(result[1],"from same distribution"))
	return result
def plot_fitted_worker_histo(fcn):
    metrics_lst = ['Precision [COCO]','Recall [COCO]','Jaccard [COCO]',\
                   'Precision [Self]','Recall [Self]','Jaccard [Self]']
    NUM_COL = 3
    NUM_ROW = 2
    NUM_PLOTS = NUM_COL*NUM_ROW

    fig, axs = plt.subplots(NUM_ROW,NUM_COL, figsize=(NUM_ROW*3,NUM_COL*1.5), sharex='col')
    stitle = fig.suptitle("Worker Metric Distribution [{}] ".format(fcn.name),fontsize=16,y=1.05)

    axs = axs.ravel()
    table_data = []
    for i,metric in zip(range(len(metrics_lst)),metrics_lst):
        metric_value = np.array(bb_info[metric][bb_info[metric]>FILTER_CRITERION][bb_info[metric]<=1]) 
        ax = axs[i]
        ax.set_title(metric)
        #ax.hist(metric_value,bins=30)
        ax.set_xlim(FILTER_CRITERION,1.03)


        metric_value = np.array(bb_info[metric][bb_info[metric]>0][bb_info[metric]<=1]) 
        params = fcn.fit(metric_value)
    #     histo,bin_edges = np.histogram(metric_value, 50, normed=1)
    #     bins = ((bin_edges+np.roll(bin_edges,-1))/2)[:-1]
        n, bins, patches = ax.hist(metric_value, 40, normed=1, facecolor='blue', alpha=0.75)
        y = fcn.pdf(bins, *params)
        l = ax.plot(bins, y, 'r--', linewidth=2) 
    fig.tight_layout()
    fig.savefig('{}_fitted_metric_histogram.pdf'.format(fcn.name), bbox_inches='tight',bbox_extra_artists=[stitle])