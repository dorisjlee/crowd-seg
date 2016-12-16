import os 
import csv
import sqlite3
from glob import glob
from os.path import expanduser
import pandas as pd 
from PIL import Image
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