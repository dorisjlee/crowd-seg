import os 
import csv
import sqlite3
from glob import glob
from os.path import expanduser
import pandas as pd 
from PIL import Image
def save_db_as_csv(postgres=True):
    '''
    Create CSV file of each table from app.db
    '''
    path = "/Users/dorislee/Desktop/Fall2016/Research/seg/data/"
    table_names = ["bounding_box","image","object","object_location","worker","hit"]
    for table_name in table_names :
        if postgres:
            os.system("psql segment  -F , --no-align  -c  'SELECT * FROM {0}' > {1}/{0}.csv".format(table_name,path))
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
	img_info = pd.read_csv("image.csv")
	object_info = pd.read_csv("object.csv")
	object_location = pd.read_csv("object_location.csv")
	object_tbl = object_info.merge(object_location,how="inner",left_on="id",right_on="object_id")
	bb_info = pd.read_csv("bounding_box.csv")
	hit_info = pd.read_csv("hit.csv")
	return [img_info,object_tbl,bb_info,hit_info]

def get_size(fname):
    #Open image for computing width and height of image 
    im = Image.open(fname)
    width = im.size[0]
    height = im.size[1]
    return width, height