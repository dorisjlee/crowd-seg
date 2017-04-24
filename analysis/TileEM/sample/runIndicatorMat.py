import sys
import json
import datetime
try:
	Nworkers=int(sys.argv[1])
except(IndexError):
	Nworkers=-1
try:
	randseed = int(sys.argv[2])
except(IndexError):
	randseed=111

import main 
sys.path.append("../..")
# Create Data File  
f = open("data.json",'a')
json.dump({"Creation Date":datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
			"N_worker":Nworkers,
			"random_state":randseed},f)
f.close()
print "Working on {0} workers with random seed = {1}".format(Nworkers,randseed)
from BB2TileExact import *
from TileEM_plot_toolbox import *
object_lst = list(object_tbl.id)
failed_objs=[]
for objid in tqdm(object_lst):
	# try:
	print "Working on Obj: ",objid
	createObjIndicatorMatrix(objid,load_existing_tiles_from_file=False,random_state=randseed,sampleNworkers=Nworkers,PRINT=True,SAVE=True,tqdm_on=True)
	# except:
	# 	print "Failed object :",objid
	# 	failed_objs.append(objid)
	# 	pass