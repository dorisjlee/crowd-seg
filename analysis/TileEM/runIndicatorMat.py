import sys
import json
import datetime
try:
	Nworkers=int(sys.argv[1])
except(IndexError):
	Nworkers=-1
import main 

# Create Data File  
f = open("data.json",'a')
json.dump({"Creation Date":datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
			"N_worker":Nworkers},f)
f.close()
from BB2TileExact import *
from TileEM_plot_toolbox import *
object_lst = list(object_tbl.id)
for objid in tqdm(object_lst):
	print "Working on Obj: ",objid
	createObjIndicatorMatrix(objid,load_existing_tiles=False,sampleNworkers=Nworkers,PRINT=True,SAVE=True,tqdm_on= True)
