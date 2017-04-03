from BB2TileExact import *
from TileEM_plot_toolbox import *
object_lst = list(object_tbl.id)
for objid in tqdm(object_lst):
	print "Working on Obj: ",objid
	createObjIndicatorMatrix(objid,load_existing_tiles=True,sampleNworkers=-1,PRINT=True,SAVE=True,tqdm_on= True)
