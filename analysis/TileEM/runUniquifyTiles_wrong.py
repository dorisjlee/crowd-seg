import sys
import json
import datetime
#sys.path.append("..")
from glob import glob
from BB2TileExact import *
from TileEM_plot_toolbox import *
object_lst = list(object_tbl.id)
failed_objs=[]
base_dir="final_run"
sys.path.append(base_dir)
import poly_utils
worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
sampleN_lst=worker_Nbatches.keys()
for Nworker in sampleN_lst:
    for batch_num in range(worker_Nbatches[Nworker]):
	dir_name = "{0}worker_rand{1}".format(Nworker,batch_num)
        print "Working on :", dir_name
        #os.chdir(base_dir+"/"+dir_name)
	DATA_DIR = base_dir+"/"+dir_name
        for objtile in glob(DATA_DIR+"/tile*.pkl"):
	    tiles = pkl.load(open(objtile))
	    objid = int(objtile.split("/")[-1].split(".")[0][5:])
	    tiles = pkl.load(open("{0}/tiles{1}.pkl".format(DATA_DIR,objid)))
            vtiles,overlap_area,total_area=uniqify(tiles, overlap_threshold=0.2, SAVE=False, SAVEPATH=None, PLOT=False)
            print "Overlap ratio:",overlap_area/float(total_area)
            pkl.dump(vtiles,open("{0}/vtiles{1}.pkl".format(DATA_DIR,objid),'w'))
            createObjIndicatorMatrix(objid,vtiles,sampleNworkers=Nworkers,PRINT=True,SAVE=True,tqdm_on= True)
	    print objid 
        #os.chdir("../../")
	#print "Working on Obj: ",objid
	#tiles = pkl.load(open("{0}/tiles{1}.pkl".format(DATA_DIR,objid)))
	#vtiles =compute_unique_tileset(tiles)
	#pkl.dump(vtiles,open("{0}/vtiles{1}.pkl".format(DATA_DIR,objid),'w'))
	#createObjIndicatorMatrix(objid,vtiles,sampleNworkers=Nworkers,PRINT=True,SAVE=True,tqdm_on= True)
	# except:
	# 	print "Failed object :",objid
	# 	failed_objs.append(objid)
	# 	pass
