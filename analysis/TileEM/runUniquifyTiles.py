import sys
import json
import datetime
try:
    Nworkers=int(sys.argv[1])
except(IndexError):
    Nworkers=-1
# import main
# # Create Data File
# f = open("data.json",'a')
# json.dump({"Creation Date":datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
#             "N_worker":Nworkers},f)
# f.close()
from BB2TileExact import *
from TileEM_plot_toolbox import *
object_lst = list(object_tbl.id)
failed_objs=[]
DATA_DIR="sampletopworst5"
# os.chdir(DATA_DIR)
for objid in tqdm(object_lst):
    # try:
    print "Working on Obj: ",objid
    tiles = pkl.load(open("{0}/tiles{1}.pkl".format(DATA_DIR,objid)))
    vtiles =compute_unique_tileset(tiles)
    pkl.dump(vtiles,open("{0}/vtiles{1}.pkl".format(DATA_DIR,objid),'w'))
    createObjIndicatorMatrix(objid,vtiles,sampleNworkers=Nworkers,PRINT=True,SAVE=True,tqdm_on= True)
    # except:
    #     print "Failed object :",objid
    #     failed_objs.append(objid)
    #     pass

