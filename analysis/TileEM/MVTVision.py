from TileEM_plot_toolbox import *
from qualityBaseline import *
from glob import glob
import numpy as np 
worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
sampleN_lst=worker_Nbatches.keys()
for Nworker in sampleN_lst:
    for batch_num in range(worker_Nbatches[Nworker]):
        dir_name = "{0}worker_rand{1}".format(Nworker,batch_num)
        print dir_name
        os.chdir("sample/"+dir_name)
        for objid in object_lst:
            tiles = pkl.load(open("vtiles{}.pkl".format(objid)))
            indMat = pkl.load(open("indMat{}.pkl".format(objid)))
            workers = pkl.load(open("worker{}.pkl".format(objid)))

            area = indMat[-1]
            votes = indMat[:-1].sum(axis=0)

            tidx=np.where(votes>np.shape(indMat[:-1])[0]/2.)[0]
            #MVT_T = join_tiles(tidx,tiles)
            #pkl.dump(MVT_T,open("MVT{}.pkl".format(objid),'w'))
	    pkl.dump(tidx,open("MVT_idx_{}.pkl".format(objid),'w'))
        os.chdir("../../")
