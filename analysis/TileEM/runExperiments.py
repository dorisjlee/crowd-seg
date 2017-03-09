from numpy import  *
from dataset import Dataset
from BB2tile import *
from greedy import *
from data import *
from experiment import *
import pandas as pd
import string
from tqdm import tqdm
import pickle as pkl
def run_all_experiments(tiles, objIndicatorMat,exhaustive_step_size=1000):
    Ta, La, ga, solutionLista = experiment_avg(objIndicatorMat)
    Tm, Lm, gm, solutionListm = experiment_median(objIndicatorMat)
    Tl, Ll, gl, solutionListl = experiment_local(1, objIndicatorMat)
    Te, Le, ge, solutionListe = experiment_exhaustive(1,  objIndicatorMat,step_size=exhaustive_step_size)
    Le = np.array(Le)
    try:
        Le = Le[Le!=np.inf]
        maxidx = argmax(Le)
        Le = Le[maxidx] #Maximum likelihood
        Te = Te[maxidx]
    except(ValueError):
        print Le
        Le = np.inf
        Te =np.nan
    return (Ta,Tm,Tl,Te),(La,Lm,Ll,Le),\
            np.array([gmat2arr(ga),gmat2arr(gm),gmat2arr(gl),gmat2arr(ge)]),\
            [solutionLista,solutionListm,solutionListl,solutionListe]
def gmat2arr(g):
    if g is None:
        return []
    else:
        return np.array(g.T)[0]
if __name__ == "__main__":
    img_info,object_tbl,bb_info,hit_info = load_info()
    object_lst = list(object_tbl.id)
    Tfile = open("output/Tarea.txt",'a')
    Lfile = open("output/likelihood.txt",'a')
    for objid in tqdm(object_lst):
        print "Working on obj:",objid
        tiles, objIndicatorMat = createObjIndicatorMatrix(objid,PRINT=False)
        print "Saving checkpoint..."
        Tfile = open("output/Tarea.txt",'a')
        Lfile = open("output/likelihood.txt",'a')
        T,L,g,soln = run_all_experiments(tiles, objIndicatorMat)

        Tfile.write(T.__repr__().replace('(','').replace(')','\n'))
        Lfile.write(L.__repr__().replace('(','').replace(')','\n'))

        Tfile.close()
        Lfile.close()
        pkl.dump(g,open("output/gfile{}.pkl".format(objid),'w'))
        pkl.dump(soln,open("output/solnfile{}.pkl".format(objid),'w'))
        pkl.dump(tiles,open("output/tiles{}.pkl".format(objid),'w'))
    Tfile.close()
    Lfile.close()
