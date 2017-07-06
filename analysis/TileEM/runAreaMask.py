from tqdm import tqdm
from sample_worker_seeds import sample_specs
import numpy as np 
import pickle as pkl
from PIL import Image, ImageDraw

def tarea_mask(sample, objid):
    tiles = pkl.load(open("uniqueTiles/{}/vtiles{}.pkl".format(sample,objid)))
    MV = pkl.load(open("pixel_em/{}/obj{}/MV_mask.pkl".format(sample,objid)))
    mega_mask= np.zeros_like(MV)
    t_area = np.array([t.area for t in tiles])
    # sorted by largest to smallest area
    #so that small area assignments for a pixel overrides the large area assignments
    ordered_tiles = np.array(tiles)[np.argsort(t_area)[::-1]]
    tarea_lst = []
    # mask_lst = []
    overlap_count=0
    for tidx,tile in enumerate(ordered_tiles):
    #     print tidx,len(tarea_lst)
        x,y = tile.exterior.xy
        tarea = tile.area
        img = Image.new('L', ( np.shape(MV)[1], np.shape(MV)[0]), 0)
        ImageDraw.Draw(img).polygon(zip(x,y), outline=1, fill=1)
        mask = np.array(img)*(tidx+1) #tarea
        replace_overlap = np.where((mega_mask!=0)&(mask!=0))
        if len(replace_overlap[0])>0:
            overlap_count+=1
            mega_mask[replace_overlap]=0
        mega_mask+=mask
        tarea_lst.append(tarea)
    mega_mask[mega_mask==0]=0 
    # # NOTE: approximate outer area, in pixel units, not in the same units as the rest of polygonal tarea!!!
    approx_outer_area  = mega_mask.size-sum(tarea_lst)
    tarea_lst.insert(0,approx_outer_area)
    tidx_mask = mega_mask.astype('int')
    pkl.dump(tarea_lst,open("pixel_em/{}/obj{}/tarea.pkl".format(sample,objid),'w'))
    pkl.dump(tidx_mask,open("pixel_em/{}/obj{}/tidx_mask.pkl".format(sample,objid),'w'))
object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
for sample in tqdm(sample_specs.keys()):
    for objid in object_lst:
	try:
            tarea_mask(sample,objid)
	except(IOError):
	    pass
