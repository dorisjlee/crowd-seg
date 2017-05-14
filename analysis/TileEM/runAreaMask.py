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
    for tile in ordered_tiles:
        x,y = tile.exterior.xy
        tarea = tile.area
        img = Image.new('L', ( np.shape(MV)[1], np.shape(MV)[0]), 0)
        ImageDraw.Draw(img).polygon(zip(x,y), outline=1, fill=1)
        mask = np.array(img)*tarea
        mega_mask+=mask
    pkl.dump(mega_mask,open("pixel_em/{}/obj{}/tarea_mask.pkl".format(sample,objid),'w'))

for sample in tqdm(sample_specs.keys()[23:]):
    for objid in range(1,48):
	try:
            tarea_mask(sample,objid)
	except(IOError):
	    pass
