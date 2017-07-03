from PixelEM import *
import glob
import pandas as pd

BASE_DIR = '/home/jlee782/crowd-seg/analysis/TileEM/'
PIXEL_EM_DIR = BASE_DIR + 'pixel_em/'

from sample_worker_seeds import sample_specs
from tqdm import tqdm


mode="isoGT"

df = []
fieldnames = ['num_workers', 'sample_num', 'objid', 'thresh', 'EM_precision', 'EM_recall','EM_jaccard']
for sample_name in tqdm(sample_specs.keys()):
    print sample_name
    num_workers = int(sample_name.split('workers')[0])
    sample_num = int(sample_name.split("rand")[-1])
    for obj_path in glob.glob('{}{}/obj*/*_gt_est_ground_truth_mask_thresh*.pkl'.format(PIXEL_EM_DIR ,sample_name)):
        
        objid = int(obj_path.split("/")[-2][3:])
        pkl_fname=obj_path.split("/")[-1].split('_')
        algo = pkl_fname[0]
        thresh = float(pkl_fname[-1].split("thresh")[-1][:-4])
        
        result = pkl.load(open(obj_path))
        [p, r, j] = faster_compute_prj(result, get_gt_mask(objid)) 
        df.append([num_workers,sample_num,objid,algo,thresh,p,r,j])        

df_tbl = pd.DataFrame(df,columns=['num_workers', 'sample_num', 'objid', 'algorithm','thresh', 'precision', 'recall','jaccard'])

df_tbl.to_csv('{}ground_truth_rerun_full_PRJ_table.csv'.format(PIXEL_EM_DIR))

print 'save to {}ground_truth_rerun_full_PRJ_table.csv'.format(PIXEL_EM_DIR)
