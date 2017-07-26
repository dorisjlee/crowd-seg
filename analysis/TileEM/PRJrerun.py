from PixelEM import *
import glob
import pandas as pd

BASE_DIR = '/home/jlee782/crowd-seg/analysis/TileEM/'
PIXEL_EM_DIR = BASE_DIR + 'pixel_em/'

from sample_worker_seeds import sample_specs
from tqdm import tqdm

df = []
fieldnames = ['num_workers', 'sample_num', 'objid', 'thresh', 'EM_precision', 'EM_recall','EM_jaccard']
for sample_name in ['5workers_rand0','10workers_rand0','15workers_rand0','20workers_rand0','25workers_rand0','30workers_rand0']:#tqdm(sample_specs.keys()):
    print sample_name
    num_workers = int(sample_name.split('workers')[0])
    sample_num = int(sample_name.split("rand")[-1])
    for obj_path in glob.glob('{}{}/obj*/*_gt_est_ground_truth_mask_thresh*.pkl'.format(PIXEL_EM_DIR ,sample_name)):
        
        objid = int(obj_path.split("/")[-2][3:])
        pkl_fname=obj_path.split("/")[-1].split('_')
        algo = pkl_fname[0]
	if 'AW' in algo: 
            thresh = float(pkl_fname[-1].split("thresh")[-1][:-4])
            if thresh in [-2,-1,0,1,2]: 
	        #include only these parameter (the finer parameter were not used in the rerun)  
                result = pkl.load(open(obj_path))
                [p, r, j] = faster_compute_prj(result, get_gt_mask(objid)) 
                df.append([num_workers,sample_num,objid,algo,thresh,p,r,j])        

df_tbl = pd.DataFrame(df,columns=['num_workers', 'sample_num', 'objid', 'algorithm','thresh', 'precision', 'recall','jaccard'])
fname = '{}ground_truth_rerun_full_PRJ_table_AW.csv'.format(PIXEL_EM_DIR)
df_tbl.to_csv(fname)
print "saved to: ", fname 
