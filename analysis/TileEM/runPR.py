import sys
from TileEM_plot_toolbox import *
os.chdir("output")
try:
	print "Running PR Curves for ",sys.argv[1]
except(IndexError):
	print '''
	  method='gamma_threshold','majority_top_k','gamma_top_k'
	  '''
object_lst = list(object_tbl.id)
for objid in tqdm(object_lst):
    plot_dual_PR_curves(objid,method= sys.argv[1])