import sys
from TileEM_plot_toolbox import *
os.chdir("output")
try:
	print "Running Dual PR Curves for ",sys.argv[1]
except(IndexError):
	print '''
	  method='all','gamma_threshold','majority_top_k','gamma_top_k'
	  '''
object_lst = list(object_tbl.id)
if sys.argv[1]=='all':
	for method in ['gamma_threshold','majority_top_k','gamma_top_k']:
		print "Working on ",method
		for objid in tqdm(object_lst):
			plot_dual_PR_curves(objid,method= method,PLOT_WORKER=True)#,legend=True)
else:
	for objid in tqdm(object_lst):
		plot_dual_PR_curves(objid,method= sys.argv[1],PLOT_WORKER=True)#,legend=True)
