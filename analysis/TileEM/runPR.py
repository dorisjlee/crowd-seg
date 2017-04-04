import sys
from TileEM_plot_toolbox import *
os.chdir("exactOutput")
try:
	print "Running PR Curves while varying ",sys.argv[1]
except(IndexError):
	print '''
	  method='postprocess','T-search'
	  '''
object_lst = list(object_tbl.id)
if sys.argv[1]=='postprocess':
	for objid in tqdm(object_lst):
	    plot_all_postprocess_PR_curves(objid,legend=True) 
elif sys.argv[1]=='T-search':
	for objid in tqdm(object_lst):
	    plot_all_T_search_PR_curves(objid,'majority-top-k')
	    plot_all_T_search_PR_curves(objid,'tile-threshold')
	    plot_all_T_search_PR_curves(objid,'tile-top-k')