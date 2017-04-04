import sys
from analysis_toolbox import *
if sys.argv[1]=='overall':
    df_stats_tbl = compute_all_fittings()
else:
    fit_results =test_all_Ji_fit_fcn(fcns_to_test=['norm','johnsonsu','exponpow','t','genpareto'],RAND_SAMPLING=False)
