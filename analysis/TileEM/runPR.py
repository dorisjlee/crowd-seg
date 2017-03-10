from TileEM_plot_toolbox import *
os.chdir("output")

object_lst = list(object_tbl.id)
for objid in tqdm(object_lst[5:]):
    plot_dual_PR_curves(objid)