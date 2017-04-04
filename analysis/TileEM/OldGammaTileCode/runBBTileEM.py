from BB2tile import *
from dataset import Dataset
img_info,object_tbl,bb_info,hit_info = load_info()
obj_lst = list(set(object_tbl.object_id))
my_BBG  = pd.read_csv("../my_ground_truth.csv")
object_id = 18
ground_truth_match = my_BBG[my_BBG.object_id==object_id]
BBG =  Polygon(zip(*process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])))
tiles, objIndicatorMat = createObjIndicatorMatrix(object_id)
T_true =BBG.area
tile_dataset = Dataset(tiles,objIndicatorMat,100)
solution = greedySearch(tile_dataset)
solution.printSolution()
