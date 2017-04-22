def ground_truth_T(object_id):
    my_BBG  = pd.read_csv("my_ground_truth.csv")
    ground_truth_match = my_BBG[my_BBG.object_id==object_id]
    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    T = Polygon(zip(x_locs,y_locs))
    return T
def estimate_Qj(T,tiles,indMat,workers):
    Qj=[]
    for wid,j in zip(workers,range(len(workers))):
        Qj.append(Qjfunc(indMat,T,j))
    return Qj