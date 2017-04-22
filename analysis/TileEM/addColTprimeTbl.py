from TileEM_plot_toolbox import *
from scatter_toolbox import *
from TileEM_Models import *
my_BBG  = pd.read_csv("../my_ground_truth.csv")
os.chdir(DATA_DIR)
df = pd.read_csv("all_tile_combo_metric_snowball.csv",index_col=0)
df = df[:100]
def add_column_to_Tprime_tbl(df,func_name_lst):
    all_val_lst=[]
    for Tprime_data in tqdm(df.iterrows()):
        #Tprime_data=df._ix[i]
        objid = Tprime_data[1]["objid"]
        Tprime_idx = ast.literal_eval(Tprime_data[1]["T prime"])
        tiles = pkl.load(open("vtiles{}.pkl".format(objid)))
        # Ground truth
        ground_truth_match = my_BBG[my_BBG.object_id==objid]
        x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
        BBG = shapely.geometry.Polygon(zip(x_locs,y_locs))
        ########### Insert Function calls here ############
        #for func in func_lst:
        #os.chdir("final_all_tiles/")
        val_lst=[]
        val_lst.append(AreaTprimeScore(objid,Tprime_idx,BBG))
        for A_percentile in [90,95,99]:
            val_lst.append(pTprimeGTLSA(objid,Tprime_idx,BBG,A_percentile))
        all_val_lst.append(val_lst)
        #os.chdir("..")
    funcVal = np.array(all_val_lst).T
    # print shape(funcVal)
    # print funcVal
    #print len(funcVal[0])
    #print len(df)
    for func_name,val_lst in zip(func_name_lst,funcVal):
        df[func_name]=val_lst
    return df
def compute_jaccard(objid,solnset,tiles):
    '''
    Compute Jaccard Index against ground truth bounding box
    for a given solution set and tile coordinates.
    '''
    if len(solnset)==1:
        joined_bb=tiles[solnset[0]]
        problematic_tiles=[]
    else:
        try:
            joined_bb,problematic_tiles = join_tiles(solnset,tiles)
        except(ValueError):
            return -1,-1
    ground_truth_match = my_BBG[my_BBG.object_id==objid]
    x_locs,y_locs =  process_raw_locs([ground_truth_match["x_locs"].iloc[0],ground_truth_match["y_locs"].iloc[0]])
    BBG = shapely.geometry.Polygon(zip(x_locs,y_locs))
    if problematic_tiles!=[]:
        intersect_area =0
        joined_bb_area =0
        for jbb in joined_bb:
            ia = intersection_area(BBG,jbb)
            intersect_area += ia
            joined_bb_area += jbb.area
    else:
        intersect_area=intersection_area(BBG,joined_bb)
        joined_bb_area =joined_bb.area
    try:
        union_area = shapely.ops.unary_union([joined_bb,BBG]).area
    except(ValueError):
        union_area=joined_bb.area+BBG.area
    jaccard = intersect_area/float(union_area)
    return jaccard
# df_new = add_column_to_Tprime_tbl(df,compute_jaccard,"Jaccard")

# df_new = add_column_to_Tprime_tbl(df,AreaTprimeScore,"AreaTprimeScore")
# pTprimeGTLSA(objid,Tprime,T,A_percentile)
new_df = add_column_to_Tprime_tbl(df,["AreaTprimeScore","pTprimeGTLSA[Athres>90%]","pTprimeGTLSA[Athres>95%]","pTprimeGTLSA[Athres>99%]"])
new_df.to_csv("test_new_all_tile_combo_metric_snowball.csv")
