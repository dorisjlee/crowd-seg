from analysis_toolbox import * 
from qualityBaseline import *
from PIL import Image

img_info,object_info,bb_info,hit_info=load_info()

def get_img_size(object_id):
    img_name = img_info[img_info.id==int(object_info[object_info.id==object_id]["image_id"])]["filename"].iloc[0]
    fname = "../../web-app/app/static/"+img_name+".png"
    #Open image for computing width and height of image
    im = Image.open(fname)
    width = im.size[0]
    height = im.size[1]
    return width* height
def create_object_tbl():
    # Task difficulty 
    task_ambiguity = [1,4,35,40,41,42]
    small_area = [1,15,22]
    high_numPts=[5,7,9,10,11,12,28,29,30,37]
    lossless_vtiles=[ 5,  6,  8,  9, 14, 15, 18, 19, 20, 21, 23, 24, 25, 27, 29, 30,33, 35, 37, 46] #  with decent looking vtiles 
    all_hard_tasks = list(set(np.concatenate((task_ambiguity,small_area,high_numPts))))
    easy_tasks = [objid  for objid in object_lst if objid not in all_hard_tasks]
    selected_objids = [objid  for objid in easy_tasks if objid not in lossless_vtiles]

    object_tbl = []
    for objid in object_lst:
        gt = ground_truth_T(objid)
        gt_numPts = len(gt.boundary.coords.xy[0])
        is_hard = False
        is_task_ambiguity=False
        is_small_area = False
        is_high_numPts = False

        if objid in all_hard_tasks:
            is_hard=True
            if objid in task_ambiguity: is_task_ambiguity=True
            elif objid in small_area: is_small_area = True
            elif objid in high_numPts: is_high_numPts = True
        object_tbl.append([objid,gt.area,get_img_size(objid),gt_numPts,is_hard,is_task_ambiguity,is_small_area,is_high_numPts])
    object_tbl = pd.DataFrame(object_tbl,columns=["objid","GT area","Image area","GT num Points","is_hard","is_task_ambiguity","is_small_area","is_high_numPts"])
    object_tbl.to_csv("all_object_info.csv")
def create_worker_tbl():
    computed_GT_info =  pd.read_csv("../computed_my_COCO_BBvals.csv",index_col=0)

    keys=['Num Points','Area Ratio','Precision [Self]','Recall [Self]','Jaccard [Self]','TPR [Self]','FNR [Self]','TNR [Self]', 'FPR [Self]']
    img_info,object_tbl,bb_info,hit_info = load_info()

    worker_tbl = []
    GT_info_keys = ['Num Points','Area Ratio','Precision [Self]','Recall [Self]','Jaccard [Self]','TPR [Self]','FNR [Self]','TNR [Self]', 'FPR [Self]']
    for objid in object_lst:
        bb_objects = bb_info[bb_info["object_id"]==objid]
        worker_lst  = list(bb_objects.worker_id)
        for worker_id in worker_lst:
            if worker_id>3:
                numPts,area_ratio, P,R,J,TPR,FNR,TNR,FPR = computed_GT_info[(computed_GT_info["object_id"]==objid)&(computed_GT_info["worker_id"]==worker_id)][GT_info_keys].values[0]
                worker_tbl.append([objid,worker_id,numPts,area_ratio, P,R,J,TPR,FNR,TNR,FPR])
    worker_tbl = pd.DataFrame(worker_tbl,columns=["objid","worker id",'Num Points','Area Ratio','P [GT]','R [GT]','J [GT]','TPR [GT]','FNR [GT]','TNR [GT]', 'FPR [GT]'])
    worker_tbl.to_csv("all_worker_response_info.csv")

def create_object_batch_tbl_tile():
    Tile_PR = pd.read_csv("Tile_PR_all.csv",index_col=0)
    object_batch_tbl = []
    Tile_PR_keys = ['P [MVT]', 'R [MVT]','J [MVT]']
    for batch in ['5workers_rand0','10workers_rand0','15workers_rand0','20workers_rand0','25workers_rand0','30workers_rand0']:
        Nworker = int(batch.split('workers')[0])
        batch_num = int(batch.split('rand')[-1])

        for objid in object_lst:
            worker_lst = pkl.load(open("uniqueTiles/{}/worker{}.pkl".format(batch,objid)))
            indMat = pkl.load(open("uniqueTiles/{}/indMat{}.pkl".format(batch,objid)))
    #         worker_keys,tidxs = np.where(indMat[:-1]!=0)
            try:
                MV_P, MV_R, MV_J = Tile_PR[(Tile_PR["object_id"]==objid)&(Tile_PR["Nworker"]==Nworker)&(Tile_PR["batch_num"]==batch_num)][Tile_PR_keys].values[0]
            except(IndexError):
                #More Tile stuff skipped because they might not have been computed since intersection issue and stuff
    #                 print "skipped object ",objid, Nworker,batch_num
                pass
            object_batch_tbl.append([batch,objid,MV_P,MV_R, MV_J])
    object_batch_tbl = pd.DataFrame(object_batch_tbl,columns=["batch","objid",'P [MVT]', 'R [MVT]','J [MVT]'])
    object_batch_tbl.to_csv("all_object_batch_info.csv")
def create_worker_tile_tbl_tile():
    worker_tile_tbl = []
    for batch in ['5workers_rand0','10workers_rand0','15workers_rand0','20workers_rand0','25workers_rand0','30workers_rand0']:
        Nworker = int(batch.split('workers')[0])
        batch_num = int(batch.split('rand')[-1])
        for objid in object_lst:
            worker_lst = pkl.load(open("uniqueTiles/{}/worker{}.pkl".format(batch,objid)))
            indMat = pkl.load(open("uniqueTiles/{}/indMat{}.pkl".format(batch,objid)))
            worker_keys,tidxs = np.where(indMat[:-1]!=0)
            for worker_key,tidx in zip(worker_keys,tidxs):
                if worker_key>3:
                    worker_tile_tbl.append([batch,objid,worker_lst[worker_key],tidx])

    worker_tile_tbl = pd.DataFrame(worker_tile_tbl,columns=["batch","objid","worker id","tile id"])
    worker_tile_tbl.to_csv("all_worker_tile_info.csv")
def create_tile_tbl_tile():
    tile_tbl =[]
    for batch in ['5workers_rand0','10workers_rand0','15workers_rand0','20workers_rand0','25workers_rand0','30workers_rand0']:
        for objid in object_lst:
            vtiles = pkl.load(open("uniqueTiles/{}/vtiles{}.pkl".format(batch,objid)))
            for tidx in range(len(vtiles)):
                tile = vtiles[tidx]
                gt = ground_truth_T(objid)
                try:
                    if tile.intersects(gt):
                        overlap_area  = intersection_area(tile,gt)
                    else:
                        overlap_area  = 0
                except(shapely.geos.TopologicalError):
                    overlap_area=-1
                tarea = tile.area
                tile_tbl.append([batch,objid,tidx,tarea,overlap_area])
    tile_tbl = pd.DataFrame(tile_tbl,columns=["batch","objid","tile id","tile area", "gt overlap area"])
    tile_tbl.to_csv("all_tile_info.csv")
