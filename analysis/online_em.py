# Implementing the continuous valued case of the online EM algorithm
# Peter Welinder and Pietro Perona. 2010. Online crowdsourcing: Rating annotators and obtaining cost-effective labels. 
# 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition - Workshops, CVPRW 2010: 25–32. 

################################################
##                                            ##
##        Preprocessing and Alignment         ##
##                                            ##
################################################
def coord_to_bb(bbw,bbh,type='model'): 
    '''
    Given 4 coordinates, 
    if all : generate the set of 4 coordinates (x,y) for this BB
    if model : generate the set of top-left and lower right coordinates (x,y) for this BB
    '''
    bbw = sort(bbw)
    bbh = sort(bbh)
    if type=="all":
        return np.array([[bbw[0],bbh[1]],
                [bbw[1],bbh[1]],
                [bbw[1],bbh[0]],
                [bbw[0],bbh[0]],
               ],dtype=float)
    elif type=="model":
        return np.array([bbw[0],bbh[1],bbw[1],bbh[0]])

def make_synthetic_bb(N_sample):
    # Generate Synthetic Bounding Boxes 
    img_width=5
    img_height=10
    random.seed(131)

    data =[]
    for _i in range(N_sample):
        bbw = [random.randint(img_width,img_height) for x in range(2)]
        bbh = [random.randint(img_width,img_height) for x in range(2)]
        data.append(coord_to_bb(bbw,bbh,'model'))
    return data

from scipy.interpolate import splprep,splev
def single_parametric_interpolate(obj_x_loc,obj_y_loc,numPts=50):
    '''
    Interpolate a single given bounding box obj_x_loc,obj_y_loc
    return a new set of coordinates interpolated on numPts 
    '''
    tck, u =splprep(np.array([obj_x_loc,obj_y_loc]),s=0,per=1)
    u_new = np.linspace(u.min(),u.max(),numPts)
    new_points = splev(u_new, tck,der=0)
    return new_points

def single_randomwalk_interpolate(obj_x_loc,obj_y_loc,numPts=50):
	'''
	Random sampling numPts points along the BB polygon's boundary
	by random walk from the vertices given by the xy coordinates
	'''
    n = len(obj_x_loc)
    vi = [[obj_x_loc[(i+1)%n] - obj_x_loc[i],
         obj_y_loc[(i+1)%n] - obj_y_loc[i]] for i in range(n)]
    si = [np.linalg.norm(v) for v in vi]
    di = np.linspace(0, sum(si), numPts, endpoint=False)
    new_points = []
    for d in di:
        for i,s in enumerate(si):
            if d>s: d -= s
            else: break
        l = d/s
        new_points.append([obj_x_loc[i] + l*vi[i][0],
                           obj_y_loc[i] + l*vi[i][1]])
    return new_points

def interpolate_align_bb():
    '''
    Interpolate and Align bounding boxes and save results to interpolated_aligned_bb_info.csv
    1. subsample 50 points 
    2. compute distance to origin 
    3. find point with min distance to origin 
    4. determine clockwise or counter clockwise by:
        seeing if the next array element is to the right of this point 
        concatenate the array portions accordingly 
    '''
    img_info,object_tbl,bb_info,hit_info = load_info()
    coord_lst =[]
    icoord_lst =[]
    aicoord_lst =[]
    aicoord_tbl=[]
    for bb in tqdm(list(bb_info.iterrows())):
        oid = bb[1]["object_id"]
        wid = bb[1]["worker_id"]
        #Image information 
        image_id = int(object_tbl[object_tbl.object_id==oid].image_id)
        img_name = img_info["filename"][image_id-1]

        bbx_path= bb[1]["x_locs"]
        bby_path= bb[1]["y_locs"]
        worker_x_locs,worker_y_locs= process_raw_locs([bbx_path,bby_path])
        worker_x_locs,worker_y_locs = zip(*list(OrderedDict.fromkeys(zip(worker_x_locs,worker_y_locs))))
        coord_lst.append(zip(worker_x_locs,worker_y_locs))

        ixylocs = single_randomwalk_interpolate(worker_x_locs,worker_y_locs)
        icoord = np.array(ixylocs) #interpolated coordinates
        icoord_lst.append(icoord)
        # find point with min distance to origin
        oidx=np.argmin(sqrt(icoord[:,0]**2+icoord[:,1]**2))
        # aligning coordinates
        try:
            next_coord  = icoord[oidx+1]
        except(IndexError):
            # oidx is last element of the array
            next_coord  = icoord[0]
        if icoord[oidx][0] <next_coord[0]:
            # CW
            aicoord = np.concatenate((icoord[oidx:-1],icoord[:oidx]))
        else: 
            # CCW
            aicoord = np.concatenate((icoord[:oidx-1][::-1],icoord[oidx:][::-1]))
        aicoord_lst.append(aicoord)
        aicoord_tbl.append([oid,wid,str(list(aicoord[:,0])),str(list(aicoord[:,1]))])
    ai_tbl = pd.DataFrame(aicoord_tbl,columns=["object_id","worker_id","aix_locs","aiy_locs"])
    ai_tbl.to_csv("interpolated_aligned_bb_info.csv")
    return ai_tbl
def plot_interpolated_aligned_obj_data(oid,numPts=50):
    '''
    Plot the Interpolated aligned data for the object corresponding to the given oid 
    The colorbar indicates the index of the points 
    Should observe a perfect gradation when alignment is done right
    '''
    ai_tbl= pd.read_csv("interpolated_aligned_bb_info.csv",index_col=0)
    ai_tbl = ai_tbl[ai_tbl["object_id"]==oid]

    selected_idx=range(numPts-1)
    steps = np.linspace(0, 1, len(selected_idx))
    colors = cm.rainbow(steps)
    Z = [[0,0],[0,0]]
    CS3 = plt.contourf(Z, selected_idx, cmap=cm.rainbow)
    plt.clf()
    plt.colorbar(CS3)
    colors = cm.rainbow(np.linspace(0, 1, len(selected_idx)))
    for coord in ai_tbl.iterrows():
        x,y = process_raw_locs([coord[1]["aix_locs"],coord[1]["aiy_locs"]])
        coord = zip(x,y)
        for i in selected_idx:
            x,y = coord[i]
            plt.plot(x,y,'.',color=colors[selected_idx.index(i)],label = "Point {}".format(i))    
    plt.gca().invert_yaxis()
    plt.title("Interpolated, Aligned Data [Obj{}]".format(oid),fontsize=13)
    plt.savefig('interpolated_aligned_data_obj{}.pdf'.format(oid))