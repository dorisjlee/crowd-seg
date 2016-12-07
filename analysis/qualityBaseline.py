import numpy as np
# Given all the x and y annotations for that object, which contains all responses from every worker
# If we want to compute ground truth comparison simply input 
# obj_x_locs = [[worker i response],[ground truth]]
# obj_y_locs = [[worker i response],[ground truth]]

from PIL import Image, ImageDraw
def majority_vote(obj_x_locs,obj_y_locs,width,height): 
    '''
    Jaccard Simmilarity or Overlap Method
    used for PASCAL VOC challenge
    '''
    mega_mask = np.zeros((height,width))
    img = Image.new('L', (width, height), 0)
    for x_locs, y_locs in zip(obj_x_locs,obj_y_locs):
        ImageDraw.Draw(img).polygon(zip(x_locs,y_locs), outline=1, fill=1)
        mask = np.array(img)==1
    #     plt.imshow(mask)
        mega_mask+=mask
    #     plt.plot(x_locs,y_locs,'-',color="#f442df",linewidth=5)
    #     plt.fill_between(x_locs,y_locs,  color="none",facecolor='#f442df', alpha=0.4)
    # Show Majority Vote area
    # plt.imshow(mega_mask)
    # plt.colorbar()
    # Compute Jaccard Simmilarity 
    intersection = len(np.where(mega_mask == mega_mask.max())[0])
    union  =len(np.where(mega_mask !=0)[0])
    return float(union)/intersection


from munkres import Munkres, print_matrix

def MunkresEuclidean(bb1,bb2):
    '''
    Given two worker's responses, 
    Compares Euclidean distances of all points in the polygon, 
    then find the best matching (min dist) config via Kuhn-Munkres
    '''
    matrix = spatial.distance.cdist(bb1,bb2,'euclidean')
    m = Munkres()
    indexes = m.compute(np.ma.masked_equal(matrix,0))

    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
#         print '(%d, %d) -> %d' % (row, column, value)
    return total         

def DistAllWorkers(obj_x_locs,obj_y_locs,dist = MunkresEuclidean):
    '''
    Given all worker's responses,
    Perform pairwise distance comparison with all other workers
    returns quality for each worker
    #NOTE THIS NEEDS TO BE CHANGED TO INCORPORATE ALL PAIRWISE COMPARISONS
    '''
    minDistList=[]
    for i in np.arange(len(obj_x_locs)-1):
        # Compare worker with another worker
        bb1 = np.array([obj_x_locs[i],obj_y_locs[i]]).T
        bb2  = np.array([obj_x_locs[i+1],obj_y_locs[i+1]]).T
        minDistList.append(dist(bb1,bb2))
    #worker's scores
    return minDistList/max(minDistList)