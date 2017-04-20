# Compute individual worker qualities based on ground truth data as T
from TileEM_Models import *
#QjBasic(SAVE=True)
# for A_thres in [200,500]:
#     print "Working on A_thres:",A_thres
#     QjLSA(A_thres,SAVE=True)
for A_thres in [1,5,10,50,100]:
    print "Working on A_thres:",A_thres
    QjGTLSA(A_thres,SAVE=True)