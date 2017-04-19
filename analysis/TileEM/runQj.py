# Compute individual worker qualities based on ground truth data as T
from TileEM_Models import *
#QjBasic(SAVE=True)
# when A_thres=200 + then some objects don't have tiles larger than that area so q1 calculation errors
# q1 = large_Ncorrect/float(large_Ncorrect+large_Nwrong)  ZeroDivisionError: float division by zero
# because 0/0
#[1,10,50,100]:
for A_thres in [200,500]:
    print "Working on A_thres:",A_thres
    QjLSA(A_thres,SAVE=True)
