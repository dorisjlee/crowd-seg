import pandas as pd
import pickle as pkl
# Contain PRJs for best summarization score, Basic Pixel EM, Pixel MV 
Pixel_PR =pd.read_csv("Pixel_PR.csv",index_col=0)
PR_pixelEM_GT = pd.read_csv("pixel_em/GTfull_PRJ_table.csv")

updated_Pixel_PR= Pixel_PR
for thresh in [-4,-2,0,2,4]:
    PR_pixelEM_GTi =PR_pixelEM_GT[PR_pixelEM_GT["thresh"]==thresh]
    PR_pixelEM_GTi = PR_pixelEM_GTi.rename(index=str,columns={'EM_precision':'P [GT PixelEM thres={}]'.format(thresh),\
                       'EM_recall':'R [GT PixelEM thres={}]'.format(thresh),\
                       'EM_jaccard':'J [GT PixelEM thres={}]'.format(thresh),
                       'objid':"object_id",\
                       'num_workers':'Nworker',\
                        'sample_num':'batch_num'})
    PR_pixelEM_GTi = PR_pixelEM_GTi.drop('thresh',axis=1)
    updated_Pixel_PR= updated_Pixel_PR.merge(PR_pixelEM_GTi)

updated_Pixel_PR.to_csv("updated_Pixel_PR.csv")
