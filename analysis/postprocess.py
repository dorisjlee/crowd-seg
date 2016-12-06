import numpy as np
import pandas as pd
from analysis_toolbox import *
import ast
from boto.mturk.connection import MTurkRequestError
from boto.mturk.connection import MTurkConnection
import datetime
from secret import SECRET_KEY,ACCESS_KEY,AMAZON_HOST

save_db_as_csv(connect=False)
print "Run DB update separately"
img_info,object_tbl,bb_info,hit_info = load_info()
print 'Updated DB info'

#Start Configuration Variables
AWS_ACCESS_KEY_ID = ACCESS_KEY
AWS_SECRET_ACCESS_KEY = SECRET_KEY

connection = MTurkConnection(aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                             host=AMAZON_HOST)
print 'Connected to AMT'

#all_hits = [hit for hit in connection.get_reviewable_hits()]
all_hits = [hit for hit in connection.get_all_hits()]
print all_hits
error_log = open("error.log",'w')
if os.path.isfile('reject.log') :
    reject_log = open("reject.log",'a')
    reject_log.write("-------------------------------------")
    reject_log.write("Post-Processed New Batch: "+str(datetime.datetime.now()))
else:
    reject_log = open("reject.log",'w')
    reject_log.write("AssignmentId,numClicks,BBbox,object_id,worker_id")

hit_info['status'] = pd.Series(['undetermined' for _i in range(len(hit_info))], index=hit_info.index)
object_tbl['approvedBB_count'] = pd.Series(np.zeros(len(object_tbl)), index=object_tbl.index)
print object_tbl #must have
for hit in all_hits:
    assignments = connection.get_assignments(hit.HITId)
    for assignment in assignments:
        task = hit_info[hit_info.hit_id==assignment.HITId]
        if len(task)>0:
            actions = ast.literal_eval(task.actions.get_values()[0])
            numClicks = actions.count("addClick")
            try:
                if numClicks >=3:
                    hit_info = hit_info.set_value(task.index[0],'status',"approved")
                    #Add additional counts to object table
                    # new_approved_count= object_tbl._iloc[int(task.object_id)].approvedBB_count+1
                    try:
                        new_approved_count=object_tbl._iloc[task.object_id].approvedBB_count.get_values()[0]+1
                        object_tbl = object_tbl.set_value(task.object_id,'approvedBB_count', new_approved_count)
                    except(IndexError):
                        print "something wrong with updating indexing "
                    connection.approve_assignment(assignment.AssignmentId)
                    print 'approved ', assignment.AssignmentId
                else:
                    #print 'Putting into pending reject list :', assignment.AssignmentId
                    print 'Reject ',assignment.AssignmentId, 'only :', numClicks
                    bb = bb_info[(bb_info.worker_id==int(task.worker_id)) & (bb_info.object_id==int(task.object_id))]
                    BB ='"'+ bb.x_locs.get_values()[0]+'"'
                    reject_log.write('{0},{1},{2},{3},{4}\n'.format(assignment.AssignmentId,numClicks,BB,int(task.object_id),int(task.worker_id)))
                    hit_info = hit_info.set_value(task.index[0],'status',"rejected")
                    connection.reject_assignment(assignment.AssignmentId)
            except MTurkRequestError:
                #Problably already approved or rejected this assignment previously
               #print "Previously rejected or approved : ", assignment.AssignmentId
               error_log.write(assignment.AssignmentId+'\n')
reject_log.close()
error_log.close()
object_tbl.to_csv('../../data/approved_object_tbl.csv')
hit_info.to_csv('../../data/hit_mturk.csv')
# Count the number of objects Annotated
print 'Count the number of objects Annotated'
import pandas as pd
object_tbl['BB_count'] = pd.Series(np.zeros(len(object_tbl)), index=object_tbl.index)
for hit in hit_info.iterrows():
    objIdx = hit[1]['object_id']-1
    object_tbl = object_tbl.set_value(objIdx,'BB_count', object_tbl._iloc[objIdx].BB_count+1)  
object_tbl.to_csv("../../data/BB_count_tbl.csv")                  