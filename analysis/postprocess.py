import numpy as np
import pandas as pd
from analysis_toolbox import *
import ast
from boto.mturk.connection import MTurkRequestError
from boto.mturk.connection import MTurkConnection
from secret import SECRET_KEY,ACCESS_KEY
DEV_ENVIROMENT_BOOLEAN = True

save_db_as_csv()
img_info,object_tbl,bb_info,hit_info = load_info()
print 'Updated DB info'


#This allows us to specify whether we are pushing to the sandbox or live site.
if DEV_ENVIROMENT_BOOLEAN:
    AMAZON_HOST = 'mechanicalturk.sandbox.amazonaws.com'
    MastersQualID = '2F1KVCNHMVHV8E9PBUB2A4J79LU20F'
else:
    AMAZON_HOST = 'mechanicalturk.amazonaws.com'
    MastersQualID = '2NDP2L92HECWY8NS8H3CK0CP5L9GHO'

#Start Configuration Variables
AWS_ACCESS_KEY_ID = ACCESS_KEY
AWS_SECRET_ACCESS_KEY = SECRET_KEY

connection = MTurkConnection(aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                             host=AMAZON_HOST)
print 'Connected to AMT'

all_hits = [hit for hit in connection.get_all_hits()]
error_log = open("error.log",'w')
hit_info['status'] = pd.Series(['undetermined' for _i in range(len(hit_info))], index=hit_info.index)
object_tbl['approvedBB_count'] = pd.Series(np.zeros(len(object_tbl)), index=object_tbl.index)
for hit in all_hits:
    assignments = connection.get_assignments(hit.HITId)
    for assignment in assignments:
        task = hit_info[hit_info.hit_id==assignment.HITId]
        if len(task)>0:
            actions = ast.literal_eval(task.actions.get_values()[0])
            numClicks = actions.count("addClick")
            try: 
                if numClicks >3:
                    print 'approving ', assignment.AssignmentId
                    hit_info = hit_info.set_value(task.index[0],'status',"approved")
                    #Add additional counts to object table 
                    object_tbl = object_tbl.set_value(task.object_id,'approvedBB_count', \
                    object_tbl._iloc[task.object_id].approvedBB_count.get_values()[0]+1)                    
                    connection.approve_assignment(assignment.AssignmentId)
                else:
                    print 'rejecting ', assignment.AssignmentId
                    hit_info = hit_info.set_value(task.index[0],'status',"rejected")
                    connection.reject_assignment(assignment.AssignmentId)
            except:#(MTurkRequestError):
                print "Problem with rejecting/approving: ", assignment.AssignmentId
                error_log.write(assignment.AssignmentId+'\n')
error_log.close()
object_tbl.to_csv('../../data/object_tbl_mturk.csv')
hit_info.to_csv('../../data/hit_mturk.csv')