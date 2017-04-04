import numpy as np
import pandas as pd
import ast
from boto.mturk.connection import MTurkRequestError
from boto.mturk.connection import MTurkConnection
from secret import SECRET_KEY,ACCESS_KEY,AMAZON_HOST

#Start Configuration Variables
AWS_ACCESS_KEY_ID = ACCESS_KEY
AWS_SECRET_ACCESS_KEY = SECRET_KEY

connection = MTurkConnection(aws_access_key_id=AWS_ACCESS_KEY_ID,
                             aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                             host=AMAZON_HOST)

all_hits = [hit for hit in connection.get_all_hits()]
all_reviewable_hits = [hit for hit in connection.get_reviewable_hits()]

for hit in all_hits:
    assignments = connection.get_assignments(hit.HITId)
    print assignments
    for assignment in assignments:
        print assignment
        print "Answers of the worker %s" % assignment.WorkerId
