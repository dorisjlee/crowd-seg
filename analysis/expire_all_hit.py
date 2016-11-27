#!flask/bin/python
# Script that disable/expires all current HITS released under me as a requester.
# Disable means completely delete the HIT
# Expire means Workers can't view it anymore but you can still review and approve/reject it.
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from secret import SECRET_KEY,ACCESS_KEY
DEV_ENVIROMENT_BOOLEAN = True
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
hits_lst = list(connection.get_all_hits())
print hits_lst
for hit in hits_lst:
	# print "Expiring HIT ID: ",hit.HITId
	# connection.expire_hit(hit.HITId)
	print "Disabling HIT ID: ",hit.HITId
	connection.disable_hit(hit.HITId)
