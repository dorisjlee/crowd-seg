#!flask/bin/python
# Script that disable/expires all current HITS released under me as a requester.
# Disable means completely delete the HIT
# Expire means Workers can't view it anymore but you can still review and approve/reject it.
from boto.mturk.connection import MTurkConnection
from secret import SECRET_KEY,ACCESS_KEY,AMAZON_HOST

#Start Configuration Variables
AWS_ACCESS_KEY_ID = ACCESS_KEY
AWS_SECRET_ACCESS_KEY = SECRET_KEY

connection = MTurkConnection(aws_access_key_id=AWS_ACCESS_KEY_ID,
							 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
							 host=AMAZON_HOST)
hits_lst = list(connection.get_all_hits())
print hits_lst
for hit in hits_lst:
	print "Expiring HIT ID: ",hit.HITId
	connection.expire_hit(hit.HITId)
	#print "Disabling HIT ID: ",hit.HITId
	#connection.disable_hit(hit.HITId)
