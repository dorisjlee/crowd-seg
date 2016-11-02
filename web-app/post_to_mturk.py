#!flask/bin/python
import time
import glob
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import Qualifications, PercentAssignmentsApprovedRequirement, NumberHitsApprovedRequirement, Requirement
from boto.mturk.price import Price
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

HIT_TYPE = "SEGMENT" #or "IDENTIFY"

print "Connected."
#frame_height in pixels
frame_height = 800


#Here, I create two sample qualifications
qualifications = Qualifications()
#qualifications.add(Requirement(MastersQualID,'DoesNotExist',required_to_preview=True))
qualifications.add(PercentAssignmentsApprovedRequirement(comparator="GreaterThan", integer_value="95"))
qualifications.add(NumberHitsApprovedRequirement(comparator="GreaterThan", integer_value="500"))

#This url will be the url of your application, with appropriate GET parameters
with open('ActiveHITs','a') as f:
	f.write('New batch created on : '+time.ctime())
	for fname in glob.glob("app/static/COCO_*.png")[:2]:
		img_name = fname.split('/')[-1].split('.')[0]
		if HIT_TYPE == "IDENTIFY":
			url = "https://crowd-segment.herokuapp.com/identify/{}".format(img_name)
			questionform = ExternalQuestion(url, frame_height)
			create_hit_result = connection.create_hit(
				title="Mark objects on an image",
				description="We'll give you an image, and you have to identify objects inside the image. There is 1 image per HIT (and we'll only load it once). Our interface supports keyboard input for speed!",
				keywords=["identification", "perception", "image", "fast"],
				duration = 1800,
				max_assignments=20,
				question=questionform,
				reward=Price(amount=0.15),
				lifetime=43200)#,
				#qualifications=qualifications)
			hit_id = str(create_hit_result[0].HITId)
			f.write(hit_id + "\n")
			print "Created HIT: ",hit_id
		elif HIT_TYPE == "SEGMENT":
			url = "https://crowd-segment.herokuapp.com/{}".format(img_name)
			questionform = ExternalQuestion(url, frame_height)
			create_hit_result = connection.create_hit(
				title="Segment the object on an image",
				description="We'll give you an image with a pointer to an object. You have to draw a bounding region around the boundary of the object in the image. There is 1 object per HIT. Our interface supports keyboard input for speed!",
				keywords=["segmentation", "perception", "image", "fast"],
				duration = 1800,
				max_assignments=20,
				question=questionform,
				reward=Price(amount=0.05),
				lifetime=43200)#,
				#qualifications=qualifications)
			hit_id = str(create_hit_result[0].HITId)
			f.write(hit_id + "\n")
			print "Created HIT: ",hit_id

