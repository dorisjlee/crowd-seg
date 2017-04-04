#!flask/bin/python
import time
from analysis_toolbox import load_info
from glob import glob
from boto.mturk.connection import MTurkConnection
from boto.mturk.question import ExternalQuestion
from boto.mturk.qualification import Qualifications, PercentAssignmentsApprovedRequirement, NumberHitsApprovedRequirement #, Requirement
from boto.mturk.price import Price
from secret import SECRET_KEY,ACCESS_KEY,AMAZON_HOST
import os
import pandas as pd
#Start Configuration Variables
AWS_ACCESS_KEY_ID = ACCESS_KEY
AWS_SECRET_ACCESS_KEY = SECRET_KEY

connection = MTurkConnection(aws_access_key_id=AWS_ACCESS_KEY_ID,
							 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
							 host=AMAZON_HOST)

HIT_TYPE = "SEGMENT"
#HIT_TYPE="IDENTIFY"
print "Connected."
#frame_height in pixels
frame_height = 800


#Here, I create two sample qualifications
qualifications = Qualifications()
#qualifications.add(Requirement(MastersQualID,'DoesNotExist',required_to_preview=True))
qualifications.add(PercentAssignmentsApprovedRequirement(comparator="GreaterThan", integer_value="95"))
qualifications.add(NumberHitsApprovedRequirement(comparator="GreaterThan", integer_value="500"))

#Join object and image tables
img_info,object_tbl,bb_info,hit_info  = load_info()
img_obj_tbl = object_tbl.merge(img_info,how="inner",left_on="image_id",right_on="id")

os.chdir("../web-app/app")
#This url will be the url of your application, with appropriate GET parameters
with open('ActiveHITs','a') as f:
	f.write('New batch created on : '+time.ctime())
	for fname in glob("static/COCO_*.png"):
        # fname = "static/COCO_train2014_000000000127.png"
		img_name = fname.split('/')[-1].split('.')[0]
		print img_name
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
				reward=Price(amount=0.05),
				lifetime=43200)#,
				#qualifications=qualifications)
			hit_id = str(create_hit_result[0].HITId)
			f.write(hit_id + "\n")
			print "Created HIT: ",hit_id
		elif HIT_TYPE == "SEGMENT":
			# numObj = len(img_obj_tbl[img_obj_tbl.filename==img_name])
			# print numObj, "obj in image :",img_name
			# print "max_assignments:", 30*numObj
			# objId_lst = [10,12]
			objId_lst = list(img_obj_tbl[img_obj_tbl.filename==img_name].object_id)
			print os.getcwd()
			BB_count_info = pd.read_csv("../../../data/BB_count_tbl.csv")
			for objId in objId_lst:
			# for _i in range(20*numObj):
				print objId
				url = "https://crowd-segment.herokuapp.com/segment/{0}/{1}/".format(img_name,objId)
				maxAssignment = 41-int(BB_count_info[BB_count_info.id ==objId]["BB_count"])
				print maxAssignment
				if maxAssignment>0:
					questionform = ExternalQuestion(url, frame_height)
					create_hit_result = connection.create_hit(
						title="Segment the object on an image",
						description="We'll give you an image with a pointer to an object. You have to draw a bounding region around the boundary of the object in the image. There is 1 object per HIT. Our interface supports keyboard input for speed!",
						keywords=["segmentation", "perception", "image", "fast"],
						duration = 1800,
						max_assignments=maxAssignment,
						question=questionform,
						reward=Price(amount=0.05),
						lifetime=43200)#,
						#qualifications=qualifications)
					hit_id = str(create_hit_result[0].HITId)
					f.write(hit_id + "\n")
					print "Created HIT for img:{0}, objId:{1}: {2}".format(img_name,objId,hit_id)
