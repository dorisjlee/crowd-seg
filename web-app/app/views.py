from flask import render_template, flash, redirect, request, make_response, send_from_directory, url_for
from app import app, db, models
import json
from random import randint
LOCAL_TESTING = False
DEV_ENVIROMENT_BOOLEAN = True
#This allows us to specify whether we are pushing to the sandbox or live site.
if DEV_ENVIROMENT_BOOLEAN:
	AMAZON_HOST = "https://workersandbox.mturk.com/mturk/externalSubmit"
else:
	AMAZON_HOST = "https://www.mturk.com/mturk/externalSubmit"


@app.route('/<img>', methods=['GET', 'POST'])
def segment(img):
	print "segment"
	render_data = {"worker_id": request.args.get("workerId"),
					"assignment_id": request.args.get("assignmentId"),
					"amazon_host": AMAZON_HOST,
					"hit_id": request.args.get("hitId")}
	print render_data
	filename = 'static/' + img + '.png'

	#Read objects that have already been identified from the database
	worker = models.Worker.query.filter_by(turker=request.args.get("workerId")).first()
	if worker is None and render_data["worker_id"] != 'None':
		db.session.add(models.Worker(turker=request.args.get("workerId")))
		db.session.commit()
	# print "img: "+img
	image = models.Image.query.filter_by(filename=img).first()
	if image is None:
		db.session.add(models.Image(filename=img))
		db.session.commit()

	image_id = models.Image.query.filter_by(filename=img).first().id

	objects = models.Object.query.filter_by(image_id=image_id).order_by(models.Object.name).all()
	#Read object locations for these objects
	object_locations = models.ObjectLocation.query.filter((models.ObjectLocation.object_id.in_([x.id for x in objects]))).all()

	#Make this data easy to use
	objects = {x.id:x.name for x in objects}
	object_locations = {x.object_id:(x.x_loc,x.y_loc) for x in object_locations} #ASSUMES THAT EACH OBJECT IS MARKED BY EXACTLY 1 WORKER

	#The following code segment can be used to check if the turker has accepted the task yet
	if request.args.get("assignmentId") == "ASSIGNMENT_ID_NOT_AVAILABLE":
		#Our worker hasn't accepted the HIT (task) yet
		resp = make_response(render_template('page.html',name=render_data,filename=filename,ht=384,wd=512,accepted=False,objects=objects,locs=object_locations,img=img))
	else:
		#Our worker accepted the task
		resp = make_response(render_template('page.html',name=render_data,filename=filename,ht=384,wd=512,accepted=True,objects=objects,locs=object_locations,img=img))
	resp.headers['x-frame-options'] = 'this_can_be_anything'
	return resp

@app.route('/identify/<img>', methods=['GET', 'POST'])
def identify(img):
	print "identify"
	render_data = {"worker_id": request.args.get("workerId"),
					"assignment_id": request.args.get("assignmentId"),
					"amazon_host": AMAZON_HOST,
					"hit_id": request.args.get("hitId")}
	print render_data
	if request.method == 'POST':
		x_locs = json.loads(request.form['x-locs'])
		y_locs = json.loads(request.form['y-locs'])
		object_names = json.loads(request.form['obj-names'])
		comment = json.loads(request.form['comment-input'])

		#Store all the collected data in the database
		worker_id = models.Worker.query.filter_by(turker=request.args.get("workerId")).first().id
		image_id = models.Image.query.filter_by(filename=img).first().id
		for name,x,y in zip(object_names,x_locs,y_locs):
			obj = models.Object(image_id=image_id,name=name)
			db.session.add(obj)
			db.session.commit()
			obj_location = models.ObjectLocation(object_id=obj.id,worker_id=worker_id,x_loc=x,y_loc=y)
			db.session.add(obj_location)
			db.session.commit()

		resp = make_response(render_template('submit.html',name=render_data))
		resp.headers['x-frame-options'] = 'this_can_be_anything'
		return resp
	filename = '../static/' + img + '.png'
	#Read objects that have already been identified from the database
	worker = models.Worker.query.filter_by(turker=request.args.get("workerId")).first()
	if worker is None and render_data["worker_id"] != 'None':
		db.session.add(models.Worker(turker=request.args.get("workerId")))
		db.session.commit()
	image = models.Image.query.filter_by(filename=img).first()
	if image is None:
		db.session.add(models.Image(filename=img))
		db.session.commit()

	image_id = models.Image.query.filter_by(filename=img).first().id

	objects = models.Object.query.filter_by(image_id=image_id).order_by(models.Object.name).all()
	#Read object locations for these objects
	object_locations = models.ObjectLocation.query.filter((models.ObjectLocation.object_id.in_([x.id for x in objects]))).all()

	#Make this data easy to use
	objects = {x.id:x.name for x in objects}
	object_locations = {x.object_id:(x.x_loc,x.y_loc) for x in object_locations} #ASSUMES THAT EACH OBJECT IS MARKED BY EXACTLY 1 WORKER

	#The following code segment can be used to check if the turker has accepted the task yet
	if request.args.get("assignmentId") == "ASSIGNMENT_ID_NOT_AVAILABLE":
		#Our worker hasn't accepted the HIT (task) yet
		# resp = make_response(render_template('identify.html',name=render_data,filename=filename,ht=384,wd=512,accepted=False,objects=objects,locs=object_locations,img=img))
		return render_template('identify.html',name=render_data,filename=filename,ht=384,wd=512,accepted=False,objects=objects,locs=object_locations,img=img)
	else:
		#Our worker accepted the task
		# resp = make_response(render_template('identify.html',name=render_data,filename=filename,ht=384,wd=512,accepted=True,objects=objects,locs=object_locations,img=img))
		return render_template('identify.html',name=render_data,filename=filename,ht=384,wd=512,accepted=True,objects=objects,locs=object_locations,img=img)

@app.route('/identify/submit', methods=['GET','POST'])
def submit():
	print "submit"
	render_data = {"worker_id": request.args.get("workerId"),
					"assignment_id": request.args.get("assignmentId"),
					"amazon_host": AMAZON_HOST,
					"hit_id": request.args.get("hitId")}
	print render_data
	x_locs = json.loads(request.form['x-locs'])
	y_locs = json.loads(request.form['y-locs'])
	img = json.loads(request.form['image-id'])
	object_names = json.loads(request.form['obj-names'])
	comment = json.loads(request.form['comment-input'])
	times = json.loads(request.form['times'])
	actions = json.loads(request.form['actions'])

	#Store all the collected data in the database
	worker_id = models.Worker.query.filter_by(turker=request.args.get("workerId")).first().id
	image_id = models.Image.query.filter_by(filename=img).first().id

	for name,x,y in zip(object_names,x_locs,y_locs):
		obj = models.Object(image_id=image_id,name=name)
		db.session.add(obj)
		db.session.commit()
		obj_location = models.ObjectLocation(object_id=obj.id,worker_id=worker_id,x_loc=x,y_loc=y)
		db.session.add(obj_location)
		db.session.commit()
 	#for debugging purposes use random worker_id to ensure no NULL or UNIQUE violation
	if ( LOCAL_TESTING or (request.args.get("workerId") is None) ):
		# worker_id =randint(100, 999) 	  #request.args.get("workerId")
		assignment_id = randint(100, 999)
		hit_id = randint(100, 999)
	else:
		assignment_id = request.args.get("assignmentId")
		hit_id =  request.args.get("hitId")
	hit = models.Hit(assignment_id=assignment_id,hit_id=hit_id,object_id=obj.id,worker_id=worker_id,image_id=image_id,times=str(times),actions=str(actions))
	db.session.add(hit)
	db.session.commit()
	resp = make_response(render_template('submit.html',name=render_data,x_locs=x_locs,y_locs=y_locs,img=img,object_names=object_names,comment=comment))
	resp.headers['x-frame-options'] = 'this_can_be_anything'
	return resp

@app.route('/identify/submit', methods=['GET'])
def new_submit(x_locs,y_locs,img,object_names,comment):
	print "new_submit"
	render_data = {"worker_id": request.args.get("workerId"),
					"assignment_id": request.args.get("assignmentId"),
					"amazon_host": AMAZON_HOST,
					"hit_id": request.args.get("hitId")}
	resp = make_response(render_template('submit.html',name=render_data,x_locs=x_locs,y_locs=y_locs,img=img,object_names=object_names,comment=comment))
	resp.headers['x-frame-options'] = 'this_can_be_anything'
	return resp

@app.route('/<filename>')
def send_file(filename):
	return send_from_directory('static/', filename)

@app.route('/segmentation/submit', methods=['GET','POST'])
def segmentation_submit():
	print "segmentation_submit"
	print request.form
	x_locs = json.loads(request.form['x-locs'])
	y_locs = json.loads(request.form['y-locs'])
	object_id = json.loads(request.form['object_id'])
	comment = json.loads(request.form['comment-input'])
 	times = json.loads(request.form['times'])
	actions = json.loads(request.form['actions'])
	img = json.loads(request.form['image-id'])
	print x_locs,object_id,img
	print request.form['workerId']
	worker = request.form['workerId']
	print "worker:",worker
	print "after worker"
	if (LOCAL_TESTING or worker is None):
		#for debugging purposes use random worker_id to ensure no NULL or UNIQUE violation
		# worker_id =randint(100, 999)
		assignment_id = randint(100, 999)
		hit_id = randint(100, 999)
	else:
		print "here"
		try:
			worker_id = models.Worker.query.filter_by(turker=worker).first().id
		except:
			db.session.add(models.Worker(turker=worker))
			db.session.commit()
			worker_id = models.Worker.query.filter_by(turker=worker).first().id
		print worker_id
		print request.form['assignmentId']
		print request.form['hitId']
		assignment_id = request.form['assignmentId']
		hit_id =  request.form['hitId']
		print worker_id,assignment_id,hit_id

	image_id = models.Image.query.filter_by(filename=img).first().id
	print image_id
	bounding_box= models.BoundingBox(object_id=object_id,worker_id=worker_id,x_locs=str(x_locs),y_locs=str(y_locs))
	print "assignmentId: ",assignment_id
	print "hitId: ", hit_id
	hit = models.Hit(assignment_id=assignment_id,hit_id=hit_id,object_id=object_id,worker_id=worker_id,image_id=image_id,times=str(times),actions=str(actions))
	print "after HIT"
	print bounding_box
	db.session.add(bounding_box)
	print "BB:",bounding_box
	db.session.add(hit)
	print "hit:",hit
	db.session.commit()
	print "DB committed"
	resp = make_response(render_template('submit_segmentation.html',name=render_data,x_locs=x_locs,y_locs=y_locs,img=img,comment=comment)) #img=img,
	print "made response"
	resp.headers['x-frame-options'] = 'this_can_be_anything'
	return resp
