from flask import render_template, flash, redirect, request, make_response, send_from_directory, url_for
from app import app, db, models
import json
import os


DEV_ENVIROMENT_BOOLEAN = True
#This allows us to specify whether we are pushing to the sandbox or live site.
if DEV_ENVIROMENT_BOOLEAN:
	AMAZON_HOST = "https://workersandbox.mturk.com/mturk/externalSubmit"
else:
	AMAZON_HOST = "https://www.mturk.com/mturk/externalSubmit"


@app.route('/<img>', methods=['GET', 'POST'])
def segment(img):
	render_data = {"worker_id": request.args.get("workerId"), 
					"assignment_id": request.args.get("assignmentId"), 
					"amazon_host": AMAZON_HOST, 
					"hit_id": request.args.get("hitId")}

	filename = 'static/' + img + '.png'

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
		resp = make_response(render_template('page.html',name=render_data,filename=filename,ht=384,wd=512,accepted=False,objects=objects,locs=object_locations,img=img))
	else:
		#Our worker accepted the task
		resp = make_response(render_template('page.html',name=render_data,filename=filename,ht=384,wd=512,accepted=True,objects=objects,locs=object_locations,img=img))
	resp.headers['x-frame-options'] = 'this_can_be_anything'
	return resp

@app.route('/identify/<img>', methods=['GET', 'POST'])
def identify(img):
	render_data = {"worker_id": request.args.get("workerId"), 
					"assignment_id": request.args.get("assignmentId"), 
					"amazon_host": AMAZON_HOST, 
					"hit_id": request.args.get("hitId")}

	if request.method == 'POST':
		x_locs = json.loads(request.form['x-locs'])
		y_locs = json.loads(request.form['y-locs'])
		# img = json.loads(request.form['image-id'])
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
		# url = url_for('new_submit',x_locs=x_locs,y_locs=y_locs,img=img,object_names=object_names,comment=comment)
		# url = request.url.replace('http://', 'https://', 1)
		# return redirect(url,code=307)

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
	# resp.headers['x-frame-options'] = 'this_can_be_anything'
	# return resp

@app.route('/identify/submit', methods=['GET','POST'])
def submit():
	render_data = {"worker_id": request.args.get("workerId"), 
					"assignment_id": request.args.get("assignmentId"), 
					"amazon_host": AMAZON_HOST, 
					"hit_id": request.args.get("hitId")}

	x_locs = json.loads(request.form['x-locs'])
	y_locs = json.loads(request.form['y-locs'])
	img = json.loads(request.form['image-id'])
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

	resp = make_response(render_template('submit.html',name=render_data,x_locs=x_locs,y_locs=y_locs,img=img,object_names=object_names,comment=comment))
	resp.headers['x-frame-options'] = 'this_can_be_anything'
	return resp

@app.route('/identify/submit', methods=['GET'])
def new_submit(x_locs,y_locs,img,object_names,comment):
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
	render_data = {"worker_id": request.args.get("workerId"), 
					"assignment_id": request.args.get("assignmentId"), 
					"amazon_host": AMAZON_HOST, 
					"hit_id": request.args.get("hitId")}

	x_locs = json.loads(request.form['x-locs'])
	y_locs = json.loads(request.form['y-locs'])
	img = json.loads(request.form['image-id'])
	# object_names = json.loads(request.form['obj-names'])
	comment = json.loads(request.form['comment-input'])
	# #Store all the collected data in the database
	# assume that the worker and object id is given for each task 
	worker_id =123
	print "workerid",request.args.get("workerId")
	# worker_id = models.Worker.query.filter_by(turker=request.args.get("workerId")).first().id
	# print "Worker id", worker_id
	# image_id = models.Image.query.filter_by(filename=img).first().id
	# print worker_id
	# print image_id
	# print "xlocs type:", type(x_locs) 
	# print "image : ", img
	# print "image id : ", image_id
	obj_id = 5
	bounding_box= models.BoundingBox(object_id=obj_id,worker_id=worker_id,x_locs=str(x_locs),y_locs=str(y_locs))
	db.session.add(bounding_box)
	db.session.commit()

	resp = make_response(render_template('submit_segmentation.html',name=render_data,x_locs=x_locs,y_locs=y_locs,comment=comment)) #img=img,
	resp.headers['x-frame-options'] = 'this_can_be_anything'
	return resp
