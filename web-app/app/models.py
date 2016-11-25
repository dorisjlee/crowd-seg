from app import db

class ObjectLocation(db.Model):
	object_id = db.Column(db.Integer, db.ForeignKey('object.id'),primary_key = True)
	worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'),primary_key = True)
	x_loc = db.Column(db.Float, index=True)
	y_loc = db.Column(db.Float, index=True)

	def __repr__(self):
		return '<obj=%d,worker=%d,x=%1.3f,y=%1.3f>' % (self.object_id,self.worker_id,self.x_loc,self.y_loc)

class Object(db.Model):
	id = db.Column(db.Integer, primary_key = True)
	image_id = db.Column(db.Integer, db.ForeignKey('image.id'))
	name = db.Column(db.String(64), index=True)

	def __repr__(self):
		return '<id=%d,image=%d,obj-name=%s>' % (self.id,self.image_id,self.name)

class Image(db.Model):
	id = db.Column(db.Integer, primary_key = True)
	filename = db.Column(db.String(150), unique = True)

	def __repr__(self):
		return '<id=%d,img-name=%s>' % (self.id,self.filename)
class Hit(db.Model):
	#Each identify or segment session, for identify the object_id is always 999
	id = db.Column(db.Integer, primary_key = True)
	assignment_id = db.Column(db.String(50))
	hit_id = db.Column(db.String(50))
	object_id = db.Column(db.Integer, db.ForeignKey('object.id'))
	worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'))
	image_id = db.Column(db.Integer, db.ForeignKey('image.id'))
	times = db.Column(db.String(5000))
	actions = db.Column(db.String(5000))
	def __repr__(self):
		return '<assignment=%s,hit=%s, object_id=%d,worker=%d,img-id=%d,\n actions=%s,times=%s>' % (self.assignment_id,self.hit_id,self.object_id,self.worker_id,self.image_id,self.actions,self.times)

class Worker(db.Model):
	id = db.Column(db.Integer, primary_key = True)
	turker = db.Column(db.String(50), unique=True)

	def __repr__(self):
		return '<id=%d,turker-id=%s>' % (self.id,self.turker)
class BoundingBox(db.Model):
	'''
	the x and y locations are a string that is formatted as [x1,x2,x3 ..etc]
	'''
	object_id = db.Column(db.Integer, db.ForeignKey('object.id'),primary_key = True)
	worker_id = db.Column(db.Integer, db.ForeignKey('worker.id'),primary_key = True)
	x_locs = db.Column(db.String, index=True)
	y_locs = db.Column(db.String, index=True)

	def __repr__(self):
		return '<obj=%d,worker=%d,x=%s,y=%s>' % (self.object_id,self.worker_id,self.x_locs,self.y_locs)
