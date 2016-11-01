#!flask/bin/python
from app import app
import os
from config import LOCAL
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port, debug=True)
if not LOCAL : 
	from flask.ext.heroku import Heroku
	heroku = Heroku(app)
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
