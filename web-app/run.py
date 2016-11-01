#!flask/bin/python
from app import app
app.run(debug=True)
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
