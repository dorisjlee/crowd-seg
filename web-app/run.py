#!flask/bin/python
from app import app
import os
port = int(os.environ.get("PORT", 5000))
app.run(host='0.0.0.0', port=port, debug=True)
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
