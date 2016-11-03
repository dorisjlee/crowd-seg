from app import app
import logging
app.run(debug=True)
app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)
# app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
