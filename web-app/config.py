CSRF_ENABLED = True
WTF_CSRF_ENABLED = True
import os
basedir = os.path.abspath(os.path.dirname(__file__))

# SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'app.db')
# SQLALCHEMY_DATABASE_URI = 'postgresql://localhost/segment'
SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL","postgresql://localhost/segment")
SQLALCHEMY_MIGRATE_REPO = os.path.join(basedir, 'db_repository')

