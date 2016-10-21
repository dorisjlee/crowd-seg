from sqlalchemy import *
from migrate import *


from migrate.changeset import schema
pre_meta = MetaData()
post_meta = MetaData()
bounding_box = Table('bounding_box', post_meta,
    Column('object_id', Integer, primary_key=True, nullable=False),
    Column('worker_id', Integer, primary_key=True, nullable=False),
    Column('x_locs', String),
    Column('y_locs', String),
)


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    post_meta.tables['bounding_box'].create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    post_meta.tables['bounding_box'].drop()
