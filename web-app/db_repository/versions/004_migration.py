from sqlalchemy import *
from migrate import *


from migrate.changeset import schema
pre_meta = MetaData()
post_meta = MetaData()
HIT = Table('HIT', pre_meta,
    Column('id', INTEGER, primary_key=True, nullable=False),
    Column('assignment_id', INTEGER),
    Column('hit_id', INTEGER),
    Column('object_id', INTEGER),
    Column('worker_id', INTEGER),
    Column('image_id', INTEGER),
    Column('times', VARCHAR(length=5000)),
    Column('actions', VARCHAR(length=5000)),
)

hit = Table('hit', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('assignment_id', Integer),
    Column('hit_id', Integer),
    Column('object_id', Integer),
    Column('worker_id', Integer),
    Column('image_id', Integer),
    Column('times', String(length=5000)),
    Column('actions', String(length=5000)),
)


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    pre_meta.tables['HIT'].drop()
    post_meta.tables['hit'].create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    pre_meta.tables['HIT'].create()
    post_meta.tables['hit'].drop()
