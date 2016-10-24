from sqlalchemy import *
from migrate import *


from migrate.changeset import schema
pre_meta = MetaData()
post_meta = MetaData()
HIT = Table('HIT', post_meta,
    Column('id', Integer, primary_key=True, nullable=False),
    Column('assignment_id', Integer),
    Column('hit_id', Integer),
    Column('object_id', Integer),
    Column('worker_id', Integer),
    Column('image_id', Integer),
    Column('times', String(length=150)),
    Column('actions', String(length=150)),
)


def upgrade(migrate_engine):
    # Upgrade operations go here. Don't create your own engine; bind
    # migrate_engine to your metadata
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    post_meta.tables['HIT'].create()


def downgrade(migrate_engine):
    # Operations to reverse the above upgrade go here.
    pre_meta.bind = migrate_engine
    post_meta.bind = migrate_engine
    post_meta.tables['HIT'].drop()
