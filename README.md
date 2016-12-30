# crowd-seg

Crowsourcing image segmentation project

## Web-app 

The web-app is built with Flask and served on Heroku. The Flask app is inside ``app``:

- ``views.py``: specifies the url redirect actions and data to be stored
- ``models.py``: defines the data models as object classes.
- ``page.html`` and ``identify.html``: HTML templates for  the ``segment`` and ``identify`` web interface.
- ``submit.html`` and ```submit_segmentation.html``: HTML templates for the submission interface for ``segment`` and ``identify``  mode.

To serve the webapp locally, turn on the local [Postgres server](https://www.postgresql.org/) and  run : 

​		``python run.py``

### URL

The URL for the webapp is as follows: 

​	``https://<base-url>/<type>/<img-name>/<object-id>/``

- ``base-url`` : https://crowd-segment.herokuapp.com for production or http://localhost:5000/ for local testing
- ``type``: ``identify``mode is a point-and-label interface that asks the user to label all objects in the image. ``segment``mode asks the user to draw a bounding box around a specified object in the image.
- ``img-name``:  name of main image for the task,  without .png .
- [Optional] ``object-id``: object id of image as stored inside the PSQL database. This keyword is not required in the ``identify`` mode

For example, 

​	``https://localhost:5000/segment/COCO_train2014_000000000307/14/``

​	``https://crowd-segment.herokuapp.com/identify/COCO_train2014_000000000643``

### Production

The script ``db_create.py`` needs to be run only once for the first time, after that if the data models (``models.py``) has been modified, run ``db_migrate.py`` to update the DB schema to reflect the changes.

When launching the web-app into production, set ``DEV_ENVIRONMENT_BOOLEAN`` in ``secret.py`` as False, otherwise, for testing, everything should happen in sandbox mode. The ``secret.py`` looks like this: 

```
ACCESS_KEY = <AWS-ACCESS-KEY-HERE>
SECRET_KEY = <AWS-SECRET-KEY-HERE>
DEV_ENVIROMENT_BOOLEAN = False
#This allows us to specify whether we are pushing to the sandbox or live site.
if DEV_ENVIROMENT_BOOLEAN:
    AMAZON_HOST = 'mechanicalturk.sandbox.amazonaws.com'
else:
    AMAZON_HOST = 'mechanicalturk.amazonaws.com'
```

To push any changes onto heroku, commit the changes to the git repo. Since the Flask files need to be at the root node in order for Heroku to detect the webapp, push only the web/ folder subtree to Heroku: 

​	``git subtree push --prefix web-app heroku master``

### Postgres Databases

Since Heroku does not allow writing to local filesystem for commiting the transactions into a SQLite database, we use [Heroku's Postgres Service](https://www.heroku.com/postgres).  To pull the most updated DB to analysis run ``herokuDBupdate.sh``, and enter the password which can be read out in the format:  	

​	``postgresql://<host>:<port>/<dbname>?user=<username>&password=<password>``

The database can also be accessed through SQL queries (read-only) through [DataClips](https://dataclips.heroku.com/clips/). Another quick way to connect and interact with database is through Postico and this [connector plugin](https://www.npmjs.com/package/heroku-postico).

## Analysis

The scripts in ``analysis`` are for analyzing worker annotations and managing tasks on AMT. Most of the test code are written as Jupyter notebooks. The working versions of frequently used functions are then compiled into helper modules.

- Analysis modules:
  - ``analysis_toolbox.py``: DB access, data processing, cleaning, visualization functions. Statistics helpers for basic statistics, KS test, fitting and plotting. 
  - ``qualityBaseline.py``: basic functions for processing and evaluating quality of bounding box against ground truth.
- HIT Management:
  - ``post_to_mturk.py``: Allocate task on MTurk according to how many task were already recorded.
  - ``postprocess.py``: approve HITS.
  - ````expire_all_hit.py````
  - ``listAllReviewableHITS.py``



