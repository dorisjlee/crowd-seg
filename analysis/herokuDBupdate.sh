# heroku pg:backups capture --app crowd-segment
# curl -o latest.dump `heroku pg:backups public-url --app crowd-segment`
# pg_restore -l latest.dump >latest.db
#dropdb crowd-segment
#createdb crowd-segment
#psql crowd-segment < latest.db
echo "Connecting to DB from Heroku"
heroku config:get DATABASE_URL
pg_dump -h ec2-54-235-221-102.compute-1.amazonaws.com -p 5432  --username=ywnbumggutyzsd  --dbname=dfn25t8hp2tpld --password>latest.db
# psql -d crowd-segment -f latest.db
dropdb crowd-segment
createdb crowd-segment
psql crowd-segment < latest.db
