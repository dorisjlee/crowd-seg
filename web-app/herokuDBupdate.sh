heroku pg:backups capture --app crowd-segment
curl -o latest.dump `heroku pg:backups public-url`
pg_restore -l latest.dump
createdb segdump
psql segdump < latest.db
