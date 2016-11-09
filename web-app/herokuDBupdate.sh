heroku pg:backups capture --app crowd-segment
curl -o latest.dump `heroku pg:backups public-url --app crowd-segment`
pg_restore -l latest.dump >latest.db
dropdb segdump
createdb segdump
psql segdump < latest.db
