
library(tidyverse)
library(rpart)
library(rpart.plot)
library(RColorBrewer)

con = dbConnect(pg, user="kthomas1", password="",
                host="localhost", port=5432, dbname="nfldb")

drives <- dbGetQuery(con, "with plays as
	(select gsis_id, drive_id, sum(passing_first_down) as pass_first, sum(penalty_first_down) as penalty_first, sum(rushing_first_down) as rushing_first, sum(first_down) as first, sum(third_down_conv) as third
                     from play
                     group by gsis_id, drive_id),
                     drives as
                     (select gsis_id, drive_id, pos_team, result, start_field, pos_time, first_downs, penalty_yards, yards_gained, play_count
                     from drive)
                     select * from drives
                     inner join plays on (drives.gsis_id=plays.gsis_id and drives.drive_id=plays.drive_id)")

x <- drives %>%
    subset(., select=which(!duplicated(names(.)))) %>%
    mutate(pos_time = as.integer(gsub("[()]", "", pos_time)),
           start_field = as.integer(gsub("[()]", "", start_field)),
           td = ifelse(result=="Touchdown",1,0)) %>%
    select(-gsis_id,-drive_id,-pos_team,-result,-yards_gained)



fit <- rpart(td ~ ., data=x)

rpart.plot(fit)
