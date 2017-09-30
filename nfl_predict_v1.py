#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 10:28:01 2017

@author: kthomas1
"""

next_week = 3

import warnings
warnings.filterwarnings('ignore')

# import pandas and numpy for 
import pandas as pd
import numpy as np

# connect to PostgreSQL
import psycopg2
conn=psycopg2.connect("dbname='nfldb' user='kthomas1' host='localhost' password='' port=5432")

# query game results
game_results=pd.read_sql("""select season_year, week, home_team, home_score, away_team, away_score
from game
where season_type='Regular'""",con=conn)

# replace la with stl
game_results.replace(to_replace='LA', value='STL', inplace=True)

# compute wins and ties
game_results['home_win'] = [1 if x>y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]
game_results['away_win'] = [1 if x<y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]
game_results['tie'] = [1 if x==y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]

# sort the dataframe
game_results=game_results.sort_values(by=['season_year','home_team','week'])

# rename the year
game_results=game_results.rename(columns = {'season_year':'year'})

# print first 10 entries
game_results.head(10)

# total number of games
total_games = len(game_results)

# total number of home wins
home_wins = game_results.home_win.sum()

# total home wins/total number of games
home_win_rate = home_wins/total_games

print("Home Team Win Rate: {:.2f}% ".format(home_win_rate*100))

stats=pd.read_sql("""select drive.pos_team, drive.drive_id, drive.pos_time, drive.first_downs, drive.yards_gained, drive.play_count, drive.result, game.season_year, game.week, game.season_type, game.home_team, game.away_team
from drive
inner join game on drive.gsis_id=game.gsis_id
where season_type='Regular'""",con=conn)

#replace la with stl
stats.replace(to_replace='LA', value='STL', inplace=True)
stats.replace(to_replace='LAC', value='SD', inplace=True)

# encode points results
stats['points'] = [3 if x=="Field Goal" else 7 if x=="Touchdown" else 0 for x in stats['result']]

# encode turnover results
stats['turnover'] = [1 if x==("Interception" or "Fumble" or "Safety" or "Blocked FG" or "Fumble, Safety" or "Blocked Punt" or "Blocked Punt, Downs" or "Blocked FG, Downs") else 0 for x in stats['result']]

# look at the table
stats.head(10)

#add next weeks stats
games_17 = game_results[game_results['year']==2017]

nweek = pd.DataFrame({
        'year': np.array([2017] * 32,dtype='int32'),
        'week': np.array([next_week] * 32,dtype='int32'),
        'team': np.array((list(set(stats.pos_team))),dtype=str)})

nweek_d = nweek
nweek_o = nweek

# defense
stats_d = stats
stats_d['opp_team'] = np.where(stats_d['pos_team']==stats_d['home_team'], stats_d['away_team'], stats_d['home_team'])
#subset to defensive stats
stats_d = stats_d[['season_year','week','opp_team','yards_gained','points','turnover']]
# rename columns
stats_d.columns = ['year','week','team','yards_allowed','points_allowed','turnovers_forced']

#add next week
stats_d = stats_d.append(nweek_d)

# aggregate rolling 5 week
## sort at year, team, week
stats_d.sort_values(by=['team','year','week'],inplace=True)
## sum across year team week
stats_d=stats_d.groupby(by=['team','year','week'],as_index=False).sum()
## rolling 2 week lagged
rolling = stats_d.groupby(['team'],as_index=False)['yards_allowed','points_allowed','turnovers_forced'].rolling(5).sum().shift(1).reset_index()
## join together
stats_d=stats_d.join(rolling,lsuffix='_weekly',rsuffix='_rolling')

stats_d.head(10)

# offense
stats_o = stats
stats_o=stats_o.rename(columns = {'pos_team':'team'})
stats_o=stats_o.rename(columns = {'season_year':'year'})
stats_o = stats_o[['team','year','week','first_downs','yards_gained','play_count','points','turnover']]

#add next week
stats_o = stats_o.append(nweek_o)

# aggregate rolling 5 week
## sort at year, team, week
stats_o.sort_values(by=['team','year','week'],inplace=True)
## sum across year team week
stats_o=stats_o.groupby(by=['team','year','week'],as_index=False).sum()
## rolling 5 week lagged
rolling = stats_o.groupby(['team'],as_index=False)['first_downs','yards_gained','play_count','points','turnover'].rolling(5).sum().shift(1).reset_index()
## join together
stats_o=stats_o.join(rolling,lsuffix='_weekly',rsuffix='_rolling')

stats_o.head(10)

# drop the level variables from offense and defensive
stats_o = stats_o.drop(['level_0','level_1'], axis=1)
stats_d = stats_d.drop(['level_0','level_1'], axis=1)

# combine offense and defense stats
stats_od=pd.concat([stats_d,stats_o],axis=1)
stats_od=stats_od.T.drop_duplicates().T

x = pd.merge(stats_d,stats_o,how='inner',on=['team','year','week'])

# drop the year 2009 becasue of the blank weeks
stats_od=stats_od[stats_od['year']!=2009]

# drop the weekly stats because we won't be needing them
weekly_stats = [col for col in stats_od if col.endswith('weekly')]
stats_od = stats_od.drop(weekly_stats, axis=1)

# convert to numeric
stats_od=stats_od.apply(pd.to_numeric, errors='ignore')

# create a new games dataframe from game_results
games = game_results

# rename columns
stats_od.columns=['team','year','week','ya','pa','tf','fd','yg','pc','p','t']

# merge game results with stats; there need to be two merges because both home and away teams need statistics
games=pd.merge(pd.merge(games,stats_od,left_on=['home_team','year','week'],right_on=['team','year','week']),stats_od,left_on=['away_team','year','week'],right_on=['team','year','week'],suffixes=['_home','_away'])

# comptue diffs for each variable
diffs=['ya','pa','tf','fd','yg','pc','p','t']
for i in diffs:
    diff_column = i + "_diff"
    home_column = i + "_home"
    away_column = i + "_away"
    games[diff_column] = games[home_column] - games[away_column]

# we only need the diffs, so drop all the home/away specific stats columns
home = [col for col in games if col.endswith('home')]
away = [col for col in games if col.endswith('away')]
games = games.drop(home,axis=1)
games = games.drop(away,axis=1)

import statsmodels.api as sm

# create past games df that will be used to train our model
past_games = games[(games['year']!=max(games.year)) & (games['week']!=max(games_17.week))]

# create future games df that will be predicted using our trained model
future_games = games[(games['year']==max(games.year)) & (games['week']==next_week)]

# for statsmodels, we need to specify
past_games['intercept'] = 1.0
future_games['intercept'] = 1.0

# our training columns will be the diffs
training_cols = [col for col in games if col.endswith('diff')]
# need to add the intercept column
training_cols = training_cols + ["intercept"]

# perform the regression
logit = sm.Logit(past_games['home_win'], past_games[training_cols])

# save the results and print
result = logit.fit()
print(result.summary())

#log odds
print(np.exp(result.params))

# predict the results 
preds=result.predict(future_games[training_cols])

# add probabilities to next week
future_games['win_prob'] = preds

# home team wins if team has greater than 50% chance of winning
future_games['winner'] = np.where(future_games['win_prob']>.5,future_games['home_team'],future_games['away_team'])

# show select columns
future_games[['home_team','away_team','winner','win_prob']]

# import sklearn
from sklearn.linear_model import LogisticRegression

# define sklearn logit with default intercept
logit = LogisticRegression(fit_intercept=True)

# fit the
logit.fit(past_games[training_cols],past_games['home_win'])

# retrieve and display the probabilities
preds=logit.predict(future_games[training_cols])
future_games['prediction'] = preds
future_games['winner'] = np.where(future_games['prediction']==1,future_games['home_team'],future_games['away_team'])
future_games['win_prob'] = logit.predict_proba(future_games[training_cols])[:,1]
future_games[['home_team','away_team','winner','win_prob']]

from sklearn import preprocessing
from sklearn import metrics, cross_validation

# scale
past_games_scaled = pd.DataFrame(preprocessing.scale(past_games[training_cols]))

# cross validate
scores = cross_validation.cross_val_score(logit, past_games_scaled, past_games['home_win'], cv=10)

# accuracy
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# import cross validated logistic regression
from sklearn.linear_model import LogisticRegressionCV

# define sklearn logit with default intercept
logitcv = LogisticRegressionCV()

# fit the
logitcv.fit(past_games[training_cols],past_games['home_win'])

preds=logitcv.predict(future_games[training_cols])
future_games['prediction'] = preds
future_games['winner'] = np.where(future_games['prediction']==1,future_games['home_team'],future_games['away_team'])
future_games['win_prob'] = logitcv.predict_proba(future_games[training_cols])[:,1]
future_games[['home_team','away_team','winner','win_prob']]