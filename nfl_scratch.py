import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import psycopg2
import plotly as py
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
conn=psycopg2.connect("dbname='nfldb' user='kthomas1' host='localhost' password='' port=5432")
cur=conn.cursor()


cur.execute("""SELECT college, count(*) as freq 
from player 
where college is not null 
group by college 
order by freq DESC 
limit 10""")
rows = cur.fetchall()
t2=[("School","Count")] + rows
table = ff.create_table(t2)
iplot(table, filename='simple_table')


cur.execute("""select season_year, week, home_team, home_score, away_team, away_score
from game
where season_type='Regular'""")

rows = cur.fetchall()

game_results=pd.DataFrame(rows,columns=['year','week','home_team','home_score','away_team','away_score'])

#replace la with stl
game_results.replace(to_replace='LA', value='STL', inplace=True)


game_results['home_win'] = [1 if x>y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]
game_results['away_win'] = [1 if x<y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]
game_results['tie'] = [1 if x==y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]

game_results=game_results.sort(columns=['year','home_team','week'])

w2017 = pd.read_csv('~/desktop/2017_w1.csv')

game_results = game_results.append(w2017)







cur.execute("""select drive.pos_team, drive.drive_id, drive.pos_time, drive.first_downs, drive.yards_gained, drive.play_count, drive.result, game.season_year, game.week, game.season_type, game.home_team, game.away_team
from drive
inner join game on drive.gsis_id=game.gsis_id
where season_type='Regular'""")

stats = cur.fetchall()

stats=pd.DataFrame(stats,columns=['pos_team','drive_id','pos_time','first_downs','yards_gained','play_count','result','season_year','week','season_type','home_team','away_team'])

#replace la with stl
stats.replace(to_replace='LA', value='STL', inplace=True)

# points
stats['points'] = [3 if x=="Field Goal" else 7 if x=="Touchdown" else 0 for x in stats['result']]

# turnover
stats['turnover'] = [1 if x==("Interception" or "Fumble" or "Safety" or "Blocked FG" or "Fumble, Safety" or "Blocked Punt" or "Blocked Punt, Downs" or "Blocked FG, Downs") else 0 for x in stats['result']]


# defense
stats_d = stats
stats_d['opp_team'] = np.where(stats_d['pos_team']==stats_d['home_team'], stats_d['away_team'], stats_d['home_team'])
#subset to defensive stats
stats_d = stats_d[['season_year','week','opp_team','yards_gained','points','turnover']]
# rename columns
stats_d.columns = ['year','week','team','yards_allowed','points_allowed','turnovers_forced']

stats_d_2017 = pd.read_csv('~/desktop/nfl_2017_stats_d.csv')
stats_d=stats_d.append(stats_d_2017)

# look at the numbers
# table = pd.pivot_table(stats_d, values=['yards_allowed','points_allowed','turnovers_forced'], index=['year', 'team'], aggfunc=np.sum)
# table.sort_values(by='turnovers_forced',ascending=False)

# aggregate rolling 5 week
## sort at year, team, week
stats_d.sort_values(by=['team','year','week'],inplace=True)
## sum across year team week
stats_d=stats_d.groupby(by=['team','year','week'],as_index=False).sum()
## rolling 2 week lagged
rolling = stats_d.groupby(['team'],as_index=False)['yards_allowed','points_allowed','turnovers_forced'].rolling(2).sum().shift(1).reset_index()
## join together
stats_d=stats_d.join(rolling,lsuffix='_weekly',rsuffix='_rolling')


# offense
stats_o = stats
stats_o=stats_o.rename(columns = {'pos_team':'team'})
stats_o=stats_o.rename(columns = {'season_year':'year'})
stats_o = stats_o[['team','year','week','first_downs','yards_gained','play_count','points','turnover']]

stats_o_2017 = pd.read_csv('~/desktop/nfl_2017_stats_o.csv')
stats_o=stats_o.append(stats_o_2017)

# aggregate rolling 5 week
## sort at year, team, week
stats_o.sort_values(by=['team','year','week'],inplace=True)
## sum across year team week
stats_o=stats_o.groupby(by=['team','year','week'],as_index=False).sum()
## rolling 5 week lagged
rolling = stats_o.groupby(['team'],as_index=False)['first_downs','yards_gained','play_count','points','turnover'].rolling(2).sum().shift(1).reset_index()
## join together
stats_o=stats_o.join(rolling,lsuffix='_weekly',rsuffix='_rolling')


## combine offense and defense
stats_o = stats_o.drop(['level_0','level_1'], axis=1)
stats_d = stats_d.drop(['level_0','level_1'], axis=1)
stats_od=pd.concat([stats_d,stats_o],axis=1)
stats_od=stats_od.T.drop_duplicates().T


#simplify dataset
stats_od=stats_od[stats_od['year']!=2009]
weekly_stats = [col for col in stats_od if col.endswith('weekly')]
stats_od = stats_od.drop(weekly_stats, axis=1)

# convert to numeric
stats_od=stats_od.apply(pd.to_numeric, errors='ignore')


#reorganize so that we are predicting home team victory
#home team --> home team stats
#away team --> away team stats
#training columns ---> diff between home and away stats

games = game_results

stats_od.columns=['team','year','week','ya','pa','tf','fd','yg','pc','p','t']

games=pd.merge(pd.merge(games,stats_od,left_on=['home_team','year','week'],right_on=['team','year','week']),stats_od,left_on=['away_team','year','week'],right_on=['team','year','week'],suffixes=['_home','_away'])



diffs=['ya','pa','tf','fd','yg','pc','p','t']

for i in diffs:
    diff_column = i + "_diff"
    home_column = i + "_home"
    away_column = i + "_away"
    games[diff_column] = games[home_column] - games[away_column]



home = [col for col in games if col.endswith('home')]
away = [col for col in games if col.endswith('away')]

games_w = games.drop(home,axis=1)
games_w = games_w.drop(away,axis=1)






games_16 = games_w[games_w['year']<2017] 
games_17 = games_w[games_w['year']==2017]

training_cols = [col for col in games_w if col.endswith('diff')]

logit = LogisticRegression(fit_intercept=True)
logit.fit(games_16[training_cols],games_16['home_win'])

preds=logit.predict(games_17[training_cols])

games_17['prediction'] = preds

games_17.to_csv('games_17.csv')











#training cols
training_cols = [col for col in games_w if col.endswith('diff')]

#fit model
logit = sm.Logit(games_w['home_win'], games_w[training_cols])

result = logit.fit()

print(result.summary())
print(np.exp(result.params))

#---------------------------------------



from sklearn import metrics, cross_validation
logreg=LogisticRegression()
predicted = cross_validation.cross_val_predict(logreg, games_w[training_cols], games_w['home_win'], cv=10)
print(metrics.accuracy_score(games_w['home_win'], predicted))
print(metrics.classification_report(games_w['home_win'], predicted))



#---------------------------------------

#training cols
training_cols = [col for col in games_w if col.endswith('diff')]

games_15 = games_w[games_w['year']<2016] 
games_16 = games_w[games_w['year']==2016]

#fit model
logit = sm.Logit(games_15['home_win'], games_15[training_cols])
result = logit.fit()

logit.predict(games_16[training_cols])

print(result.summary())
print(np.exp(result.params))

#---------------------------------------

#this works well --> need to figure out how for statsmodel bc it has better tables
logit = LogisticRegression(fit_intercept=True)
logit.fit(games_15[training_cols],games_15['home_win'])

preds=logit.predict(games_16[training_cols])

aa = games_16[['year','week','home_team','away_team','home_win']]
aa['prediction'] = preds

aa.to_csv('aa.csv')

#---------------------------------------
















from sklearn.linear_model import LinearRegression

trainingData = np.array([ [2.3,4.3,2.5], [1.3,5.2,5.2], [3.3,2.9,0.8], [3.1,4.3,4.0]  ])
trainingScores = np.array([3.4,7.5,4.5,1.6])

clf = LinearRegression(fit_intercept=True)
clf.fit(trainingData,trainingScores)

predictionData = np.array([ [2.5,2.4,2.7], [2.7,3.2,1.2] ])
clf.predict(predictionData)






















#training cols
training_cols = [col for col in nfl_w if col.endswith('rolling')]

#fit model
logit = sm.Logit(nfl_w['win'], nfl_w[training_cols])

result = logit.fit()

print(result.summary())
print(np.exp(result.params))



#sklearn version
#http://nbviewer.jupyter.org/gist/justmarkham/6d5c061ca5aee67c4316471f8c2ae976
X=nfl_w[training_cols]
y=nfl_w['win']
model = LogisticRegression()
model = model.fit(X, y)
model.score(X, y)
y.mean()
pd.DataFrame(zip(X.columns, np.transpose(model.coef_)))















nfl.to_csv('nfl.csv')


nfl = pd.merge(stats_od, wins, how='left', on=['team', 'year', 'week'])

nfl = stats_od.merge(wins, how='left', left_on=['team', 'year', 'week'])





zz.to_csv('test.csv')
wins.to_csv('testx.csv')






# net yards

df=stats_d


cumsums = df.groupby(['team', 'year', 'week']).rolling(5).sum().fillna(0).groupby(level=0).cumsum()
df.set_index(['team', 'year', 'week'], inplace=True)
df['a','b','c'] = cumsums
df.reset_index(inplace=True)


cumsums = df.groupby(['team', 'year', 'week']).sum().fillna(0).groupby(level=0).cumsum()
df.set_index(['team', 'year', 'week'], inplace=True)
df['a','b','c'] = cumsums
df.reset_index(inplace=True)









x=x.groupby('team')['yards_allowed','points_allowed'].rolling(5).sum().reset_index(0,drop=True)

x.to_csv('test.csv')





stats_d.set_index(['year','team','week'],inplace=True)



x.loc[x['team'].isin('CAR')]


## rolling 5 week window
stats_d.set_index(['year','team','week'],inplace=True)
stats_d.rolling(5).sum()







## index at year, team, week
stats_d.set_index(['year','team','week'],inplace=True)
## sum at year, team, week
stats_d=pd.pivot_table(stats_d, values=['yards_allowed','points_allowed','turnovers_forced'], index=['year', 'team', 'week'], aggfunc=np.sum)

stats_d.groupby(level=0, group_keys=False).rolling(5).sum()


# offense





no_ties = game_results

filter_col = ["year","week"] + [col for col in no_ties if col.startswith('home')]
home = no_ties[filter_col]
home.columns = ['year','week','team','score','win']

filter_col = ["year","week"] + [col for col in no_ties if col.startswith('away')]
away = no_ties[filter_col]
away.columns = ['year','week','team','score','win']

wins = home
wins = wins.append(away)
wins = wins.sort_values(by=['team','year','week']).reset_index()




import seaborn as sns
sns.set(style="ticks")

# Load the example dataset for Anscombe's quartet
df = sns.load_dataset("anscombe")

# Show the results of a linear regression within each dataset
sns.lmplot(x="x", y="y", col="dataset", hue="dataset", data=df,
           col_wrap=2, ci=None, palette="muted", size=4,
           scatter_kws={"s": 50, "alpha": 1})




















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

# add week 2017 to results --> not yet in nfldb so has to be added manually
w2017 = pd.read_csv('~/desktop/2017_w1.csv')
game_results = game_results.append(w2017)

# print first 10 entries
#game_results.head(10)



#base line is just always guessing home wins --> compare to cross validated accuracy score

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

# encode points results
stats['points'] = [3 if x=="Field Goal" else 7 if x=="Touchdown" else 0 for x in stats['result']]

# encode turnover results
stats['turnover'] = [1 if x==("Interception" or "Fumble" or "Safety" or "Blocked FG" or "Fumble, Safety" or "Blocked Punt" or "Blocked Punt, Downs" or "Blocked FG, Downs") else 0 for x in stats['result']]



# defense
stats_d = stats
stats_d['opp_team'] = np.where(stats_d['pos_team']==stats_d['home_team'], stats_d['away_team'], stats_d['home_team'])
#subset to defensive stats
stats_d = stats_d[['season_year','week','opp_team','yards_gained','points','turnover']]
# rename columns
stats_d.columns = ['year','week','team','yards_allowed','points_allowed','turnovers_forced']

stats_d_2017 = pd.read_csv('~/desktop/nfl_2017_stats_d.csv')
stats_d=stats_d.append(stats_d_2017)

# aggregate rolling 5 week
## sort at year, team, week
stats_d.sort_values(by=['team','year','week'],inplace=True)
## sum across year team week
stats_d=stats_d.groupby(by=['team','year','week'],as_index=False).sum()
## rolling 2 week lagged
rolling = stats_d.groupby(['team'],as_index=False)['yards_allowed','points_allowed','turnovers_forced'].rolling(5).sum().shift(1).reset_index()
## join together
stats_d=stats_d.join(rolling,lsuffix='_weekly',rsuffix='_rolling')





# offense
stats_o = stats
stats_o=stats_o.rename(columns = {'pos_team':'team'})
stats_o=stats_o.rename(columns = {'season_year':'year'})
stats_o = stats_o[['team','year','week','first_downs','yards_gained','play_count','points','turnover']]

stats_o_2017 = pd.read_csv('~/desktop/nfl_2017_stats_o.csv')
stats_o=stats_o.append(stats_o_2017)

# aggregate rolling 5 week
## sort at year, team, week
stats_o.sort_values(by=['team','year','week'],inplace=True)
## sum across year team week
stats_o=stats_o.groupby(by=['team','year','week'],as_index=False).sum()
## rolling 5 week lagged
rolling = stats_o.groupby(['team'],as_index=False)['first_downs','yards_gained','play_count','points','turnover'].rolling(5).sum().shift(1).reset_index()
## join together
stats_o=stats_o.join(rolling,lsuffix='_weekly',rsuffix='_rolling')






## combine offense and defense
stats_o = stats_o.drop(['level_0','level_1'], axis=1)
stats_d = stats_d.drop(['level_0','level_1'], axis=1)
stats_od=pd.concat([stats_d,stats_o],axis=1)
stats_od=stats_od.T.drop_duplicates().T

#simplify dataset
stats_od=stats_od[stats_od['year']!=2009]
weekly_stats = [col for col in stats_od if col.endswith('weekly')]
stats_od = stats_od.drop(weekly_stats, axis=1)

# convert to numeric
stats_od=stats_od.apply(pd.to_numeric, errors='ignore')



games = game_results

stats_od.columns=['team','year','week','ya','pa','tf','fd','yg','pc','p','t']

games=pd.merge(pd.merge(games,stats_od,left_on=['home_team','year','week'],right_on=['team','year','week']),stats_od,left_on=['away_team','year','week'],right_on=['team','year','week'],suffixes=['_home','_away'])

diffs=['ya','pa','tf','fd','yg','pc','p','t']

for i in diffs:
    diff_column = i + "_diff"
    home_column = i + "_home"
    away_column = i + "_away"
    games[diff_column] = games[home_column] - games[away_column]


home = [col for col in games if col.endswith('home')]
away = [col for col in games if col.endswith('away')]

games_w = games.drop(home,axis=1)
games_w = games_w.drop(away,axis=1)




import statsmodels.api as sm

games_16 = games_w[games_w['year']<2017] 
games_17 = games_w[games_w['year']==2017]

games_16['intercept'] = 1.0
games_17['intercept'] = 1.0

training_cols = [col for col in games_w if col.endswith('diff')]
training_cols = training_cols + ["intercept"]

logit = sm.Logit(games_16['home_win'], games_16[training_cols])

result = logit.fit()

print(result.summary())










































