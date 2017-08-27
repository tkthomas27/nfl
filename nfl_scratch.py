import pandas as pd
import numpy as np
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

game_results['home_win'] = [1 if x>y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]
game_results['away_win'] = [1 if x<y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]
game_results['tie'] = [1 if x==y else 0 for x,y in zip(game_results['home_score'],game_results['away_score'])]

game_results=game_results.sort(columns=['year','home_team','week'])

no_ties = game_results.drop(game_results[game_results.tie==1].index)

filter_col = ["year","week"] + [col for col in no_ties if col.startswith('home')]
home = no_ties[filter_col]
home.columns = ['year','week','team','score','win']

filter_col = ["year","week"] + [col for col in no_ties if col.startswith('away')]
away = no_ties[filter_col]
away.columns = ['year','week','team','score','win']

wins = home
wins = wins.append(away)
wins = wins.sort_values(by=['year','team','week'])




cur.execute("""select drive.pos_team, drive.drive_id, drive.pos_time, drive.first_downs, drive.yards_gained, drive.play_count, drive.result, game.season_year, game.week, game.season_type, game.home_team, game.away_team
from drive
inner join game on drive.gsis_id=game.gsis_id
where season_type='Regular'""")

stats = cur.fetchall()

stats=pd.DataFrame(stats,columns=['pos_team','drive_id','pos_time','first_downs','yards_gained','play_count','result','season_year','week','season_type','home_team','away_team'])

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

# look at the numbers
# table = pd.pivot_table(stats_d, values=['yards_allowed','points_allowed','turnovers_forced'], index=['year', 'team'], aggfunc=np.sum)
# table.sort_values(by='turnovers_forced',ascending=False)

# aggregate rolling 5 week
## sort at year, team, week
stats_d.sort_values(by=['team','year','week'],inplace=True)
## sum across year team week
stats_d=stats_d.groupby(by=['team','year','week'],as_index=False).sum()
## rolling 5 week lagged
rolling = stats_d.groupby(['team'],as_index=False)['yards_allowed','points_allowed','turnovers_forced'].rolling(5).sum().shift(1).reset_index()
## join together
stats_d=stats_d.join(rolling,lsuffix='_weekly',rsuffix='_rolling')









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

