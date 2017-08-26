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
# turnover

# defense
# turnovers forced
# yards allowed



