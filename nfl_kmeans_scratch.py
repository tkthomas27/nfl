#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 19:23:33 2017

@author: kthomas1
"""



# import pandas and numpy for 
import pandas as pd
import numpy as np

# connect to PostgreSQL
import psycopg2
conn=psycopg2.connect("dbname='nfldb' user='kthomas1' host='localhost' password='' port=5432")


# tiers of colleges in the nfl
# list of colleges -> must have at least 50 players?

# aggregate player stats from play_player



# can we use offensive stats from the NFL to predict what conference a player is from
# label colleges with power5 conference labels
# aggregate offensive statistics: TDs, Yards


players=pd.read_sql("""select player_id, position, college 
                    from player""",con=conn)

# merge in power five lables
power5 = pd.read_csv("~/github_data/power5.csv")

players_power5 = pd.merge(players,power5,how='inner',on=['college'])

# count of players in each conference
pd.crosstab(index=players_power5['conf'],columns = "count")



# get stats
o_stats=pd.read_sql("""select player_id, count(player_id) as freq, sum(passing_cmp_air_yds) as pass_yds, sum(passing_tds) as pass_tds, sum(receiving_tds) as rec_tds, sum(receiving_yds) rec_yds, sum(rushing_yds) as rush_yds, sum(rushing_tds) as rush_tds
from play_player
group by player_id""",con=conn)

stats_conf = pd.merge(o_stats, players_power5, how='inner', on=['player_id'])

stats_conf['tds'] = stats_conf['rush_tds'] + stats_conf['pass_tds'] + stats_conf['rec_tds']

stats_conf['yds'] = stats_conf['rush_yds'] + stats_conf['pass_yds'] + stats_conf['rec_yds']

pd.pivot_table(stats_conf, values = ['tds','yds'], index=['conf'], aggfunc=(np.sum))


#import matplotlib.pyplot as plt
#plt.hist(stats_conf['tds'])
#plt.show()
#plt.hist(stats_conf['yds'])
#plt.show()

#np.percentile(stats_conf['freq'],50)
# median is 45 so let's take 

# sum(x >= np.percentile(stats_conf['freq'],25) for x in stats_conf['freq'])

#stats_conf = stats_conf[stats_conf.freq >= np.percentile(stats_conf['freq'],50)]

#stats_conf = stats_conf[stats_conf.yds >= 1000]

pd.crosstab(index=stats_conf['conf'],columns = "count")
pd.crosstab(index=stats_conf['position'],columns = "count")

stats = stats_conf[['pass_yds','pass_tds','rec_tds','rec_yds','rush_yds','rush_tds']]
conf = stats_conf[['conf']].reset_index()
position = stats_conf[['position']].reset_index()



# standardize columns and fit kmeans
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=5)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

pipeline.fit(stats)

pred_conf = pd.DataFrame(pipeline.predict(stats),columns=['pred_conf'])

pd.crosstab(index=pred_conf['pred_conf'],columns = "count")


kmeans_pred = conf.join(pred_conf)

pd.crosstab(kmeans_pred['pred_conf'],kmeans_pred['conf'])





xs = stats_conf['tds']
ys = stats_conf['yds']
plt.scatter(xs,ys,alpha=0.5)































# get stats
o_stats = pd.read_sql("""select * from play_player""",con=conn)
defense = o_stats.groupby(['player_id'], as_index=False)[o_stats.filter(regex='defense_.*').columns].sum()
offense = o_stats.groupby(['player_id'], as_index=False)[o_stats.filter(regex='defense_.*').columns].sum()

stats_conf = pd.merge(o_stats, players_power5, how='inner', on=['player_id'])

stats_conf['tds'] = stats_conf['rush_tds'] + stats_conf['pass_tds'] + stats_conf['rec_tds']

stats_conf['yds'] = stats_conf['rush_yds'] + stats_conf['pass_yds'] + stats_conf['rec_yds']

pd.pivot_table(stats_conf, values = ['tds','yds'], index=['conf'], aggfunc=(np.sum))


#import matplotlib.pyplot as plt
#plt.hist(stats_conf['tds'])
#plt.show()
#plt.hist(stats_conf['yds'])
#plt.show()

#np.percentile(stats_conf['freq'],50)
# median is 45 so let's take 

# sum(x >= np.percentile(stats_conf['freq'],25) for x in stats_conf['freq'])

#stats_conf = stats_conf[stats_conf.freq >= np.percentile(stats_conf['freq'],50)]

#stats_conf = stats_conf[stats_conf.yds >= 1000]

pd.crosstab(index=stats_conf['conf'],columns = "count")
pd.crosstab(index=stats_conf['position'],columns = "count")

stats = stats_conf[['pass_yds','pass_tds','rec_tds','rec_yds','rush_yds','rush_tds']]
conf = stats_conf[['conf']].reset_index()
position = stats_conf[['position']].reset_index()



# standardize columns and fit kmeans
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=5)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

pipeline.fit(stats)

pred_conf = pd.DataFrame(pipeline.predict(stats),columns=['pred_conf'])

pd.crosstab(index=pred_conf['pred_conf'],columns = "count")


kmeans_pred = conf.join(pred_conf)

pd.crosstab(kmeans_pred['pred_conf'],kmeans_pred['conf'])





xs = stats_conf['tds']
ys = stats_conf['yds']
plt.scatter(xs,ys,alpha=0.5)






















# standardize columns and fit kmeans
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# group quarterbacks

# get stats
pass_stats=pd.read_sql("""select player_id, passing_att, passing_cmp, passing_cmp_air_yds, passing_incmp, passing_incmp_air_yds, passing_int, passing_sk, passing_sk_yds, passing_tds, passing_twopta, passing_twoptm, passing_twoptmissed, passing_yds
from play_player""",con=conn)

pass_stats = pass_stats.groupby('player_id').sum()
pass_stats['player_id']=pass_stats.index

qbs=pd.read_sql("""select player_id, position, full_name
from player
where position='QB'""",con=conn)

qb_stats = pd.merge(pass_stats, qbs, how='inner', on='player_id')

qb_names = qb_stats[['full_name']]
qb_nums = qb_stats[qb_stats.filter(regex='passing.*').columns]


# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=5)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

pipeline.fit(qb_nums)

qb_group = pd.DataFrame(pipeline.predict(qb_nums),columns=['qb_group'])

pd.crosstab(index=qb_group['qb_group'],columns = "count")

kmeans_group = qb_names.join(qb_group)


ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    result=model.fit(qb_nums)
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# looks like 2 is a good number of groups --> but I think this is being influenced by all the zeroes
# remove zeroes and 


qb_stats = pd.merge(pass_stats, qbs, how='inner', on='player_id')

qb_stats = qb_stats[qb_stats.passing_att >= np.percentile(qb_stats.passing_att,25)]

qb_names = qb_stats[['full_name']]
qb_nums = qb_stats[qb_stats.filter(regex='passing.*').columns]


# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=5)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

pipeline.fit(qb_nums)

qb_group = pd.DataFrame(pipeline.predict(qb_nums),columns=['qb_group'])

pd.crosstab(index=qb_group['qb_group'],columns = "count")

kmeans_group = qb_names.join(qb_group)



#feature selection 













defense = o_stats.groupby(['player_id'], as_index=False)[o_stats.filter(regex='defense_.*').columns].sum()
offense = o_stats.groupby(['player_id'], as_index=False)[o_stats.filter(regex='defense_.*').columns].sum()

stats_conf = pd.merge(o_stats, players_power5, how='inner', on=['player_id'])

stats_conf['tds'] = stats_conf['rush_tds'] + stats_conf['pass_tds'] + stats_conf['rec_tds']

stats_conf['yds'] = stats_conf['rush_yds'] + stats_conf['pass_yds'] + stats_conf['rec_yds']

pd.pivot_table(stats_conf, values = ['tds','yds'], index=['conf'], aggfunc=(np.sum))


#import matplotlib.pyplot as plt
#plt.hist(stats_conf['tds'])
#plt.show()
#plt.hist(stats_conf['yds'])
#plt.show()

#np.percentile(stats_conf['freq'],50)
# median is 45 so let's take 

# sum(x >= np.percentile(stats_conf['freq'],25) for x in stats_conf['freq'])

#stats_conf = stats_conf[stats_conf.freq >= np.percentile(stats_conf['freq'],50)]

#stats_conf = stats_conf[stats_conf.yds >= 1000]

pd.crosstab(index=stats_conf['conf'],columns = "count")
pd.crosstab(index=stats_conf['position'],columns = "count")

stats = stats_conf[['pass_yds','pass_tds','rec_tds','rec_yds','rush_yds','rush_tds']]
conf = stats_conf[['conf']].reset_index()
position = stats_conf[['position']].reset_index()



# standardize columns and fit kmeans
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=5)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

pipeline.fit(stats)

pred_conf = pd.DataFrame(pipeline.predict(stats),columns=['pred_conf'])

pd.crosstab(index=pred_conf['pred_conf'],columns = "count")


kmeans_pred = conf.join(pred_conf)

pd.crosstab(kmeans_pred['pred_conf'],kmeans_pred['conf'])





xs = stats_conf['tds']
ys = stats_conf['yds']
plt.scatter(xs,ys,alpha=0.5)







x = "abcabcffab"
for char in x:
    if x.count(char)>1:
        print(x.index(char))


for i in enumerate(x):
    indices = [a for a, b in enumerate(x[i[0]:]) if b == i[1]]
    print(i[0],indices[1])

















