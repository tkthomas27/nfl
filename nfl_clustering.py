#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:36:26 2017

@author: kthomas1
"""

# import pandas and numpy for 
import pandas as pd
import numpy as np

# connect to PostgreSQL
import psycopg2
conn=psycopg2.connect("dbname='nfldb' user='kthomas1' host='localhost' password='' port=5432")




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


##################################################
#create tiers of players
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

print(kmeans.inertia_)
##################################################

##################################################
#query cosine similarities to other players
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize

qb_nums['passing_sk_yds'] = qb_nums['passing_sk_yds']*-1

nmf = NMF(n_components=5)
nmf_features = nmf.fit_transform(qb_nums)
norm_features = normalize(nmf_features)
tom_brady = norm_features[0,:]
similarities = norm_features.dot(tom_brady)

a = pd.DataFrame(similarities)
x = a.join(qb_names)

#is 

##################################################
#t-sne
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

model = TSNE(learning_rate=190)
tx = model.fit_transform(qb_nums)
xs=tx[:,0]
ys = tx[:,1]
plt.scatter(xs,ys)
plt.show()

##################################################
#pca