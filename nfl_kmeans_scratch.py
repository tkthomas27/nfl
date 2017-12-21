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


for char in 'aaaa':
    locs = [pos for pos, ele in enumerate('aaaa') if ele == char]
    print(locs)


    substr_dict = {}
    
    x=0
    
    for char in line:
        locs = [pos for pos, ele in enumerate(line) if ele == char]
        max_loc = max(locs)
        subst = line[x:max_loc+1]
        substr_dict[subst]=len(subst)
        x+=1
    

    for x in range(0,str_len):
        if len(set(line[first_index:final_index])) == len(line[first_index:final_index]):
            fi=first_index
        else:
            first_index+1
            








line = 'abcabcffab'

str_len = len(line)

while str_len>-1:
    if len(set(line[0:str_len])) == len(line[0:str_len]):
        last=str_len
        break
    else:
        print(str_len)
        str_len = str_len-1
            
first_index = 0

while first_index<len(line):
    if len(set(line[first_index:last])) == len(line[first_index:last]):
        first=first_index
        break
    else:
        first_index = first_index+1

answer = line[first:last]

print(answer)



line = 'abcabcffab'

first_index = 0
last_index = len(line)

while str_len>-1:
    if len(set(line[first_index:last_index])) == len(line[first_index:last_index]):
        first=first_index
    elif len(set(line[first_index:last_index])) == len(line[first_index:last_index]):
        last=str_len
    elif first>=0 and last>0:
        break
    else:
        first_index = first_index+1
        last_index = last_index-1

answer = line[first:last]

print(answer)





line = 'abcabcffab'

first_index = 0
last_index = len(line)

while str_len>-1:
    if len(set(line[first_index:last])) == len(line[first_index:last]):
        first=first_index
        break
    else:
        first_index = first_index+1

answer = line[first:last]

print(answer)



line = 'abcabcffab'
fw = 0
lw = len(line)
window = lw-fw


while window>0:
    while lw<len(line):
        if len(set(line[fw:lw])) == len(line[fw:lw]):
            first = fw
            last = lw
            break
        else:
            fw=fw+1
            lw=lw+1
    window = window-1
    fw=0
    lw=window

        

0,10
0,9
1,10
0,8
1,9
2,10
0,7
1,8
2,9
3,10
    
    



line = 'abcabcffab'
fw = 0
lw = len(line)
window = lw-fw

while window>0:
    while lw<=len(line):
        if len(set(line[fw:lw])) == len(line[fw:lw]):
            first = fw
            last = lw
            break
        else:
            fw=fw+1
            lw=window
    window = window-1
    print(lw)
    fw=0
    lw=window

line[first:last]




def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)

factorial(3)


def window(n):
    i = n-1
    if i==n:
        return 1
    else:
        return n * factorial(n-1)

window(10)




def u(n):
  ux = u0
  for i in xrange(n):
    ux=f(ux)
  return ux


0,10
0,9
1,10
0,8
1,9
2,10
0,7
1,8
2,9
3,10
    

def a(n):
    y=0
    while count<n:
        y = y+1
        count = count+1


def string_splosion(string):
    return ''.join(string[:i] for i in range(1, len(string) + 1))

x=0
while x<10:
    for i in range(0,x):
        print([i,10-i])
    x=x+1


line = 'abdjwawk'
x=0
while x<len(line):
    for i in range(0,x):
        w=len(line)-i
        if len(set(line[i:w])) == len(line[i:w]):
                    first = i
                    last = w
                    break
    x=x+1

line[i:w]


x=0
while x<10:
    for i in range(0,x):
        print(i)
    x=x+1

#1,1,2,1,2,3,1,2,3,4

x=10
while x>-2:
    for i in range(10,x,-1):
        print(i)
    x=x-1
#10,10,9,10,9,10,9,8

x=10
while x>=0:
    for i in range(x,11):
        print(i)
    x=x-1

#10,9,10,8,9,10




line = 'aaaaa'

alist = []
blist = []

a=0
while a<len(line):
    for i in range(0,a):
        alist.append(i)
    a=a+1

b=len(line)
c=b+1
while b>=0:
    for i in range(b,c):
        blist.append(i)
    b=b-1

inds = list(zip(alist,blist))

for i in range(0,len(inds)):
    x=inds[i][0]
    y=inds[i][1]
    print(line[x:y])
    if len(set(line[x:y])) == len(line[x:y]):
        first = inds[i][0]
        last = inds[i][1]
        break

line[first:last]






def intervals(line):
    last = {}
    start = end = 0
    for i, letter in enumerate(line):
        end += 1
        if letter in last and start <= last[letter]:
            start = last[letter] + 1
        yield line[start:end]
        last[letter] = i


def non_repeat(line):
    return max(intervals(line), key=len, default='')

non_repeat('abdjwawk')
    

def non_repeat(line):
    return (line if len(line) == len(set(line)) else
            max(non_repeat(line[:-1]),
                non_repeat(line[1:]), key=len)
            )
            
    
non_repeat('abdjwawk')
    
    
    
def create_intervals(data):
    
    l = sorted(data)
    
    ans=[]
    
    while len(l)>0:
        for x,y in enumerate(l):
            try:
                if (l[x+1]-l[x])>1:
                    ans.append((l[0],l[x]))
                    l=l[x:]
            except:
                pass

create_intervals({1, 2, 3, 4, 5, 7, 8, 12})
    
    


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



a=21437
b=-21437

if (a>=0 and b>=0) or (a<=0 and b<=0):
    la = list(range(1,abs(a)+1))
    lb = list(range(1,abs(b)+1))
    la.extend(lb)
            
elif a<0:
    lb = list(range(1,abs(b)+1))
    la = lb[:a]
            
elif b<0:
    lb = list(range(1,abs(a)+1))
    la = lb[:b]
        
print(len(la))





x=[1,2,3,4]



def reverse(self, x):
    s = cmp(x, 0)
    r = int(`s*x`[::-1])
    return s*r * (r < 2**31)







import numpy as np
import scipy.stats as stats
dist = stats.beta
data = stats.bernoulli.rvs(0.5, size=n_trials[-1])
x = np.linspace(0, 1, 100)
















