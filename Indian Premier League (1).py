#!/usr/bin/env python
# coding: utf-8

# # Data Science & Business Analytics internship at TSF Group

# # The Sparks Foundation

# Task 5- Perform  Exploratory Data Analysis on dataset 'Indian Premier League'
# 
# As a sports analysts, find out the most successful teams, players and factorscontributing win or loss of a team.
# 
# By - Vishal Kobarne Data Science & Business Analytics intern at The Sparks Foundation (TSF)
# 
# 

# # Step -1:- Importing the required Libraries
# 

# In[56]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
print('Libraries are imported Successfully..')


# # Step-2:- Importing the dataset

# In[57]:


df_deliveries = pd.read_csv(r"C:\Users\Kiran\Downloads\Indian Premier League.zip (Unzipped Files)-20230403T075754Z-001\deliveries.csv")
print("Deliveries Data Loaded Successfully...")


# In[58]:


df_deliveries


# In[59]:


df_matches = pd.read_csv(r"C:\Users\Kiran\Downloads\Indian Premier League.zip (Unzipped Files)-20230403T075754Z-001\matches.csv")
print("Matches data Loaded Successfully... ")


# In[60]:


df_matches


# 

# # Checking the shape of data
# 

# In[61]:


df_deliveries.shape


# In[62]:


df_matches.shape


# # Full Data Summary
# 

# In[63]:


df_deliveries.info()


# In[64]:


df_matches.info()


# # Statistical Summary of Data

# In[65]:


df_deliveries.describe()


# In[66]:


df_matches.describe()


# # Columns in the data

# In[67]:


df_deliveries.columns


# In[68]:


df_matches.columns


# # Finding out Null values in Each Columns

# In[69]:


df_deliveries.isnull().sum()


# # Total number of NaN values in the deliveries dataset

# In[70]:


df_deliveries.isnull().sum().sum()


# --->In Deliveries dataset we can see that player_dismissed ,dismissal_kind and fielder columns contain null values.<---
# 

# In[71]:


df_matches.isnull().sum()


# # Total number of NaN values in the matches dataset

# In[72]:


df_matches.isnull().sum().sum()


# ---> In matches dataset we can see that city,winner,player_of_match,umpire1,umpire2 and umpire3 columns contain null values <---

# # Dropping of Columns having signigicant number of Null values

# In[73]:


df_matches1 =df_matches.drop(columns=['umpire3'], axis =1)


# # Verification of Dropped Column

# In[74]:


df_matches


# In[75]:



df_matches.isnull().sum()


# In[76]:


df_matches.fillna(0,inplace=True)


# In[77]:


df_matches


# In[78]:


df_matches.isnull().sum()


# We have successfully replace the Null values with 'Zeros'

# # Merging the dataset 

# In[79]:


season_data = df_matches[['id','season','winner']]
df  = df_deliveries.merge(season_data,how = 'inner',left_on = 'match_id',right_on = 'id')


# In[80]:


df.columns


# #                                   Data Visualization

# In[81]:



teams_wins_per_season =df_matches.groupby('season')['winner'].value_counts()


# In[82]:


teams_wins_per_season


# # Number of matches played in each IPL season

# In[83]:


plt.figure(figsize = (18,10))
sns.countplot(x='season',data=df_matches, palette='cool')
plt.title("Number of matches played in each IPL season ",fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("season",fontsize=15)
plt.ylabel("Matches",fontsize=15)
plt.show()


# # Numbers of matches won by team 

# In[84]:


plt.figure(figsize = (18,10))
sns.countplot(x='winner',data=df_matches, palette='hsv')
plt.title("Numbers of matches won by team ",fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Teams",fontsize=15)
plt.ylabel("No of wins",fontsize=15)
plt.show()


# In[85]:



df_matches.result.value_counts()


# In[86]:


plt.subplots(figsize=(10,6))
sns.countplot(x='season', hue='toss_decision', data=df_matches)
plt.show()


# # Matches played across each seasons

# In[87]:


plt.subplots(figsize=(10,8))
ax=df_matches['toss_winner'].value_counts().plot.bar(width=0.9,color=sns.color_palette('RdYlGn', 20))
for p in ax.patches:
          ax.annotate(format(p.get_height()),(p.get_x()+0.15, p.get_height()+1))
plt.show()                                                     
                


# # Top 10 Batsman from the dataset

# In[88]:


plt.subplots(figsize=(10,6))
max_runs=df_deliveries.groupby(['batsman'])['batsman_runs'].sum()
ax=max_runs.sort_values(ascending=False)[:10].plot.bar(width=0.8,color=sns.color_palette('cool',20))
for p in ax.patches:
    ax.annotate(format(p.get_height()),(p.get_x()+0.1, p.get_height()+50),fontsize=15)
plt.show()


# # Number of matches won by Toss winning side

# In[89]:


plt.figure(figsize = (18,10))
sns.countplot('season',hue='toss_winner',data=df_matches,palette='hsv')
plt.title("Numbers of matches won by batting and bowling first ",fontsize=20)
plt.xlabel("season",fontsize=15)
plt.ylabel("Count",fontsize=15)
plt.show()


# #  we will print winner season wise

# In[90]:


final_matches=df_matches.drop_duplicates(subset=['season'], keep='last')

final_matches[['season','winner']].reset_index(drop=True).sort_values('season')


# # Most Successful Teams in IPL

# In[91]:


wins = df_matches.groupby('winner').count()['result']


# In[92]:



wins = wins.sort_values(ascending=False)


# In[93]:



plt.figure(figsize=(10,5))
sns.barplot(x=wins.index, y=wins, palette="rocket")
plt.xticks(rotation=90)
plt.xlabel('Teams')
plt.ylabel('Number of Wins')
plt.title('Most Successful Teams in IPL')
plt.show()


# In[94]:


runs = df_deliveries.groupby('batsman').sum()['batsman_runs']


# In[95]:



wickets = df_deliveries[df_deliveries['dismissal_kind'] != 'run out'].groupby('bowler').count()['dismissal_kind']


# In[96]:



runs = runs.sort_values(ascending=False)[:10]
wickets = wickets.sort_values(ascending=False)[:10]


# # Most Successful Batsmen in IPL

# In[123]:


plt.figure(figsize=(10,5))
sns.barplot(x=runs.index, y=runs, palette="cool")
plt.xticks(rotation=90)
plt.xlabel('Players')
plt.ylabel('Total Runs')
plt.title('Most Successful Batsmen in IPL')
plt.show()


# # Most Successful Bowlers in IPL

# In[98]:




plt.figure(figsize=(10,5))
sns.barplot(x=wickets.index, y=wickets, palette="hsv")
plt.xticks(rotation=90)
plt.xlabel('Players')
plt.ylabel('Total Wickets')
plt.title('Most Successful Bowlers in IPL')
plt.show()


# # Percentage of Times Each Decision Led to a Win

# In[99]:


toss_decisions = df_matches[['toss_winner', 'toss_decision', 'winner']]
toss_decisions['toss_win_match_win'] = np.where(toss_decisions['toss_winner'] == toss_decisions['winner'], 'Yes', 'No')

bat_first_win_pct = round((toss_decisions[(toss_decisions['toss_decision'] == 'bat') & (toss_decisions['toss_win_match_win'] == 'Yes')].shape[0] / toss_decisions[toss_decisions['toss_decision'] == 'bat'].shape[0]) * 100, 2)
field_first_win_pct = round((toss_decisions[(toss_decisions['toss_decision'] == 'field') & (toss_decisions['toss_win_match_win'] == 'Yes')].shape[0] / toss_decisions[toss_decisions['toss_decision'] == 'field'].shape[0]) * 100, 2)

plt.figure(figsize=(6, 6))
plt.pie([bat_first_win_pct, field_first_win_pct], labels=['Bat First', 'Field First'], autopct='%1.1f%%')
plt.title('Percentage of Times Each Decision Led to a Win')
plt.show()


# # Teams which has won More number of Toss

# In[100]:


toss_ser =df_matches['toss_winner'].value_counts()
toss_df_matches=pd.DataFrame(columns=['team','wins'])

for items in toss_ser.iteritems():
    temp_df3=pd.DataFrame({
        'team':[items[0]],
        'wins':[items[1]]
    })
    toss_df_matches = toss_df_matches.append(temp_df3,ignore_index=True)


# In[101]:


plt.title('Which team won more number of Toss')
sns.barplot(x='wins', y='team', data=toss_df_matches, palette='bright')


# # Toss Result

# In[119]:


Toss=final_matches.toss_decision.value_counts()
labels=np.array(Toss.index)
sizes = Toss.values
colors = ['#FFBF00', '#FA8072']
plt.figure(figsize = (6,6))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True,startangle=90)
plt.title('Toss Result', fontsize=20)
plt.axis('equal')
plt.show()


# # We will print IPL finals venues and winners along with the number of wins

# In[103]:


final_matches.groupby(['city','winner']).size()


# # We will print number of season won by teams

# In[104]:


final_matches['winner'].value_counts()


# # We will print toss_winner , toss_decision , winner in final matches

# In[105]:


final_matches[['toss_winner','toss_decision','winner']].reset_index(drop = True)


# # We will print Man of the match

# In[106]:


final_matches[['winner','player_of_match']].reset_index(drop = True)


# In[107]:


len(final_matches[final_matches['toss_winner']== final_matches['winner']]['winner'])


#  # we will plot graph on four hit by teams

# In[108]:


season_data=df_matches[['id','season','winner']]
complete_data=df_deliveries.merge(season_data,how='inner',left_on='match_id',right_on='id')
four_data=complete_data[complete_data['batsman_runs']==4]
four_data.groupby('batting_team')['batsman_runs'].agg([('runs by fours','sum'),('fours','count')])


# In[109]:


batsman_four=four_data.groupby('batsman')['batsman_runs'].agg([('four','count')]).reset_index().sort_values('four',ascending=0)
ax=batsman_four.iloc[:10,:].plot('batsman','four',kind='bar',color='purple')
plt.title("Numbers of fours hit by playes ",fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Player name",fontsize=15)
plt.ylabel("No of fours",fontsize=15)
plt.show()


# #  we will plot graph on no of four hit in each season
# 

# In[110]:


ax=four_data.groupby('season')['batsman_runs'].agg([('four','count')]).reset_index().plot('season','four',kind='bar',color = 'yellow')
plt.title("Numbers of fours hit in each season ",fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("season",fontsize=15)
plt.ylabel("No of fours",fontsize=15)
plt.show()


# # we will print no of sixes hit by team
# 

# In[111]:


six_data=complete_data[complete_data['batsman_runs']==6]
six_data.groupby('batting_team')['batsman_runs'].agg([('runs by six','sum'),('sixes','count')])


# In[112]:


batsman_six=six_data.groupby('batsman')['batsman_runs'].agg([('six','count')]).reset_index().sort_values('six',ascending=0)
ax=batsman_six.iloc[:10,:].plot('batsman','six',kind='bar',color='Blue')
plt.title("Numbers of six hit by playes ",fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("Player name",fontsize=15)
plt.ylabel("No of six",fontsize=15)
plt.show()


# # we will plot graph on no of six hit in each season
# 

# In[113]:


ax=six_data.groupby('season')['batsman_runs'].agg([('six','count')]).reset_index().plot('season','six',kind='bar',color = 'Orange')
plt.title("Numbers of fours hit in each season ",fontsize=20)
plt.xticks(rotation=50)
plt.xlabel("season",fontsize=15)
plt.ylabel("No of fours",fontsize=15)
plt.show()


# # we will print the top 10 leading run scorer in IPL
# 

# In[114]:


batsman_score = df_deliveries.groupby('batsman')['batsman_runs'].agg(['sum']).reset_index().sort_values('sum',ascending = False).reset_index(drop = True)
batsman_score = batsman_score.rename(columns = {'sum':'batsman_runs'})
print("--+--+--+--+--> TOP 10 LEADING RUN SCORER IN IPL <--+--+--+--+-- ")
batsman_score.iloc[:10,:]


# # We will print no of matches played by batsman

# In[115]:


No_of_matches_played = df_deliveries[['match_id','player_dismissed']]
No_of_matches_played = No_of_matches_played.groupby('player_dismissed')['match_id'].count().reset_index().sort_values(by = 'match_id',ascending = False).reset_index(drop = True)
No_of_matches_played.columns = ['batsman','No_of_matches_played']
No_of_matches_played.head()


# # Dismissals in IPL
# 

# In[116]:



plt.figure(figsize=(18,10))
ax=sns.countplot(df_deliveries.dismissal_kind)
plt.title("Dismissals in IPL",fontsize=20)
plt.xlabel("Dismissals kind",fontsize=15)
plt.ylabel("count",fontsize=15)
plt.xticks(rotation=50)
plt.show()


# In[117]:


wicket_data=df_deliveries.dropna(subset=['dismissal_kind'])
wicket_data=wicket_data[~wicket_data['dismissal_kind'].isin(['run out','retired hurt','obstructing the field'])]


# # We will print IPL most  wicket taking bowlers

# In[118]:


wicket_data.groupby('bowler')['dismissal_kind'].agg(['count']).reset_index().sort_values('count',ascending=False).reset_index(drop=True).iloc[:10,:]


# # Conslusion :
# The highest number of match played in IPL season was 2013,2014,2015.
# 
# The highest number of match won by Mumbai Indians i.e 4 match out of 12 matches.
# 
# Teams which Bowl first has higher chances of winning then the team which bat first.
# 
# After winning toss more teams decide to do fielding first.
# 
# In finals teams which decide to do fielding first win the matches more then the team which bat first.
# 
# In finals most teams after winning toss decide to do fielding first.
# 
# Top player of match winning are CH gayle, AB de villers.
# 
# It is interesting that out of 12 IPL finals,9 times the team that won the toss was also the winner of IPL.
# 
# The highest number of four hit by player is Shikar Dhawan.
# 
# The highest number of six hit by player is CH gayle.
# 
# Top leading run scorer in IPL are Virat kholi, SK Raina, RG Sharma.
# 
# Dismissals in IPL was most by Catch out.
# 
# The IPL most wicket taken blower is SL Malinga.
# 
# The highest number of matches played by player name are SK Raina, RG Sharma.
# 
# Teams which has won more number of toss is Mumbai Indians.
# 

# In[ ]:





# In[ ]:




