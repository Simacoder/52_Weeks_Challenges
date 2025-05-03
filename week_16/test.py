import pandas as pd
import os
import sys

import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# % matplotlib inline

# Load the data set via PANDAS
dataset = "data/netflix_titles.csv"
netflix_df = pd.read_csv(dataset)
# pd.set_option('display.max_columns', 12)
# pd.set_option('display.max_rows', 8807) 
# pd.set_option('display.max_colwidth', 25) 
print(netflix_df.head(5))

#print(netflix_df.info())
#print(netflix_df.describe())
#print(netflix_df.shape)

# how many years of data (1966-2021)
#print("Years of data: ", netflix_df["release_year"].value_counts())

# how many movies and shows or type
print("Movies and Shows: ", netflix_df["type"].value_counts())

# check for null values 
print("Null values: ", netflix_df.isnull().sum())

# replace Nan values with TV-MA
netflix_df["rating"].replace(np.nan, "TV-MA", inplace= True)

# repalce Nan values with united states
netflix_df["country"].replace(np.nan, "United States", inplace = True)

# rename the column "Listed_in" to "genre"
netflix_df.rename(columns={"listed_in": "genre"}, inplace = True, errors = "raise")
"""
Exploratory Analysis and Visualization
I have analysed and visualised many relationships and connections between different columns to get interesting insights from our dataset.

What have I done:

Computed the Sum(s) and Mean(s)
I have Explored the correlation & relationship between different data columns
I have noted a few of the interesting insights from the exploratory analysis
"""

# set the style of seaborn 
sns.set_style("darkgrid")
matplotlib.rcParams["font.size"] = 14
matplotlib.rcParams["figure.figsize"] = (9, 5)
matplotlib.rcParams["figure.facecolor"] = "white"
matplotlib.rcParams["figure.dpi"] = 100

# calculte the mean of released year
print(netflix_df["release_year"].value_counts().mean())

# The Netflix Release years, visualized via HORIZONTAL BAR GRAPH.

plt.title("Netflix Release Years")
netflix_df["release_year"].value_counts().head(10).plot.barh(figsize = (10, 5), color= "blue")
plt.show()


# TV Shows and movies 
plt.figure(figsize = (10, 5))
plt.title("TV Shows and Movies on Netfilx")
sns.barplot(x = netflix_df["type"].value_counts().index, y = netflix_df["type"].value_counts().values, palette = "Set2")
plt.xlabel("Type")
plt.ylabel("Count")
plt.show()

# The TV Show VS Movies On NETFLIX - visualised by a PIE CHART.

plt.title("Types of shows in netflix")
netflix_df["type"].value_counts().plot.pie(autopct="%1.1f%%", figsize=(10, 5) , colors= ["#ff9999", "#66b3ff"], startangle=90, shadow=True, explode=(0.1, 0), labels=["TV Shows", "Movies"])
plt.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(loc="upper right")
plt.show()

# The Netflix Rating Distribution, visualised by a COUNT PLOT.

plt.figure(figsize =(10, 5))
plt.title("Netflix ratings distribution")
sns.countplot( x = 'rating', data = netflix_df, palette= 'Set2')
plt.xticks(rotation = 90)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10, 5))
plt.title('Netflix ratings distribution separated by type of release')

# Pick a color palette for the hue categories ('Movie', 'TV Show')
hue_colors = sns.color_palette("Set2", len(netflix_df['type'].unique()))
hue_map = dict(zip(netflix_df['type'].unique(), hue_colors))

sns.countplot(
    x='rating',
    data=netflix_df,
    hue='type',
    palette=hue_map  # Map of 'Movie' -> color, 'TV Show' -> color
)

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Top 10 countries with the most movies and shows
plt.figure(figsize=(10,5))
plt.title('Top 10 Netflix Countries')
colors = sns.color_palette("husl", 9)
netflix_df["country"].value_counts().nlargest(n=10).plot.pie(autopct='%1.1f%%',figsize=(10,5), colors=colors)
plt.show()

# Top 10 countries with the most mocies and shows
plt.figure(figsize=(10,5))
plt.title('Top 10 Netflix Countries')
netflix_df["country"].value_counts().nlargest(n=10).plot.barh(figsize=(10,5), color="#42c2f5")
plt.show()

# Top 5 Netflix country distribution seperated by type of release (i.e Movies, TV shows), visualizing via Bar Graph.
plt.figure(figsize = (10,5))
plt.title('Top 5 Netflix country distribution seperated by type of release')
sns.countplot(x='country', data=netflix_df, hue='type', order=netflix_df.country.value_counts().iloc[:5].index, color="salmon")
plt.show()

# Top 5  in africa 
african_countries = [
    'Nigeria', 'South Africa', 'Egypt', 'Kenya', 'Morocco', 'Ghana',
    'Tunisia', 'Algeria', 'Uganda', 'Ethiopia'  
]

# Filter to rows where country is in Africa
africa_df = netflix_df[netflix_df['country'].isin(african_countries)]

# Get the top 5 most frequent countries
top5_african = africa_df['country'].value_counts().iloc[:5].index

# Plot
plt.figure(figsize=(10, 5))
plt.title('Top 5 Netflix country distribution in Africa')
sns.countplot(
    x='country',
    data=africa_df,
    hue='type',
    order=top5_african,
    palette='Set2'
)
plt.tight_layout()
plt.show()

# What is the proporationality between the GENRE of Movies and TV Shows?
# Count of movie genres
plt.figure(figsize = (10,5))
plt.title('Top 10 Genres for Movies')
colors = sns.color_palette('pastel')[0:11]
netflix_df[netflix_df["type"]=="Movie"]["genre"].value_counts()[:10].plot(kind='barh', color=colors)
plt.show()

# Count of TV Show genres
plt.figure(figsize = (10,5))
plt.title('Top 10 Genres for TV Shows')
colors = sns.color_palette("Set2")
netflix_df[netflix_df["type"]=="TV Show"]["genre"].value_counts()[:10].plot(kind='barh', color=colors)
plt.show()

"""
 Who are the top 10 actors on Netflix?
Who are the top actors? I was wondering and thought why not
"""
# Filling NAN with values
netflix_df['cast']=netflix_df['cast'].fillna('No Cast Specified') 

# Creating a DATA FRAME to store the filtered CAST (I.E ACTORS)
filtered_cast=pd.DataFrame() 
filtered_cast=netflix_df['cast'].str.split(',',expand=True).stack() 
filtered_cast=filtered_cast.to_frame()
filtered_cast.columns=['Actor']
actors=filtered_cast.groupby(['Actor']).size().reset_index(name='Total Content')
actors=actors[actors.Actor !='No Cast Specified'] 

# Sort the Values by the TOP 10
actors=actors.sort_values(by=['Total Content'],ascending=False) 
top_actors=actors.head(10) 
top_actors=top_actors.sort_values(by=['Total Content'])
x = top_actors["Actor"]
y = top_actors["Total Content"]

# PLOT the Bar Graph
plt.style.use("fivethirtyeight")
plt.figure(figsize = (10,5))
plt.title('Top 10 Actors')
sns.barplot(x = x, y = y)
plt.show()

# Does Netflix upload more Movies or TV shows?
plt.figure(figsize=(10,5))

plt.plot(netflix_df[netflix_df['type']=='TV Show'].groupby('release_year')['type'].count(), color='salmon', linewidth=4)
plt.plot(netflix_df[netflix_df['type']=='Movie'].groupby('release_year')['type'].count(), color='lightblue',linewidth=4)
plt.xlabel("Year")
plt.ylabel("Titles Added")
plt.title("Movies vs. TV Shows on Netflix\nCount of Titles Released per Year")
plt.legend(['TV Shows','Movies'])

plt.show()

# When is the Best to time to release a Movie/TV shows?
# Parsing the Dates
netflix_date = netflix_df[['date_added']].dropna()
netflix_date['year'] = netflix_date['date_added'].apply(lambda x : x.split(', ')[-1])
netflix_date['month'] = netflix_date['date_added'].apply(lambda x : x.lstrip().split(' ')[0])

# Adding the Months and Grouping 
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'][::-1]
df = netflix_date.groupby('year')['month'].value_counts().unstack().fillna(0)[month_order].T

# Customizing the figure design etc.
plt.figure(figsize=(10, 7), dpi=500)
plt.pcolor(df, cmap='Reds', edgecolors='white', linewidths=2) # heatmap

# Adding y and x ticks
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns, fontsize=7, fontfamily='serif')
plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=7, fontfamily='serif')

# Adding the Title
plt.title('Best Time to Release to Netflix', fontsize=12, position=(0.50, 1.0+0.02))

# Adding the Colorbar
cbar = plt.colorbar()

cbar.ax.tick_params(labelsize=8) 
cbar.ax.minorticks_on()
plt.show()

# What are the TV shows with largest number of seasons?

cond = netflix_df[['duration']].apply(lambda x:x.str.contains('Season|Seasons',regex =True)).any(axis =1)

query1 = netflix_df[cond]
query1.index = np.arange(len(query1))
query1['season']=query1.duration.str[:2]
list1 = list(query1['season'])
for i in range(len(list1)):
    list1[i] = int(list1[i])
query1['season'] = list1


plt.figure(figsize =(10,5))
sns.countplot(x=query1.season)
plt.title('TV Shows no.of seasons Distributions')
plt.style.use("fivethirtyeight")
plt.show()

sub_query1 =query1.season.groupby(query1.title).sum().sort_values(ascending = False)
plt.figure(figsize =(10,5))
sns.barplot(y=sub_query1[sub_query1>=9].index,x = sub_query1[sub_query1>=9].values)
plt.title('TV Shows with more no.of seasons')
plt.ylabel("Title")
plt.xlabel("Number of Seasons")
plt.show()

#  Who are the most famous directors?
query3 = netflix_df[~cond]
query3.index = np.arange(len(query3))
sub_query2 = query3.director.value_counts()[:10]
plt.figure(figsize=(10, 5))
sns.barplot(y = sub_query2.index,x = sub_query2.values)
plt.title('Director with most no.of movies')
plt.show()

query4 = netflix_df[cond]
query4.index = np.arange(len(query4))
sub_query3 = query4.director.value_counts()[:10]
plt.figure(figsize=(10,5))
sns.barplot(y = sub_query3.index,x = sub_query3.values)
plt.title('Director with most no.of TV Shows')
plt.show()

# Machine learning 



""" 
    Inferences and Conclusion
Finally, this project ended! It was an amazing project to say the least. Super intersting and educative.
 I learned a lot via doing this project!
I have deeply visualized this data on Netflix Movies & TV shows. From simple bar plots to complex heatmaps - all is here.
 I have infered that this data could be used in many ways. For example, in predicting the best time to release a movie? Or maybe how many seasons are enough and watchable? Or what type of genres are the most popular? 
All can be used from this EDA Project! I have made many visualizations, hoping to make this possible!
"""