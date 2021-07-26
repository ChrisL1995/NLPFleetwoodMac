# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:52:51 2021

@author: User
"""

import re
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pandas as pd
import glob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

lemmatizer = WordNetLemmatizer()
path = "/home/chris/Documents/PythonProjects/NLPFleetwoodMac/"
files = glob.glob(path + 'Lyrics/*.txt')
data = pd.read_excel(path+'SinglesList.xlsx')

def get_title(filepath):
    title = filepath[60:-4]
    return title

rawdata = {}
for file in files:
    lines = open(file,'r').readlines()
    for i in range(len(lines)):
        if lines[i][-1] == "\n":
            lines[i] = lines[i][:-1]
            lines[i] += ' '
    rawdata[get_title(file)] = np.array(lines)
    


# First, split into seperate words
splitdata = {}
for key in rawdata:
   splitdata[key] = np.concatenate([re.split(r'\W+',str(line)) for line in rawdata[key]])


# Lemmatize the words
for key in splitdata:
    splitdata[key] = [lemmatizer.lemmatize(word.lower()) for word in splitdata[key]]


# Get frequency of words from wordlist
lyricsplit = [splitdata[key] for key in splitdata]
lyricsplit = np.sum(lyricsplit)

# remove stopwords from wordlist
otherstops = ["","oh","ooh","baby","wa","ah","yeah"]
wordlist = [word for word in list(dict.fromkeys(lyricsplit)) if not word in stopwords.words('english')+otherstops]

lyricfreq = {}
for word in wordlist:
    lyricfreq[word] = lyricsplit.count(word)

# Sort into descending order:
    
sort = np.array(sorted(lyricfreq.items(), key = lambda x:x[1], reverse = True))
keys = sort[:,0]
values = sort[:,1].astype(int)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(keys[0:50],values[0:50])
plt.xticks(rotation=90)
ax.set_xlabel('lyric')
ax.set_ylabel('frequency')


# Topic modeling to find topics
# Use tf-idf

def join_lyrics(lyrics):
    test = ''
    for line in lyrics:
        test += line
    return test


# Get feature names from each song
vectorizer = TfidfVectorizer(stop_words=(stopwords.words('english')+otherstops), min_df = 0.1)
corpus = []
for key in rawdata:
    corpus.append(join_lyrics(rawdata[key]))

# convert corpus to lower case
for song in corpus:
    song = song.lower()
    

tfidf_freq = vectorizer.fit_transform(corpus)
# naive Bayes classifier to gauge mood?

nmf = NMF(n_components=5, random_state=1).fit(tfidf_freq)

feature_names = vectorizer.get_feature_names()

for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-10 - 1:-1]]))
    print()


# How does the prevelance of the topics change over time?

# Transform the copus to numeric values. (needed for the nmf transform but not too sure how this works exactly)
transformed_corpus = vectorizer.transform(corpus)

# Return the document-topic matrix (row = song, column = topic)

dtm = nmf.transform(transformed_corpus)


topics = np.array(["Reminiscing","Being in Love","Desire","Philophobia","Passage of Time"])
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(dtm[0])):    
    ax.plot(dtm[:,i], label = topics[i])
ax.legend()
ax.set_xticks(np.arange(0,len(dtm)))
ax.set_xlim(0,len(dtm)-1)
ax.set_xticklabels(rawdata.keys())
plt.xticks(rotation = 90)


# Need to put songs into order.
df = pd.DataFrame()
df["Song Title"] = data["Song Title"]
df["year"] = data["Song Year"]
arr1 = []
for song in df["Song Title"]:
    song = song.replace("'","")
    song = song.replace(" ", "")
    arr1.append(song)
df["Squish Title"] = arr1

arr2 = []
for song in df["Squish Title"]:
    index = np.where(song == np.array(list(rawdata.keys())))[0][0]

    
    arr2.append(dtm[index])
arr2 = np.array(arr2)

# convert to percentage:

arr2 = arr2.T/np.sum(arr2,1)
arr2 = arr2.T*100
for n, topic in enumerate(topics):
    df[topic] = arr2[:,n]

# now sort by data

df = df.sort_values(by = ["year"])

fig = plt.figure()
plotdf = pd.DataFrame()
for topic in topics:
    plotdf[topic] = df[topic]
plotdf.index =  df["Song Title"].values

ax = plotdf.plot.bar(stacked=True, figsize=(30,23))
fig = ax.get_figure()


# Create a similar plot but with only the nth popular topic

topicnum = 0
colours = ['blue','orange','green','red','purple']
fig = plt.figure()
fig.set_size_inches(30,23)
ax = fig.add_subplot(111)
for i in range(len(df["Song Title"])):
    song_topic = df[topics].iloc[i].sort_values(ascending = False)
    label = song_topic.keys()[0]
    cind = np.where(label == topics)[0][0]
    ax.bar(df['Song Title'].iloc[i], song_topic[topicnum]/song_topic[topicnum], label = song_topic.keys()[0], color=colours[cind])
plt.xticks(rotation=90)

