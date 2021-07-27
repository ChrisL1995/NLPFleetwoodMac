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

def join_lyrics(lyrics):
    test = ''
    for line in lyrics:
        test += line
    return test


def get_corpus(lyric_dataframe):
    corpus = []
    for key in lyric_dataframe:
        corpus.append(join_lyrics(lyric_dataframe[key]))
    
    # convert corpus to lower case
    for song in corpus:
        song = song.lower()
    return corpus

def squish_title(df):
    arr1 = []
    for song in df["Song Title"]:
        song = song.replace("'","")
        song = song.replace(" ", "")
        arr1.append(song)
    df["Squish Title"] = arr1
    return df    

def add_dtm(df,lyric_dataframe,dtm,topics, percent=True):
    arr2 = []
    for song in df["Squish Title"]:
        index = np.where(song == np.array(list(lyric_dataframe.keys())))[0][0]
        arr2.append(dtm[index])
    arr2 = np.array(arr2)
    
    # convert to percentage:
    if percent:
        arr2 = arr2.T/np.sum(arr2,1)
        arr2 = arr2.T*100
    for n, topic in enumerate(topics):
        df[topic] = arr2[:,n]
    
    return df

def bag_of_words(lyric_dataframe, extrastops = None):
    
    # First, split into seperate words
    splitdata = {}
    for key in lyric_dataframe:
       splitdata[key] = np.concatenate([re.split(r'\W+',str(line)) for line in lyric_dataframe[key]])

    # Lemmatize the words
    for key in splitdata:
        splitdata[key] = [lemmatizer.lemmatize(word.lower()) for word in splitdata[key]]
    

    # Get frequency of words from wordlist
    lyricsplit = [splitdata[key] for key in splitdata]
    lyricsplit = np.sum(lyricsplit)

# remove stopwords from wordlist

    wordlist = [word for word in list(dict.fromkeys(lyricsplit)) if not word in stopwords.words('english')+otherstops]

    lyricfreq = {}
    for word in wordlist:
        lyricfreq[word] = lyricsplit.count(word)
    
    # Sort into descending order:
        
    sort = np.array(sorted(lyricfreq.items(), key = lambda x:x[1], reverse = True))
    keys = sort[:,0]
    values = sort[:,1].astype(int)

    return keys, values


def feature_extraction(corpus, mindf=0.1, extrastops = None):

    # Get feature names from each song
    vectorizer = TfidfVectorizer(stop_words=(stopwords.words('english')+otherstops), min_df = 0.1)
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
    
    return nmf, dtm


def bag_of_words_artist(Artist, lyric_dataframe, extrastops=None):
    df = pd.DataFrame(data.iloc[np.where(data['Writer'] == Artist)[0]])
    df = squish_title(df)
    dfRaw = {}
    for song in df["Squish Title"]:
        dfRaw[song] = lyric_dataframe[song]
    keys, values = bag_of_words(dfRaw, extrastops)
    
    return keys, values
        

rawdata = {}
for file in files:
    lines = open(file,'r').readlines()
    for i in range(len(lines)):
        if lines[i][-1] == "\n":
            lines[i] = lines[i][:-1]
            lines[i] += ' '
    rawdata[get_title(file)] = np.array(lines)
    
otherstops = ["","oh","ooh","baby","wa","ah","yeah"]

keys, values = bag_of_words(rawdata, otherstops)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(keys[0:50],values[0:50])
plt.xticks(rotation=90)
ax.set_xlabel('lyric')
ax.set_ylabel('frequency')


# Topic modeling to find topics
# Use tf-idf
corpus = get_corpus(rawdata)

nmf, dtm = feature_extraction(corpus, extrastops = otherstops)

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
df = squish_title(df)

df = add_dtm(df, rawdata, dtm, topics)

# now sort by data

df = df.sort_values(by = ["year"])
colors = ["black","brown","red","orange","yellow"]
plotdf = pd.DataFrame()
for topic in topics:
    plotdf[topic] = df[topic]
plotdf.index =  df["Song Title"].values

ax = plotdf.plot.bar(stacked=True, figsize=(30,23), color = colors)
fig = ax.get_figure()
# fig.savefig("C:/Users/User/Documents/Python Scripts/NLPFleetwoodMac/Figures/Topic_per_song_percent.png")


# Create a similar plot but with only the nth popular topic

topicnum = 0
fig = plt.figure()
fig.set_size_inches(30,23)
ax = fig.add_subplot(111)
for i in range(len(df["Song Title"])):
    song_topic = df[topics].iloc[i].sort_values(ascending = False)
    label = song_topic.keys()[0]
    cind = np.where(label == topics)[0][0]
    ax.bar(df['Song Title'].iloc[i], song_topic[topicnum]/song_topic[topicnum], label = song_topic.keys()[0], color=colors[cind])
plt.xticks(rotation=90)
# fig.savefig("C:/Users/User/Documents/Python Scripts/NLPFleetwoodMac/Figures/Topic_0_song.png")

#%%

#Split data based on artist

# Mcvie = pd.DataFrame(data.iloc[np.where(data['Writer'] == "Christine Mcvie")[0]])
# Nicks = pd.DataFrame(data.iloc[np.where(data['Writer'] == "Stevie Nicks")[0]])
# Buckingham = pd.DataFrame(data.iloc[np.where(data['Writer'] == "Lindsey Buckingham")[0]])
# Green = pd.DataFrame(data.iloc[np.where(data['Writer'] == "Peter Green")[0]])


# # Using the above dataframes, limit the raw data:
# Mcvie = squish_title(Mcvie)
# Nicks = squish_title(Nicks)
# Buckingham = squish_title(Buckingham)
# Green = squish_title(Green)

# McvieRaw = {}
# for song in Mcvie["Squish Title"]:
#     index = np.where(song == np.array(list(rawdata.keys())))[0][0]
#     McvieRaw[song] = rawdata[song]
    
# keys, values = bag_of_words(McvieRaw, extrastops=otherstops)

McvieK, McvieV = bag_of_words_artist("Christine Mcvie",rawdata, extrastops = otherstops)
NicksK, NicksV = bag_of_words_artist("Stevie Nicks",rawdata, extrastops = otherstops)
BuckinghamK, BuckinghamV = bag_of_words_artist("Lindsey Buckingham",rawdata, extrastops = otherstops)
GreenK, GreenV = bag_of_words_artist("Peter Green",rawdata, extrastops = otherstops)

def cloud_script(K,V): 
    script = ''
    for i in range(len(K)):
        for j in range(V[i]):
            script += K[i] + " "
    return script

# Mscript = cloud_script(McvieK, McvieV)
# Nscript = cloud_script(NicksK, NicksV)
# Bscript = cloud_script(BuckinghamK, BuckinghamV)
# Gscript = cloud_script(GreenK, GreenV)

def unique(K1,V1,K2,K3,K4):
    arrk1 = []
    arrv1 = []
    for n,word in enumerate(K1):
        if word not in K2 and word not in K3 and word not in K4:
            arrk1.append(word)
            arrv1.append(V1[n])
    return arrk1, arrv1

McvieK2, McvieV2 = unique(McvieK, McvieV, BuckinghamK, NicksK, GreenK)
NicksK2, NicksV2 = unique(NicksK, NicksV, BuckinghamK, McvieK, GreenK)
BuckinghamK2, BuckinghamV2 = unique(BuckinghamK, BuckinghamV, McvieK, NicksK, GreenK)
GreenK2, GreenV2 = unique(GreenK, GreenV, BuckinghamK, NicksK, McvieK)

Mscript = cloud_script(McvieK2, McvieV2)
Nscript = cloud_script(NicksK2, NicksV2)
Bscript = cloud_script(BuckinghamK2, BuckinghamV2)
Gscript = cloud_script(GreenK2, GreenV2)

#%%

# Create pie chart with topics

Mcvie = pd.DataFrame(data.iloc[np.where(data['Writer'] == "Christine Mcvie")[0]])
Nicks = pd.DataFrame(data.iloc[np.where(data['Writer'] == "Stevie Nicks")[0]])
Buckingham = pd.DataFrame(data.iloc[np.where(data['Writer'] == "Lindsey Buckingham")[0]])
Green = pd.DataFrame(data.iloc[np.where(data['Writer'] == "Peter Green")[0]])
            
Mcvie = squish_title(Mcvie)
Nicks = squish_title(Nicks)
Buckingham = squish_title(Buckingham)
Green = squish_title(Green)

Mcvie = add_dtm(Mcvie, rawdata, dtm, topics, percent=False)
Nicks = add_dtm(Nicks, rawdata, dtm, topics, percent=False)
Buckingham = add_dtm(Buckingham, rawdata, dtm, topics, percent=False)
Green = add_dtm(Green, rawdata, dtm, topics, percent=False)

def pie_data(df):
    pie = []
    for topic in topics:
        pie.append(df[topic].sum())
    return pie
    
Mpie = pie_data(Mcvie)
Npie = pie_data(Nicks)
Bpie = pie_data(Buckingham)
Gpie = pie_data(Green)

