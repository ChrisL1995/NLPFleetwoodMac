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



