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

lemmatizer = WordNetLemmatizer()
path = 'C:/Users/User/Documents/Python Scripts/NLPFleetMacPython/NLPFleetwoodMac/'
files = glob.glob(path + 'Lyrics/*.txt')
data = pd.read_excel(path+'SinglesList.xlsx')

def get_title(filepath):
    title = filepath[80:-4]
    return title

rawdata = {}
for file in files:
    rawdata[get_title(file)] = np.array(open(file,'r').readlines())
    


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

wordlist = [word for word in list(dict.fromkeys(lyricsplit)) if not word in stopwords.words('english')+['']]

lyricfreq = {}
for word in wordlist:
    lyricfreq[word] = lyricsplit.count(word)

# Sort into descending order:
    
sort = np.array(sorted(lyricfreq.items(), key = lambda x:x[1], reverse = True))
keys = sort[:,0]
values = sort[:,1].astype(int)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(keys[0:30],values[0:30])
plt.xticks(rotation=90)

# Use bag-of-word analysis

# Use n-gram analysis

# naive Bayes classifier to gauge mood?



