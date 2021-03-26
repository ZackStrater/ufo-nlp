

import pandas as pd

from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# data = pd.read_json('../data/ufodata.json.zip', lines=True)
# print(data.shape)
#
# def data_pipeline(data):
#     '''
#     input: raw dataframe with html column
#
#     '''
#
#     cleaned = {'contents': [], 'occurred': [], 'location': [], 'shape': []}
#     i = 0
#     for row in data.html:
#         i += 1
#         print(i)
#         soup = BeautifulSoup(row, "html.parser")
#         table = soup.find("tbody")
#         try:
#             text = table.text.strip()
#         except:
#             text=''
#         a = re.findall(r'(Duration.+?\n\n\n)(.+)(\(\(NUFORC Note:)?', text)
#         occurred = re.findall(r'Occurred : (.+)\s\s\(Entered', text)
#         location = re.findall(r'Location: (.+?)Shape', text)
#         shape = re.findall(r'Shape: (.+?)Duration', text)
#
#         try: cleaned['contents'].append(a[0][1])
#         except: cleaned['contents'].append(None)
#         try: cleaned['occurred'].append(occurred[0])
#         except: cleaned['occurred'].append(None)
#         try: cleaned['location'].append(location[0])
#         except: cleaned['location'].append(None)
#         try: cleaned['shape'].append(shape[0])
#         except: cleaned['shape'].append(None)
#
#
#     return pd.DataFrame(cleaned)
#
# df = data_pipeline(data)
# df.to_csv('../data/cleaned_ufo.csv', index=False)
# print('done loading')
# print(df.shape)
df = pd.read_csv('../data/cleaned_ufo.csv')
df['state'] = df['location'].str[-2:]

def plot_bar(ax, data, labels, title, x_axis, y_axis):
    x = np.arange(len(data))
    bars = ax.bar(x, data)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, size = 15)
    ax.set_xlabel(x_axis, size=20)
    ax.set_ylabel(y_axis, size=20)
    ax.set_title(title, size=25)

shape = df['shape'].value_counts().head(10)
state = df['state'].value_counts().head(11)


# fig, ax = plt.subplots(figsize=(16, 8))
# plot_bar(ax, shape, shape.index, 'Most Common UFO Shapes', 'Shape', 'Count')
# plt.show()

#
# fig, ax = plt.subplots(figsize=(16, 8))
# indexes = state.index.tolist()
# indexes[1] = 'NA'
# plot_bar(ax, state, indexes, 'States With Highest UFO Density', 'State', 'Count')
# plt.show()


word_tokens = [word_tokenize(document) for document in df['contents']]
lemmatizer = WordNetLemmatizer()
tokens_lemmatize = [list(map(lemmatizer.lemmatize, sent)) for sent in word_tokens]
tfidf_input = [' '.join(words) for words in tokens_lemmatize]


tfidf = TfidfVectorizer(stop_words='english', strip_accents='ascii', max_features=1000, ngram_range=(1,1))
X = tfidf.fit_transform(tfidf_input).todense()
features = np.array(tfidf.get_feature_names())


nmf = NMF(n_components=10, max_iter=1000)
nmf.fit(X)
W = nmf.transform(X)
H = nmf.components_
error = nmf.reconstruction_err_
print('reconstruction error:', nmf.reconstruction_err_)

sorted = np.array(np.argsort(H, axis=1))
top_ten_words = sorted[:, 990:]
for row in top_ten_words:
    print(features[row])
