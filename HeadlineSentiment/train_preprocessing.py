

import pandas as pd
import numpy as np


import nltk
nltk.download('stopwords')
nltk.download('punkt')


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import PorterStemmer

ps = PorterStemmer()
#from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=1000)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

import re


import datetime



train_url = 'https://raw.githubusercontent.com/BrunoFromMars/ZS/master/train_file_2.csv'

df_new = pd.read_csv(train_url)

df_new.head()







appos = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not"
}




processed_title,processed_headline = [],[]
for i in range(df_new.shape[0]):
    t = df_new.Title[i].lower().split()
    h = df_new.Headline[i].lower().split()
    #print(t)
    #print(h)
    #negation handling
    reformed_t = [appos[word] if word in appos else word for word in t]
    t = " ".join(reformed_t) 
    reformed_h = [appos[word] if word in appos else word for word in h]
    h = " ".join(reformed_h)
    #tokenize
    t = word_tokenize(t)
    h = word_tokenize(h)
    #alpha numeric
    words_t = [word for word in t if word.isalpha()]
    words_h = [word for word in h if word.isalpha()]
    #stop-words
    filtered_t = [w for w in words_t if not w in stop_words]
    filtered_h = [w for w in words_h if not w in stop_words]
    #stemming
    t =""
    h =""
    for w in filtered_t:
        t += ps.stem(w) + " "
    for w in filtered_h:
        h += ps.stem(w) + " "
        
    processed_title.append(t)
    processed_headline.append(h)
#print(t)
#print(h)


vectorizer1 = TfidfVectorizer(sublinear_tf=True, max_df=1.0)
bow1 = vectorizer1.fit_transform(processed_title)

freqs1 = [(word, bow1.getcol(idx).sum()) for word, idx in vectorizer1.vocabulary_.items()]
results1=sorted (freqs1, key = lambda x: -x[1])
#print(results1)

#bow1.shape

vectorizer2 = TfidfVectorizer(sublinear_tf=True, max_df=1.0)
bow1_h = vectorizer2.fit_transform(processed_headline)

freqs1_h   = [(word, bow1_h.getcol(idx).sum()) for word, idx in vectorizer2.vocabulary_.items()]
results1_h = sorted (freqs1_h, key = lambda x: -x[1])





df_p = svd.fit_transform(bow1) 
df_p.shape


#print(results1)

df_h = svd.fit_transform(bow1_h) 

df_t= df_p
df_t[0][0:5]

df_t = pd.DataFrame(df_t)
df_h = pd.DataFrame(df_h)



df_t = min_max_scaler.fit_transform(df_t)
df_h = min_max_scaler.fit_transform(df_h)

df_t = pd.DataFrame(df_t)
df_h = pd.DataFrame(df_h)

df_final = df_t.join(df_h,lsuffix='_t',rsuffix='_h')

df_final.head()



url_test = 'https://raw.githubusercontent.com/BrunoFromMars/ZS/master/test_file_1.csv'

df_test = pd.read_csv(url_test)

list_source = df_new['Source'].tolist()

for i in range(df_test.shape[0]):
    list_source.append(df_test.Source[i])

len(list_source)




df_source = pd.DataFrame(list_source,columns=['Source'])

df_source = pd.get_dummies(df_source)

#df_source.shape

df_final = df_final.join(df_source.iloc[0:df_new.shape[0],:])

df_final.head()



df_final = df_final.join(pd.get_dummies(df_new['Topic']))

list_year, list_month, list_day, list_hour, list_min, list_sec = [],[],[],[],[],[] 


for i in range(df_new.shape[0]):
    string = df_new.PublishDate[i]
    date = datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S") 
    list_year.append(date.year)
    list_month.append(date.month)
    list_day.append(date.day)
    list_hour.append(date.hour)
    list_min.append(date.minute)
    list_sec.append(date.second)
    



df_date = pd.DataFrame(list(zip(list_year, list_month, list_day, list_hour, list_min, list_sec)), 
               columns =['list_year', 'list_month', 'list_day', 'list_hour', 'list_min', 'list_sec'])

df_date = min_max_scaler.fit_transform(df_date)

df_date = pd.DataFrame(df_date,columns =['list_year', 'list_month', 'list_day', 'list_hour', 'list_min', 'list_sec'])


df_final = df_final.join(df_date)

df_final.head()




df_sm = df_new[col_sm]

df_sm = min_max_scaler.fit_transform(df_sm)
df_sm = pd.DataFrame(df_sm,columns =col_sm)
df_final = df_final.join(df_sm)

df_final = df_final.join(df_new[['SentimentTitle','SentimentHeadline']])


