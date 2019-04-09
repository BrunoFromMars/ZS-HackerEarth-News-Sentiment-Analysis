
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

#import re


import datetime



url_test = 'https://raw.githubusercontent.com/BrunoFromMars/ZS/master/test_file_1.csv'

df_test = pd.read_csv(url_test)

list_source = df_new['Source'].tolist()

for i in range(df_test.shape[0]):
    list_source.append(df_test.Source[i])

len(list_source)




df_source = pd.DataFrame(list_source,columns=['Source'])

df_source = pd.get_dummies(df_source)



processed_title_test = []

for i in range(df_test.shape[0]):
    t = df_test.Title[i].lower().split()
    #h = df_test.Headline[i].lower().split()
    #print(t)
    #print(h)
    #negation handling
    reformed_t = [appos[word] if word in appos else word for word in t]
    t = " ".join(reformed_t) 
    #reformed_h = [appos[word] if word in appos else word for word in h]
    #h = " ".join(reformed_h)
    #tokenize
    t = word_tokenize(t)
    #h = word_tokenize(h)
    #alpha numeric
    words_t = [word for word in t if word.isalpha()]
    #words_h = [word for word in h if word.isalpha()]
    #stop-words
    filtered_t = [w for w in words_t if not w in stop_words]
    #filtered_h = [w for w in words_h if not w in stop_words]
    #stemming
    t =""
    #h =""
    for w in filtered_t:
        t += ps.stem(w) + " "
    #for w in filtered_h:
    #    h += ps.stem(w) + " "
        
    processed_title_test.append(t)
    #processed_headline_test.append(h)

bow1_test = vectorizer1.fit_transform(processed_title_test)

df_t_test = svd.fit_transform(bow1_test)

df_t_test = pd.DataFrame(df_t_test)

df_t_test = min_max_scaler.fit_transform(df_t_test)

df_t_test = pd.DataFrame(df_t_test)

df_final_test = df_t_test

df_final_test = df_final_test.join(df_source.iloc[df_new.shape[0]:,:])

df_final_test = df_final_test.join( pd.get_dummies(df_test['Topic']))

list_year, list_month, list_day, list_hour, list_min, list_sec = [],[],[],[],[],[]



for i in range(df_test.shape[0]):
    string = df_test.PublishDate[i]
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

df_final_test = df_final_test.join(df_date)

col_sm = ['Facebook', 'GooglePlus', 'LinkedIn']
df_sm = df_test[col_sm]

df_sm = min_max_scaler.fit_transform(df_sm)
df_sm = pd.DataFrame(df_sm,columns =col_sm)
df_final_test = df_final_test.join(df_sm)

