from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import tweepy
import json
import re
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import random
from flask import Flask,render_template,url_for,request
import numpy as np
import emoji
from keras.models import load_model
app = Flask(__name__)

@app.route('/')
def hello(st=''):
    print("HOME")
    return render_template('home.html',title='home')

@app.route('/analysis',methods=['POST','GET','OPTIONS'])   
def analysis():
    
    #Taking Input into variable key
    key=request.form['InputText']
    
    #Setting up keys to access twitter data
    consumer_key = '9lMcBOT8L4NIVaZGPjGYd5Hpw'
    consumer_secret = 'n6zdHxzQf9IoZRAoqjplpRvZ66poE7itfw4OhQWCJXBZfzK7Ki'
    access_token = '4866812614-UdFFNvUp1CnV0tAT3WyWSJdlaLLrjiHVJ5y7w0f'
    access_token_secret = 'Wrfm0zbzIIQCH3aCFTiIq38oi8TYcZNgWkaCTWQ7J5M06'

    #Interacting with twitter's API
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret) 
    auth.set_access_token(access_token, access_token_secret)
    #creating the API object
    api = tweepy.API (auth) 
    
    #extracting tweets
    results = [] #Array to Store fetched results
    num = 50 #Variable to indicte number of tweets to be fetched 
    for tweet in tweepy.Cursor (api.search, q = key, lang = "en").items(50): 
        results.append(tweet)
    
    #Creating dataframe to capture tweet information
    dataset=pd.DataFrame()
    dataset["tweet_id"]=pd.Series([tweet.id for tweet in results])
    dataset["username"]=pd.Series([tweet.author.screen_name for tweet in results])
    dataset["text"]=pd.Series([tweet.text for tweet in results])
    dataset["followers"]=pd.Series([tweet.author.followers_count for tweet in results])
    dataset["hashtags"]=pd.Series([tweet.entities.get('hashtags') for tweet in results])
    dataset["emojis"]=pd.Series([''.join(c for c in tweet.text if c in emoji.UNICODE_EMOJI) for tweet in results])


    
    #WordCloud of hashtags
    Hashtag_df = pd.DataFrame(columns=["Hashtag"])
    j = 0
    for tweet in range(0,len(results)):
        hashtag = results[tweet].entities.get('hashtags')
        for i in range(0,len(hashtag)):
            Htag = hashtag[i]['text'] 
            Hashtag_df.at[j,'Hashtag']=Htag
            j = j+1
    Hashtag_Combined = " ".join(Hashtag_df['Hashtag'].values.astype(str))
    text=" ".join(dataset['text'].values.astype(str))
    cleaned_text = " ".join([word for word in text.split()
                                    if word !="https"
                                    and word !="RT"
                                    and word !="co"
                                                                    
                                    ])                         
    wc = WordCloud(width=500,height=500,background_color="white", stopwords=STOPWORDS).generate(Hashtag_Combined)
    plt.imshow(wc)
    plt.axis("off")
    r =random.randint(1,101)
    st = 'G:\Opinion Mining And Analysis\webApp\static\hashtag'+ str(r) +'.png'
    plt.savefig(st, dpi=300) 

    #Creating tags list
    hashtag=Hashtag_Combined.split(" ")
    df=pd.DataFrame()
    df1=pd.DataFrame()
    df['hashtags']=pd.Series([i for i in hashtag])
    data=df['hashtags'].value_counts()
    df1['hashtags']=df['hashtags'].unique()
    df1['ccounts']=pd.Series([i for i in data])
    tag_count_list = df1.iloc[:5,1].values.tolist()    
    tag_list = df1.iloc[:5,0].values.tolist()

    
    #Section for Creating DataFrame column For Sentiments. senModelList is a list containing results from the model. senList is a list containing final sentimient values.

    #fetching vocab from .npy file
    x=np.load('tokens.npy')
    tk=Tokenizer()
    tk.fit_on_texts(x)
    sentDataFrame = dataset.copy(deep=True)

    #preprocessing data
    #removing links, hashtags,and twitter handles
    sentDataFrame['text']=sentDataFrame['text'].apply(lambda x:' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x).split()))

    #coverting to lowercase
    sentDataFrame["text"]=sentDataFrame["text"].apply(lambda x: x.lower())

    #removing punctuation
    sentDataFrame["text"]=sentDataFrame["text"].apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))

    #removing 'rt'
    sentDataFrame["text"]=sentDataFrame["text"].apply(lambda x: x.replace('rt',''))

    #Saving as csv

    
    #Prediction
    df1=pd.DataFrame()
    df1['text']=dataset['text']

    tweet_tokens = tk.texts_to_sequences(dataset['text'].values)
    tweet_tokens_pad = pad_sequences(tweet_tokens, maxlen=40)

    #predict
    f_model = load_model('final_model.h5')
    senModelList=f_model.predict(x=tweet_tokens_pad)
    senList = []
    for i in range(num):
        if(senModelList[i][0]>senModelList[i][1]):
            senList.append('n')
        else:
            senList.append('p')
    dataset['sentiment'] = pd.Series(senList)
    #Section for counting percentage of sentiments.
    #posSentPer = (dataset[dataset['sentiment']=='p'].shape[0]/num)*100
    #negSentPer = (dataset[dataset['sentiment']=='n'].shape[0]/num)*100
    posSentPer = len(dataset[dataset['sentiment']=='p'].sentiment)
    negSentPer = len(dataset[dataset['sentiment']=='n'].sentiment)
    opList = [posSentPer,negSentPer] 
    
    #Section for Calculating Visibility Percentage
    pos_dataset_for_visibility = dataset[dataset['sentiment']=='p'] 
    posVis = pos_dataset_for_visibility['followers'].sum(axis=0,skipna=True)

    neg_dataset_for_visibility = dataset[dataset['sentiment']=='n'] 
    negVis = neg_dataset_for_visibility['followers'].sum(axis=0,skipna=True)

    vbList = [posVis,negVis]

    #Fourth Grid for displaying tweets and its related data
    tw_uname = dataset['username'].values.tolist()
    tw_text = dataset['text'].values.tolist()
    tw_text2 = sentDataFrame['text'].values.tolist()
    tw_foll = dataset['followers'].values.tolist()
    return render_template('analysis.html',title='analysis',tw_text2=tw_text2,vbList=vbList,key=key,r=r,tag_list = tag_list,opList=opList,tag_count_list=tag_count_list,tw_uname=tw_uname,tw_text=tw_text,tw_foll=tw_foll)