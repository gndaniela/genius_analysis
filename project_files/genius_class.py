# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:51:37 2021

@author: gndaniela
"""

#%% Libraries


import lyricsgenius as genius
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import string 
from string import punctuation
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'


import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from pandas import json_normalize
from collections import Counter
from nltk.sentiment.vader import SentimentIntensityAnalyzer
#nltk.download('punkt')
#nltk.download('stopwords')


#%% Class to obtain sentiment analysis from band/artist lyrics

class SentimentAnalysis():
    def __init__(self):

        self.genius_access_token = 'your_genius_token'
        self.client_id = 'your_client_id'
        self.client_secret = 'your_client_secret'

    def apis_authentication(self):
        manager = SpotifyClientCredentials(self.client_id,self.client_secret)
        sp = spotipy.Spotify(client_credentials_manager=manager)
        ge = genius.Genius(self.genius_access_token)
        
        return sp, ge

    def lyrics_to_clean_text(self,lyric,language='english',add_stop_words=[],lemmatized=True):
                """
                Removes stop words and punctuation from a text
                Receives:
                    -lyric: text to clean
                    -language: language to retrieve stop words
                    -add_stop_words: an optional list in case you need to add words to remove
                    -lemmatized: True/False. While True groups together variant forms of the same word.
                    (E.g: "builds", "building" or "built" to the lemma "build")
                --------    
                Returns:
                    -lyrics lemmatized/not lemmatized: a text after transformations
                """
                new_stop_words = ['urlcopyembedcopy','hes','i',"n't","'s","'t","'m","'re","’","'ll","chorus","verse","v.o","instrumental","intro","solo","1","2","3","4","5","6"]
                new_stop_words.extend(add_stop_words)
                stop_words = set(stopwords.words(language))
                [stop_words.add(i) for i in new_stop_words]
                exclude = set(string.punctuation)
                lemma = WordNetLemmatizer()
                stopwordremoval = " ".join([i for i in lyric.lower().split() if i not in stop_words])
                punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude) #check punctuation char by char
                final_cleanse = " ".join([i for i in punctuationremoval.lower().split() if i not in stop_words])
                
                if lemmatized:
                    return_text = " ".join(lemma.lemmatize(word) for word in final_cleanse.split())
                else:
                    return_text = final_cleanse
                
                return return_text


    def lyrics_to_words(self,lyric,language='english',add_stop_words=[],lemmatized=True):
                """
                Removes stop words and punctuation from a text
                Receives:
                    -lyric: text to clean
                    -language: language to retrieve stop words
                    -add_stop_words: an optional list in case you need to add words to remove
                    -lemmatized: True/False. While True groups together variant forms of the same word.
                    (E.g: "builds", "building" or "built" to the lemma "build")
                --------    
                Returns:
                    -words lemmatized/not lemmatized: a list of words after transformations
                """
                new_stop_words = ['urlcopyembedcopy','i',"n't","'s","'t","'m","'re","’","'ll","chorus","verse","v.o","instrumental","intro","solo","1","2","3","4","5","6"]
                new_stop_words.extend(add_stop_words)
                stop_words = set(stopwords.words(language))
                [stop_words.add(i) for i in new_stop_words]
                exclude = set(string.punctuation)
                lemma = WordNetLemmatizer()
                stopwordremoval = " ".join([i for i in lyric.lower().split() if i not in stop_words])
                punctuationremoval = ''.join(ch for ch in stopwordremoval if ch not in exclude) #check punctuation char by char
                final_cleanse = " ".join([i for i in punctuationremoval.lower().split() if i not in stop_words])
                
                #return_words = []
                if lemmatized:
                    return_words = [lemma.lemmatize(word) for word in final_cleanse.split()]
                else:
                    return_words = [word for word in final_cleanse.split()]
                
                return return_words
    
    
    def plot_topn_words(self,words_list,top_n=15):
        
        """
        Frequency plot
        Receives:
            -words_list 
            -top_n: top n words to plot
        Returns:
            -bar plot
        """
        frequency_words = Counter(words_list)
        
        frequency_df = pd.DataFrame({'k':list(dict(frequency_words).keys()),'v':list(dict(frequency_words).values())})
        frequency_df = frequency_df[~frequency_df['k'].str.lower().duplicated(keep='first')] #remove duplicates
        frequency_df.sort_values('v',ascending=False, inplace=True)
       
        count = np.arange(len(frequency_df['v'][0:top_n]))
        plt.style.use('seaborn-white')
        plt.bar(count, frequency_df['v'][0:top_n], width=0.85, color='mediumturquoise', alpha=0.9)
        plt.ylabel('Frequency', fontsize=10)
        plt.xticks(count, frequency_df['k'][0:top_n], rotation=60)
        plt.title('Top {} - Most used words'.format(top_n), fontsize=15)
        plt.show()




    def search_artist_data(self,artist_name):
        
        """
        Receives:
            -artist_name: artist or band to search 
        Returns:
            -df with Spotify songs & audio features ,Genius lyrics for all artist's albums available tracks
            & sentiment analysis components of lyrics
        """
        
        sp, ge = self.apis_authentication()
        #Search artist in Spotify
        artist_search = sp.search(q=artist_name, type="artist",limit=10)  # returns a dictionary
        artist_search = artist_search['artists']
        artists_list = artist_search['items']
        artists_df = json_normalize(artists_list)
        #Get artist's albums
        albums_search = sp.artist_albums(artists_df['uri'][0], album_type='album', limit=50)  # returns a dictionary
        albums_list = albums_search['items']  # returns a list
        # Goes through pagination. As long as there's a next page, it will retrieve all items
        while albums_search['next']:
            albums_search = sp.next(albums_list)
            albums_list.extend(albums_search['items'])
        albums_df = json_normalize(albums_list)
        albums_df = albums_df[albums_df['type']=='album']
        albums_df.drop_duplicates(subset=['name','total_tracks'],inplace=True)
        albums_df.sort_values('release_date',inplace=True)
        albums_df.drop_duplicates('name', keep='first',inplace=True)
        
        
        albums_df_clean = albums_df.drop(['artists', 'available_markets', 'images', 'type'], axis=1)
        #Get albums' tracks
        tracks_search = {}
        tracks_df = pd.DataFrame()
        
        for i in albums_df_clean['id']:
            tracks_search = sp.album_tracks(i)  # returns a dictionary
            temp = json_normalize(tracks_search['items'])
            temp['album_id'] = i
            tracks_df = pd.concat([tracks_df, temp])

        artist_id = []
        artist_name = []
        
        for i in tracks_df['artists']:
            artist_id.append(i[0]['id'])
            artist_name.append(i[0]['name'])
        
        tracks_df['artist_id'] = artist_id
        tracks_df['artist_name'] = artist_name
        
        tracks_df_clean = tracks_df.drop(['artists', 'available_markets', 'is_local', 'type', 'preview_url'], axis=1)
        tracks_df_clean = pd.merge(albums_df_clean[['id', 'name']], tracks_df_clean, left_on='id', right_on='album_id')
        tracks_df_clean.drop('id_x', inplace=True, axis=1)
        tracks_df_clean.rename(columns={'name_x': 'album_name', 'id_y': 'track_id', 'name_y': 'track_name'}, 
                               inplace=True)
        tracks_df_clean.drop_duplicates('track_name'.lower(), inplace=True)
        #Get tracks' audio features
        dicts = []
        for i in tracks_df_clean['track_id']:
            dic = sp.audio_features(i)[0]
            dicts.append(dic)
        
        tracks_af_df = json_normalize(dicts)
        #Get Track Popularity
        track_pop = []
        track_id = []
        
        for i in tracks_df_clean['track_id']:
            track_pop.append(sp.track(i)['popularity'])
            track_id.append(sp.track(i)['id'])

        popularity_df = pd.DataFrame({'popularity': track_pop, 'track_id': track_id})
        #Prepare final df for analysis
        
        base_df = pd.merge(tracks_df_clean[['artist_id', 'artist_name', 'album_id', 'album_name',
                                           'track_id', 'track_name', 'track_number']], tracks_af_df, left_on='track_id', 
                          right_on='id')
        
        base_df = base_df.drop(['type', 'id',
                              'uri', 'track_href', 'analysis_url'], axis=1)
        
        # add track populatiry
        base_df = pd.merge(base_df, popularity_df)
        
        # add album's release date
        
        base_df = pd.merge(base_df, albums_df_clean[['release_date','id']],left_on='album_id', right_on='id')
        base_df = base_df.drop('id', axis=1)
        base_df['album_release_date'] = pd.to_datetime(base_df['release_date'])
        base_df['year'] = base_df['album_release_date'].apply(lambda x:x.year)
        base_df = base_df.drop('release_date', axis=1)
        base_df['duration_min'] = base_df['duration_ms'].apply(lambda x: round(x/60000,2))
        
        genius_songs = []
        for i in base_df.index:
            search = ge.search_song(base_df['track_name'].iloc[i],base_df['artist_name'].iloc[i])
            genius_songs.append(search)
        
        list_lyrics = []
        list_title = []
        
        for song in genius_songs:
            if song is not None:
                list_lyrics.append(song.lyrics)
                list_title.append(song.title)

        genius_df = pd.DataFrame({'title':list_title, 'lyric':list_lyrics})
        
        base_df = pd.merge(base_df, genius_df,left_on='track_name', right_on='title')
        
        base_df['lemmalyrics'] = base_df['lyric'].apply(lambda x: self.lyrics_to_clean_text(x))
        base_df['unique_words'] = base_df['lemmalyrics'].apply(lambda x: list(set(x.split()))) 
        base_df['unique_words_count'] = base_df['unique_words'].apply(lambda x: len(x)) 
        base_df['total_words_count'] = base_df['lemmalyrics'].apply(lambda x: len(list(x.split())))
        base_df['decade'] = base_df['year'].apply(lambda x: "70s" if x < 1980 and x >= 1970
                                else "80s" if x < 1990 and x >= 1980
                                else "90s" if x < 2000 and x >= 1990
                                else "00s" if x < 2010 and x >= 2000
                                else "10s" if x < 2020 and x >= 2010
                                else "20s" if x < 2030 and x >= 2020
                                else "")

       #Sentiment Analysis using VADER Sentiment Intensinty Model
       #Create lists to store the different scores for each word
        negative = []
        neutral = []
        positive = []
        compound = []
            
       #Initialize the model
        sid = SentimentIntensityAnalyzer()
            
       #Iterate for each row of lyrics and append the scores
        for i in base_df.index:
               if len(base_df['lemmalyrics'].iloc[i]) > 0: 
                 scores = sid.polarity_scores(base_df['lemmalyrics'].iloc[i])
                 negative.append(scores['neg'])
                 neutral.append(scores['neu'])
                 positive.append(scores['pos'])
                 compound.append(scores['compound'])
               else: 
                 negative.append(None)
                 neutral.append(None)
                 positive.append(None)
                 compound.append(None)
                
                #Create 4 columns to the main data frame  for each score 
        base_df['negative'] = negative
        base_df['neutral'] = neutral
        base_df['positive'] = positive
        base_df['compound'] = compound
                 
        return base_df

        #Plot all songs by sentiment
    def plot_lyrics_sentiment(self,df):
        """
        Receives:
            -artist complete df
        --------    
        Returns: 
            -scatter plot with songs located according to their lyrics' sentiment analysis 
            -size corresponds to unique words per song
        """
        fig = px.scatter(df, x="positive", y="negative", color="album_name",
                         size='unique_words_count', hover_data=None, hover_name='title', #,text='title'
                         labels={'album_name':'Album','unique_words_count':'Total words','negative':'Negative','positive':'Positive'},
                         title="Sentiment by Genius Lyrics Analysis")
        fig.show()
        
        #Plot all songs by audio features
    def plot_tracks_sentiment(self,df):   
        """
        Receives:
            -artist complete df
        --------    
        Returns: 
            -scatter plot with songs located according to their audio features sentiment analysis 
            -size corresponds to unique words per song
        """
        fig = px.scatter(df, x="valence", y="energy", color="album_name",
                 size='duration_min', hover_data=None, hover_name='title',
                 labels={'album_name':'Album','duration_min':'Duration (Min)','energy':'Energy','valence':'Valence'},
                 title="Sentiment by Spotify's Audio Features Analysis")
        fig.add_hline(y=df.energy.mean(), opacity=0.6,line_dash="dash",annotation_text='Mean energy')
        fig.add_vline(x=df.valence.mean(), opacity=0.6,line_dash="dash",annotation_text='Mean valence')
        fig.add_annotation(x=0.01, y=1.01, text="Turbulent/Angry", showarrow=False,opacity=0.7, font=dict(color="green",size=14))
        fig.add_annotation(x=0.9, y=1.01, text="Happy/Joyful",showarrow=False, opacity=0.7, font=dict(color="green",size=14))
        fig.add_annotation(x=0.01, y=0.02, text="Sad/Depressing",showarrow=False, opacity=0.7, font=dict(color="green",size=14))
        fig.add_annotation(x=0.9, y=0.02, text="Chill/Peaceful",showarrow=False, opacity=0.7, font=dict(color="green",size=14))
        fig.show()
  
    def stats_dfs(self,df):
        
        """
        Receives:
            -artist complete df
        --------    
        Returns: (unpack as 2 dfs)
            -df with words analysis per decade
            -df with words analysis per album
        """
        
        #df = self.df_for_stats(artist_name, top_n_songs)
        words = []
        decades = []
        decades_df = pd.DataFrame()
        albums = []
        albums_df = pd.DataFrame()
    
        #Create a DF with every word of every lyric and the decade it belongs to
        for i in df.index:    
            for word in df['unique_words'].iloc[i]:
                words.append(word)
                decades.append(df['decade'].iloc[i])
                albums.append(df['album_name'].iloc[i])
    
    
        decades_df = pd.DataFrame({'words':words,'decade':decades})
        decades_list = list(set(decades_df['decade'].tolist()))
        
        albums_df = pd.DataFrame({'words':words,'album':albums})
        albums_list = list(set(albums_df['album'].tolist()))
    
        unique_words = []
        total_words = []
        total_songs = []
    
        for i in decades_list:
            unique_words.append(len(set(decades_df['words'][decades_df['decade']==i])))
            total_words.append(len(decades_df['words'][decades_df['decade']==i]))
            total_songs.append(df[df['decade']==i]['title'].count())
    
        decades_stats = pd.DataFrame({'decade':decades_list,'unique_words':unique_words,'total_words':total_words,
                                 'total_songs':total_songs})
        decades_stats['words_per_song'] = round( decades_stats['total_words']/ decades_stats['total_songs'],0)
        
        
        unique_words_al = []
        total_words_al = []
        total_songs_al = []
        
        for i in albums_list:
            unique_words_al.append(len(set(albums_df['words'][albums_df['album']==i])))
            total_words_al.append(len(albums_df['words'][albums_df['album']==i]))
            total_songs_al.append(df[df['album_name']==i]['title'].count())
        
        albums_stats = pd.DataFrame({'album':albums_list,'unique_words':unique_words_al,'total_words':total_words_al,
                                 'total_songs':total_songs_al})
        albums_stats['words_per_song'] = round(albums_stats['total_words']/albums_stats['total_songs'],0)
        
        return decades_stats, albums_stats
    
        
   


