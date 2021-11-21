# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 21:53:00 2021

@author: dgnistor
"""


from project_files import SentimentAnalysis


sa = SentimentAnalysis()

#Get Artist complete df
#E.g.: Adele
adele_df = sa.search_artist_data('Adele')

#Get words for all artist's songs
adele_words = []
for i in adele_df['lemmalyrics']:
    adele_words.extend(sa.lyrics_to_words(i))
    
#Plot top N used words
sa.plot_topn_words(adele_words,20)
#Plot songs by sentiment, according to songs' lyrics (Genius)
sa.plot_lyrics_sentiment(adele_df)
#Plot songs by sentiment, according to songs' audio features (Spotify)
sa.plot_tracks_sentiment(adele_df)
#Get dfs with words stats by decade and album
adele_decades, adele_albums = sa.stats_dfs(adele_df)