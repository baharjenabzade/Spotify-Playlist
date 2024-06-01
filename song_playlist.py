#Data analysis
import pandas as pd
import numpy as np

#Data visualization
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
#import seaborn as sns

#Text pre-processing libraries
import re
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler,QuantileTransformer

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Libraries used to improve accuracy
from sklearn.metrics import accuracy_score, r2_score
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import TransformedTargetRegressor

#Colored text
from termcolor import colored


print(colored("Reading Dataset", "green"))

DataFrame = pd.read_csv("Spotify_Youtube.csv")

print(colored("Removing NULL values", "red"))
DataFrame.dropna(subset=['Title'], inplace=True)
DataFrame.dropna(subset=['Views'], inplace=True)
DataFrame.dropna(subset=['Stream'], inplace=True)

columns = DataFrame.columns
streams = DataFrame.Stream
views = DataFrame.Views

songFeatures = ["Danceability", "Energy", "Key", "Loudness", "Speechiness", "Acousticness", "Instrumentalness", "Liveness"]
features = DataFrame[songFeatures]
songInfo = ["Artist", "Title", "Album_type", "Views", "Stream", "Likes" , "Comments"]
songsData = DataFrame[songInfo]

scaler = MinMaxScaler()



"""
sortedSongs = DataFrame.sort_values(by = ["Stream"], ascending = False)
top10Streamed = sortedSongs[viewStream].head(10) 
sortedSongs = DataFrame.sort_values(by = ["Views"], ascending = False)
top10Viewed = sortedSongs[viewStream].head(10) 
"""

# Removing NULL values
DataFrame.dropna(subset=['Likes'], inplace=True)
DataFrame.dropna(subset=['Comments'], inplace=True)
DataFrame.dropna(subset=['Danceability'], inplace=True)
DataFrame.dropna(subset=['Energy'], inplace=True)
DataFrame.dropna(subset=['Key'], inplace=True)
DataFrame.dropna(subset=['Loudness'], inplace=True)
DataFrame.dropna(subset=['Speechiness'], inplace=True)
DataFrame.dropna(subset=['Acousticness'], inplace=True)
DataFrame.dropna(subset=['Liveness'], inplace=True)
DataFrame.dropna(subset=['Valence'], inplace=True)
DataFrame.dropna(subset=['Tempo'], inplace=True)
DataFrame.dropna(subset=['Duration_ms'], inplace=True)


def predict_album_type():
    model = LogisticRegression(fit_intercept = False, solver = "liblinear", 
                                     random_state = 42)
    print(colored("Splitting train and test dataset into 80:20", "blue"))
    DataFrame["ViewPlusStream"] = DataFrame.Stream + DataFrame.Views
    albumType = DataFrame["Album_type"]
    # Encode Album types with integers
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(albumType)

    # normalizing Dataset
    DataFrame[["Likes" , "Comments" , "ViewPlusStream" , "Duration_ms" , "Tempo"]] = scaler.fit_transform(DataFrame[["Likes","Comments", "ViewPlusStream","Duration_ms" ,"Tempo"]])

    # Drop unwanted labels
    # Some labels are deleted like Tempo and Duration and gave better accuracy
    attributes = DataFrame.drop(["Track", "Artist", "Album","Album_type", "Uri",
                              "Url_spotify", "Url_youtube", "Title", "Channel",
                               "ViewPlusStream" , "Views" , "Stream",
                               "Description", "Licensed", "official_video", "Tempo","Duration_ms"
                               ], axis = 1, inplace = False)
    #model = DecisionTreeRegressor()
    #regr_trans = TransformedTargetRegressor(regressor=model, transformer=QuantileTransformer(output_distribution='normal'))

    X_train, X_test, y_train, y_test = train_test_split(attributes, encoded_labels, test_size = 0.20, random_state = 42)

    # Code to remove extra features
    from sklearn.feature_selection import RFECV
    rfe = RFECV(LogisticRegression(fit_intercept = False, C = 1e12, solver = "liblinear", 
                               random_state = 42), cv = 5)
    X_rfe_train = rfe.fit_transform(X_train, y_train)
    X_rfe_test = rfe.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(y_pred , y_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy of Model :", accuracy)


scaler = StandardScaler()
def predict_popularity():
    model = LinearRegression()

    #Feature Selection
    sfs = SFS(LinearRegression(),
          k_features=10,
          forward=True,
          floating=False,
          scoring = 'r2',
          cv = 0)
    print(colored("Splitting train and test dataset into 80:20", "blue"))
    DataFrame["ViewPlusStream"] = DataFrame.Stream + DataFrame.Views

    #normalizing Dataset
    DataFrame[["Likes" , "Comments" , "ViewPlusStream" , "Duration_ms" , "Tempo"]] = scaler.fit_transform(DataFrame[["Likes","Comments", "ViewPlusStream","Duration_ms" ,"Tempo"]])
    totalViews = DataFrame["ViewPlusStream"]

    # Drop unwanted labels
    attributes = DataFrame.drop(["Track", "Artist", "Album","Album_type", "Uri",
                              "Url_spotify", "Url_youtube", "Title", "Channel",
                               "ViewPlusStream" , "Views" , "Stream",
                               "Description", "Licensed", "official_video", "Tempo","Duration_ms",
                               "Valence", "Key"], axis = 1, inplace = False)
    
    print(attributes)
    sfs.fit(attributes, totalViews)
    print("Most Key Features :", sfs.k_feature_names_)

    X_train, X_test, y_train, y_test = train_test_split(attributes, totalViews, test_size = 0.20, random_state = 42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(colored("Score of Model :", "red"))
    print( model.score(X_test, y_test))






def show_top_artist():
    songsData["StreamXViews"] = songsData.Stream * songsData.Views 
    songInfo = ["Artist", "Title", "Album_type", "Views", "Stream", "StreamXViews"]
    sortedSongs = songsData.sort_values(by = ["StreamXViews"], ascending = False)
    top5StreamXView = sortedSongs[songInfo].head(5) 
    top5PlayedArtists = top5StreamXView.Artist
    print(top5PlayedArtists)
    top100PopularSongs = sortedSongs[songInfo].head(100) 
    singleCount = top100PopularSongs['Album_type'].value_counts()['single']
    albumCount = top100PopularSongs['Album_type'].value_counts()['album']
    if (albumCount > singleCount):
        print("Albums are more popular")
    else:
        print("Singles are more popular")



def show_relation_likes_comments():
    
    figure, (ax0, ax1) = plt.subplots(nrows = 2 , figsize = (10,10))
    ax0.scatter(DataFrame["Likes"], DataFrame["Views"])
    ax0.set_title("Likes vs Views")
    ax1.scatter(DataFrame["Comments"], DataFrame["Views"])
    ax1.set_title("Comments vs Views")
    plt.show()

def show_relation_views_streams():
    plt.scatter(DataFrame["Stream"], DataFrame["Views"])
    plt.xlabel("Streams")
    plt.ylabel("Views")
    plt.show()

def show_relation_popularity():
    # Relation between popularity and attributes of song
    DataFrame.drop(["Track", "Artist", "Album","Album_type", "Uri",
                                    "Url_spotify", "Url_youtube", "Title", "Channel"
                                    , "Views" , "Stream",
                                    "Description", "Licensed", "official_video"], axis = 1, inplace = True)
    fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(20,30), sharey=True)
    print(DataFrame.columns[5:15])
    for ax, column in zip(axes.flatten(), DataFrame.columns):
        ax.scatter(DataFrame[column], DataFrame['ViewPlusStream'], label=column, alpha=1)
        ax.set_title(f'Popularity vs {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Popularity')

    fig.tight_layout()
    plt.show()



def boost_models(x):
    #transforming target variable through quantile transformer
    regr_trans = TransformedTargetRegressor(regressor=x, transformer=QuantileTransformer(output_distribution='normal'))

    regr_trans.fit(X_train, y_train)
    yhat = regr_trans.predict(X_test)
    algoname= x.__class__.__name__
    return algoname, round(r2_score(y_test, yhat),3), round(mean_absolute_error(y_test, yhat),2), round(np.sqrt(mean_squared_error(y_test, yhat)),2)

def menu():

    print("1. Show top artists")
    print("2. Show relation between popularity and different attributes of a song")
    print("3. Show relation between likes, comments and views")
    print("4. Predict popularity")
    print("5. Predict album type")
    print("6. Show relation between streams and views")
    print("7. Exit")

    userInput = int(input("Choose :"))
    while userInput != 6 :
        if userInput == 1:
            show_top_artist()
            break
        if userInput == 2:
            show_relation_popularity()
            break
        if userInput == 3:
            show_relation_likes_comments()
            break
        if userInput == 4:
            predict_popularity()
            break
        if userInput == 5:
            predict_album_type()
            break
        if userInput == 6:
            show_relation_views_streams()
            break
        if userInput == 7:
            break

menu()