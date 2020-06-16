import re
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt
from tweepy import OAuthHandler
from wordcloud.tokenization import score

from config import consumer_key, consumer_secret, access_token, access_token_secret

#Authentication
try:
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
except:
    print("Error: Authentication Failed")

#the handle of the twitter user
twitter_handle = ''

#the number of tweets to generate(twitter allows a max of 200)
count = 200

posts = api.user_timeline(screen_name=twitter_handle, count=count, lang="en", tweet_mode="extended")

#clean tweets
def clean_tweet(tweet):
    tweet = re.sub('@[A-Za-z0–9]+', '', tweet)
    tweet = re.sub('#', '', tweet)
    tweet = re.sub('RT[\s]+', '', tweet)
    tweet = re.sub('https?:\/\/\S+', '', tweet)
    return tweet

#express tweets in data frame format and clean them
df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
df['Tweets'] = df['Tweets'].apply(clean_tweet)

# Create a function to get the subjectivity
def getSubjectivity(text):
   return TextBlob(text).sentiment.subjectivity

# Create a function to get the Polarity
def getPolarity(text):
   return  TextBlob(text).sentiment.polarity

#create a subjectivity and polarity column for the data frame
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)

#Polarity Score
def getAnalysis(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

#apply polarity score to dataframe
df['Analysis'] = df['Polarity'].apply(getAnalysis)


# Show the value counts
df['Analysis'].value_counts()

# Plotting and visualizing the counts
plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
df['Analysis'].value_counts().plot(kind='bar')
plt.figure(figsize=(10,8))
plt.show()

# word cloud visualization
allWords = ' '.join([twts for twts in df['Tweets']])
wordCloud = WordCloud(width=500, height=500, background_color='white', random_state=21, max_font_size=110).generate(allWords)

plt.figure(figsize=(8,8), facecolor=None)
plt.imshow(wordCloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()