# importing all the required Libraries
import json
import string
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

file = '2.json'

# Importing the data as pandas DataFrame.
review = []
with open(file, encoding="utf8", errors="ignore") as data_file:
    data = data_file.read()
    for i in data.split('\n'):
        review.append(i)

reviewDataframe = []
for x in review:
    try:
        jdata = json.loads(x)
        reviewDataframe.append((jdata['overall'], jdata['verified'], jdata['reviewTime'], jdata['reviewerID'],
                                jdata['asin'], jdata['reviewerName'], jdata['reviewText'], jdata['summary'],
                                jdata['unixReviewTime']))
    except:
        pass

    # Creating a dataframe using the list of Tuples got in the previous step.
dataset = pd.DataFrame(reviewDataframe,
                       columns=['Rating', 'Verified', 'Review_Time', 'Reviewer_ID', 'Asin', 'Reviewer_Name',
                                'Review_Text', 'Summary', 'Unix_Review_Time'])


# Function to calculate sentiments using Naive Bayes Analyzer
def NaiveBaiyes_Sentimental(sentence):
    blob = TextBlob(sentence, analyzer=NaiveBayesAnalyzer())
    NaiveBayes_SentimentScore = blob.sentiment.classification
    return NaiveBayes_SentimentScore


# Function to calculate sentiments using Vader Sentiment Analyzer
# VADER sentiment analysis tool for getting Compound score.
def sentimental(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score = vs['compound']
    return score


# VADER sentiment analysis tool for getting positive , negative and neutral.
def sentimental_Score(sentence):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    score = vs['compound']
    if score >= 0.5:
        return 'pos'
    elif (score > -0.5) and (score < 0.5):
        return 'neu'
    elif score <= -0.5:
        return 'neg'

#Selected_Rows = dataset
Selected_Rows=dataset.head(100000)
Selected_Rows['Sentiment_Score'] = Selected_Rows['Review_Text'].apply(lambda x: sentimental_Score(x))

pos = Selected_Rows.loc[Selected_Rows['Sentiment_Score'] == 'pos']
neg = Selected_Rows.loc[Selected_Rows['Sentiment_Score'] == 'neg']

pos.to_csv('Positive1.csv')
neg.to_csv('Negetive1.csv')

def stemming(tokens):
    ps = PorterStemmer()
    stem_words = []
    for x1 in tokens:
        stem_words.append(ps.stem(x1))
    return stem_words


# To Generate a word corpus following steps are performed inside the function 'Word_Corpus(df)'

# Iterating over the 'summary' section of reviews such that we only get important content of a review.
# Converting the content into Lowercase.
# Using nltk.tokenize to get words from the content.
# Using string.punctuation to get rid of punctuations.
# Using stopwords from nltk.corpus to get rid of stopwords.
# Stemming of Words.
# Finally forming a word corpus and returning the word corpus.

def Word_Corpus(df):
    words_corpus = ''
    for val in df["Summary"]:
        text = val.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = stemming(tokens)
        for words in tokens:
            words_corpus = words_corpus + words + ' '
    return words_corpus

def plot_Cloud(wordCloud):
    plt.figure(figsize=(20, 10), facecolor='k')
    plt.imshow(wordCloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()
    plt.savefig('wordclouds.png', facecolor='k', bbox_inches='tight')

pos_wordcloud = WordCloud(width=600, height=400, max_font_size=70).generate(Word_Corpus(pos))
neg_wordcloud = WordCloud(width=600, height=400, max_font_size=70).generate(Word_Corpus(neg))

# plot_Cloud(pos_wordcloud)
# plot_Cloud(neg_wordcloud)
