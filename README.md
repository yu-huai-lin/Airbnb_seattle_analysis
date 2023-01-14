
#  Sentiment Analysis and NLP in Python with Airbnb Seatle Open Data
![](/images/airbnb.png)

## Udacity Data Science Nano Degree Project
## Motivation

Traveller's reviews and comments on a accomodation platform like Airbnb can often be a very good source of truth that come from the actual customers experiences and they affect our decisions on which B&B and neighbourhood to stay.  We are gonna utilize the review data to do text and sentiment analysis and uncover insights in the 80000+reveiws in the Airbnb Seattle Data sets. 

In this notebook, I will explore the Airbnb Seattle Datasets and anwser 3 business questions:
- Do people tend to leave comment about bad reviews or good reviews? Are the sentiment differ from neighbourhoods?
- Are the positive sentiment correlated with other features/metrics?
- Can you describe the vibe of each Seattle neighborhood based on customer's review?

## Overview

The analysis is done using explorative data analysis and unsupervised sentiment analysis using a pre-trained model called Vader and NLTK to do text tagging and extraction.

`Airbnb Analysis.ipynb` contains all the codes used for the analysis.
`data` folder contains all the data sets used
`image` folder includes all the pictures and charts generated from the analysis.

For non-technical reader, please also read the findings in this medium blog: https://medium.com/@yuhuailin0323/insights-behind-the-airbnb-customer-reviews-sentiment-analysis-with-vader-and-nlp-in-python-45aba7b60367

### Requirement
numpy               1.21.4
pandas              1.3.2
nltk                3.8.1
vaderSentiment      3.3.2
plotly              5.2.1
wordcloud           1.8.2.2


## Data

Seattle Airbnb Open data
https://www.kaggle.com/datasets/airbnb/seattle

| Data file Name | Description|
|:----|:-----------|
| listings.csv | Metadata for each bnb listing and attributes like geographical information, host related information etc.|
| reviews.csv| Contains date, customer's name and their comments on the listing and experiences |
| calendar.csv| Contains date and information on whether a hotel is available; if not, the price that is booked with |


## What is Sentiment Analaysis?

> Sentiment Analysis is a use case of Natural Language Processing (NLP) and comes under the category of text classification. To put it simply, Sentiment Analysis involves classifying a text into various sentiments, such as positive or negative, Happy, Sad or Neutral, etc. Thus, the ultimate goal of sentiment analysis is to decipher the underlying mood, emotion, or sentiment of a text.
https://www.analyticsvidhya.com/blog/2022/07/sentiment-analysis-using-python/

### VADER
In this notebook, I'm utlizing a pre-trained model called Vader that performs sentiment analysis on text data. VADER (Valence Aware Dictionary and sEntiment Reasoner) is sentiment analyzer that has been trained on social media text. 
In this case, the text data will be Seattle Airbnb's customer reviews (comments) data.

VADER’s SentimentIntensityAnalyzer() takes in a string and returns a dictionary of scores in each of the following categories:
- Positive
- Negative
- Neutral
- compound (the sum of positive, negative & neutral scores which is then normalized between -1(strongly negative) and +1 (strongly positive). We will mostly use the compound score for the following analysis as the compound score gives us a single measure of sentiment for a given sentence.

#### Rule based Sentiment Classification using the predicted score
Based on the sentiment score, we classify the type of customer review sentiment into positive, negative and neutral.

The classification is made by this common rules:
- positive sentiment: compound score >= 0.05
- neutral sentiment: (compound score > -0.05) and (compound score < 0.05)
- negative sentiment: compound score <= -0.05

### Natural Language Toolkit (NLTK)
To answer the 3rd question on describing Seattle neighbourhood vibes, I'm going to use Natural Language Toolkit (NLTK) to process the text of the customer reviews and find the most popular adjectives to understand each Seattle neighbourhood's vibe.
NLYK is very useful in categorizing and tagging Words.


## Insights

### 1.  Do people tend to leave comment about bad reviews or good reviews? Are the sentiment differ from neighbourhoods?

![](/images/review_count_pie_chart.png)


Based on the analysis, it is indicated that the majority of the reviews are positive, and the distribution doesn't vary a lot from the top 5 neighbourhoods in Seattle


![](/images/review_color_scale.png)
If we now include the factors of the number of reviews into the chart, the number of reviewes are skewed to a few top neighbourhoods and a lot of the neighbourhoods have very few reviews. I put the positive sentiment score on a color scale, and the colors are very similar across different neighbourhoods.


### 2.  Are the positive sentiment correlated with host response time and other metrics, say host's experience?

#### Price

![](/images/price_bins.png)
If we plot the box plot for the positive sentiment score based on different price bins (a discrete feature I created based on the range of actual prices), we can see that, the mean of the sentiment score increases as price range goes up. This means that people were happier about the B&B they stayed when the prices were higher.

Lets' plot the price and the positive sentiment score on a scatter chart, with color distinction on whether the host is a super host:
![](/images/price_scatter.png)
The relationship between price and sentiment score seems to be more or less random and has a lot of noises.

![](/images/price_scatter_3.png)

Let's look at a feature called Host Response time in the data and its corresponding positive sentiment score distribution:
![](/images/host_response_time.png)
It seems like one of the key that affect customers' happiness is to keep the response time within one day :).
![](/images/host_response_rate.png)

![](/images/host_is_super_host.png)

#### Neighbourhood and Is_super_host
We can see that if the host has a super host badge, they seem to have a higher share of the positive comments across different neighbourhoods:

![](/images/super_host.png)



#### 3. Can you describe the vibe of each Seattle neighborhood based on customer's review?
Before answering this question, we will need to clean up the text data. This process includes tokenization (split the text into words), removing the punctuation and stop words, lemmatization (transform every word into their root form), and lastly, part-of-speech tagging ( the process of identifying the words as verbs, nouns, adjectives etc).

Based on the text analysis and categorization, top words to describe the vibes are extracted for the top 5 neighbourhoods in Seattle:

![](/images/capital_hill_wc.png)

![](/images/minor_wc.png)




## Reference

https://github.com/vgacutan/Airbnb/blob/main/Airbnb_SeatlleOpenData_analysis.ipynb
https://www.kaggle.com/code/jonathanoheix/sentiment-analysis-with-hotel-reviews
https://medium.com/@nutanbhogendrasharma/sentiment-analysis-for-hotel-reviews-with-nltk-and-keras-ce5cf3db39b
