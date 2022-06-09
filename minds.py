import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from time import sleep
import json

# global variables initialization
list_titles = []
list_exc = []

list_excerpt = []
list_pos = []
list_neg = []
list_neu = []

# initialize nltk
sia = SentimentIntensityAnalyzer()

# web scraping
response = requests.get('https://www.aljazeera.com/where/mozambique/').text
soup = BeautifulSoup(response, 'lxml')
contents = soup.find_all(class_='gc__content')

# iterating through news articles
for index, content in enumerate(contents):
    # stop iterating when 10 news articles collected
    # if index == 10:
    #     break

    # extract news headline
    title = content.find('a').find('span').text
    title = title.replace("&shy", "")
    title = title.replace(";\xad", "")
    title = title.replace("\xad", "")

    # extract news excerpt
    excerpt = content.find(class_="gc__excerpt").find('p').text
    excerpt = excerpt.replace("&shy", "")
    excerpt = excerpt.replace(";\xad", "")
    excerpt = excerpt.replace("\xad", "")

    print('Title: ' + title)
    print('Excerpt: ' + excerpt)
    print()

    # create corresponding lists for the news title and excerpts
    list_titles.append(title)
    list_exc.append(excerpt)

    # generate sentiment analysis for title and excerpt
    title_sia = sia.polarity_scores(title)
    exc_sia = sia.polarity_scores(excerpt)

    # prepare lists of relevant data to be passed to dataframe
    # corresponding df will be passed to plotly for polarity visualization
    list_excerpt.append('news: ' + str(index))
    list_pos.append(exc_sia['pos'])
    list_neg.append(exc_sia['neg'])
    list_neu.append(exc_sia['neu'])

    print()


# create df for plotly
df = pd.DataFrame(list(zip(list_exc, list_pos, list_neg, list_neu)))
df.columns = ['news', 'pos', 'neg', 'neu']

# Plotly - Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add subplots for positive polarity news
fig.add_trace(
    go.Scatter(x=list_excerpt, y=list_pos, name="Polarity > 0"),
    secondary_y=False,
)

# add subplots for negative polarity news
fig.add_trace(
    go.Scatter(x=list_excerpt, y=list_neg, name="Polarity < 0"),
    secondary_y=True,
)

# add subplots for neutral polarity news
fig.add_trace(
    go.Scatter(x=list_excerpt, y=list_neu, name="Polarity = 0"),
    secondary_y=True,
)

# add figure
fig.update_layout(
    title_text="Sentiment Analysis plot"
)

# Set x-axis title
fig.update_xaxes(title_text="news")

# Set y-axes titles
fig.update_yaxes(title_text="pos yaxis title", secondary_y=False)
fig.update_yaxes(title_text="neg yaxis title", secondary_y=True)
fig.update_yaxes(title_text="neu yaxis title", secondary_y=True)

# Store title, excerpt, sentiment data in news.json
articles = []
for index, title in enumerate(list_titles):
    dict = {}
    dict["title"] = title
    dict["excerpt"] = list_exc[index]
    dict["Positive Sentiment"] = list_pos[index]
    dict["Negative Sentiment"] = list_neg[index]
    dict["Neutral Sentiment"] = list_neu[index]
    articles.append(dict)

with open("news.json", "w") as f:
    json_data = json.dumps(articles, indent=4)
    f.write(json_data)


fig.show()
