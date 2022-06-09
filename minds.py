import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
from time import sleep

sia = SentimentIntensityAnalyzer()

response = requests.get('https://www.aljazeera.com/where/mozambique/').text

soup = BeautifulSoup(response, 'lxml')

contents = soup.find_all(class_='gc__content')

stopwords = nltk.corpus.stopwords.words('english')

list_titles = []
list_exc = []

list_excerpt = []
list_pos = []
list_neg = []
list_neu = []

for index, content in enumerate(contents):
    title = str(content.find('a').find('span').text)
    title = title.replace('-', '')
    excerpt = str(content.find(class_="gc__excerpt").find('p').text)
    excerpt = excerpt.replace('-', '')

    print(title)
    print(excerpt)
    title_sia = sia.polarity_scores(title)
    exc_sia = sia.polarity_scores(excerpt)

    for i in tqdm(range(0, 100), desc="Text You Want"):
        sleep(.1)

    list_excerpt.append('news: ' + str(index))
    list_pos.append(exc_sia['pos'])
    list_neg.append(exc_sia['neg'])
    list_neu.append(exc_sia['neu'])

    print(title_sia)
    print(exc_sia)

    list_titles.append(title)
    list_exc.append(excerpt)
    print()


df = pd.DataFrame(list(zip(list_exc, list_pos, list_neg, list_neu)))
df.columns = ['news', 'pos', 'neg', 'neu']

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=list_excerpt, y=list_pos, name="yaxis data"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=list_excerpt, y=list_neg, name="yaxis2 data"),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=list_excerpt, y=list_neu, name="yaxis2 data"),
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="Double Y Axis Example"
)

# Set x-axis title
fig.update_xaxes(title_text="xaxis title")

# Set y-axes titles
fig.update_yaxes(title_text="<b>primary</b> yaxis title", secondary_y=False)
fig.update_yaxes(title_text="<b>secondary</b> yaxis title", secondary_y=True)

fig.show()

words_title = [word for word in list_titles if word.isalpha() and word not in stopwords]
words_excerpt = [word for word in list_exc if word.isalpha() and word not in stopwords]

print(words_title)
print()
print(words_excerpt)


print()

