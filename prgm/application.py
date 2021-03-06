# all the imports
import sys, os
import csv
import numpy as np
import requests, logging
try:
    from HTMLParser import HTMLParser
except ImportError:
    from html.parser import HTMLParser
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from newspaper import Article
from flask import Flask, request, session, g, redirect, url_for, abort, \
    render_template, flash
from finalModel import model
from LSTMfinal_model import lstm_model

h = HTMLParser()
from settings import APP_STATIC

# E
application = Flask(__name__)
application.debug = True

mod = model(fakeFile=os.path.join(APP_STATIC, 'fake2.txt'), realFile=os.path.join(APP_STATIC, 'real2.txt'))

def isInDictionary(d,url):
    list_sites = d.keys()
  
    if url.hostname in list_sites:
        return d[url.hostname]
    else:
        for key in list_sites:
            if key in url.hostname:
                return d[key]
    return -1

def isDomainReputable(url):
    non_credible_news = {}
    credible_news = {}
    #open the list of non credible news sources
    with open(os.path.join(APP_STATIC, 'open_sources_list.csv'), 'r') as csvfile:
        my_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #need to skip the first line of the site
        my_reader.__next__()
        for row in my_reader:
            non_credible_news[row[0]]=row[1]

    parsed_uri = urlparse(url)
    value = isInDictionary(non_credible_news,parsed_uri)
    if value is not -1:
        return value
    with open(os.path.join(APP_STATIC, 'credible.csv'), 'r') as csvfile:
        my_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        #need to skip the first line of the site
        my_reader.__next__()
        for row in my_reader:
            site = row[0]
            if site.startswith('"') and site.endswith('"'):
                site = site[1:-1]
            credible_news[site]="Credible"
    value = isInDictionary(credible_news,parsed_uri)

    if value is not -1:
        return value
    return "Site Not Found in data list!"


def getNewsTitle(url):
    try:
        r = requests.get(url)
        if r.status_code == 200 and "Fatal error" not in r.text:
            html = r.text
            start = html.find('<title>') + 7  
            end = html.find('</title>', start)
            title = html[start:end]
            title = h.unescape(title)

            if " - " in title:
                title = title.rsplit(' - ', 1)[0] 

    except requests.exceptions.ConnectionError:
        logging.error('failed to connect to: '+url)
    return title

@application.route('/analysis', methods=['GET', 'POST'])
def show_analysis():
    if request.method == 'POST':
        url = request.form['target_url']
        return redirect(url_for('show_analysis', url=url))
    else:
        entries = []
        url = request.args.get('url')
        if url is not None:
            entries.append(url)
            if url != '':
                title = getNewsTitle(url)
                entries.append(title)
                entries.append(isDomainReputable(url)) #statistic result

                if mod.queryRF(title) == 1:
                    entries.append("Real News")
                else:
                    entries.append("Fake News")

                if mod.queryRFSK(title) == 1:
                    entries.append("Real News")
                else:
                    entries.append("Fake News")

                if mod.queryDT(title) == 1:
                    entries.append("Real News")
                else:
                    entries.append("Fake News")

                if mod.queryMLP(title) == 1:
                    entries.append("Real News")
                else:
                    entries.append("Fake News")

                if mod.querySVM(title) == 1:
                    entries.append("Real News")
                else:
                    entries.append("Fake News")

                if mod.query(title) == 1:
                    entries.append("Real News")
                else:
                    entries.append("Fake News")
                
                article = Article(url)
                article.download()
                article.parse()
                article.nlp()

                # following is LSTM Analysis
                LSTM_Title_Model = lstm_model()
                LSTM_Title_Model = LSTM_Title_Model.reload_model(os.path.join(APP_STATIC, 'model_titles.json'),
                                                                    os.path.join(APP_STATIC, 'model.h5'))
                LSTM_Content_Model = lstm_model()
                LSTM_Content_Model = LSTM_Content_Model.reload_model(
                    os.path.join(APP_STATIC, 'model_keywords.json'), os.path.join(APP_STATIC, 'model.h5'))

                LSTM_Title_Model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                LSTM_Content_Model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

                title_training = lstm_model().format_testcase(title, 0, 39)
                content_training = lstm_model().format_testcase(article.keywords, 1, 19)

                title_result = np.sum(LSTM_Title_Model.predict(title_training)) / 39
                content_result = np.sum(LSTM_Content_Model.predict(content_training)) / 19

              
                if content_result > 0.5:
                    entries.append("Real News")
                else:
                    entries.append("Fake News")

        return render_template('analysis.html', entries=entries)



@application.route('/', methods=['GET', 'POST'])
def show_contents():
    if request.method == 'POST':
        url = request.form['target_url']
        return redirect(url_for('show_contents', url=url))
    else:
        entries = []
        url = request.args.get('url')
        print(url)
        if url is not None:
            entries.append(url)
            if url != '':
                title = getNewsTitle(url)
                entries.append(title)
                #if(isDomainReputable(url) == "Site Not Found in our data list!"):
                if mod.query(title) == 1:
                    predict_result = "Real News"
                else:
                    predict_result = "Fake News"
                entries.append(isDomainReputable(url)) #statistic result
                entries.append(predict_result)        #machine learning result
        return render_template('index.html', entries=entries)


if __name__ == "__main__":
    
    application.debug = True
    application.run()
