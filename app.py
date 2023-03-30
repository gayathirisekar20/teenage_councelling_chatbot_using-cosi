# from bert_embedding import BertEmbedding
#from bert_serving.client import BertClient
from flask import Flask, render_template, request
import os
import json
import requests
import pickle
import joblib
import numpy as np
import pandas as pd
#import tensorflow as tf
#all packages 
import nltk 
import string 
import re
import random
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from textblob.sentiments import *
import re
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
# import mxnet as mx
# from bert_embedding import BertEmbedding
nltk.download('punkt')
nltk.download('wordnet') 


#all packages 
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk 
import string 
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from bs4 import BeautifulSoup
nltk.download('stopwords')
import string #has the list of all punctuations
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
import pandas as pd
lemmer = nltk.stem.WordNetLemmatizer()
stop_w = stopwords.words('english')

#WordNet is a semantically-oriented dictionary of English included in NLTK.

from flask import *
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager , UserMixin , login_required ,login_user, logout_user,current_user
app=Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///db.db'
app.config['SECRET_KEY']='619619'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=True
db = SQLAlchemy(app)

login_manager = LoginManager()
login_manager.init_app(app)

class User(UserMixin,db.Model):
    id = db.Column(db.Integer,primary_key=True,autoincrement=True)
    username = db.Column(db.String(200))
    email = db.Column(db.String(200))
    password = db.Column(db.String(200))



@login_manager.user_loader
def get(id):
    return User.query.get(id)

@app.route('/home',methods=['GET'])
@login_required
def get_home():
    return render_template('home.html')

@app.route('/',methods=['GET'])
def get_login():
    return render_template('login.html')


@app.route('/signup',methods=['GET'])
def get_signup():
    return render_template('signup.html')

@app.route('/login',methods=['POST'])
def login_post():
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email).first()
    login_user(user)
    return redirect('/home')

@app.route('/signup',methods=['POST'])
def signup_post():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    user = User(username=username,email=email,password=password)
    db.session.add(user)
    db.session.commit()
    user = User.query.filter_by(email=email).first()
    login_user(user)
    return redirect('/home')

@app.route('/logout',methods=['GET'])
def logout():
    logout_user()
    return redirect('/login')
#Remove punctuctions
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def clean(column,df,stopwords=False):
  #remove stop words
  df[column] = df[column].apply(str)
  df[column] = df[column].str.lower().str.split()
  if stopwords:
        df[column]=df[column].apply(lambda x: [item for item in x if item not in stop_w])
  #remove punctuation
  df[column]=df[column].apply(lambda x: [item for item in x if item not in string.punctuation])
  df[column]=df[column].apply(lambda x: " ".join(x))

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

def LemNormalize(text):
    text = text.lower()
    clean_text = text.translate(remove_punct_dict)
    words = nltk.word_tokenize(clean_text)
    return LemTokens(words)
def parents():
    df = pd.read_csv("parents.csv",encoding='latin-1')
    df2 = df
    

    clean('questionText',df2)
    df2=df2.fillna(0)

    questions = df2["questionText"].to_list()

    answers = df2["answerText"].to_list()
    sent_tokens =questions
    ans_sent_tokens=answers

    
    # answers=re.sub("[^a-zA-Z]", " ",str(answers))
    # questions=re.sub("[^a-zA-Z]", " ",str(questions))
    ques=[]
    ans=[]
    for i in questions:
        ques.append(i)
    for j in answers:
        ans.append(j)

    sent_tokens =ques
    ans_sent_tokens=ans



    return ques,ans_sent_tokens,sent_tokens

def stud():
    df = pd.read_csv("st.csv",encoding='latin-1')
    df2 = df
    

    clean('questionText',df2)
    df2=df2.fillna(0)

    questions = df2["questionText"].to_list()

    answers = df2["answerText"].to_list()
    sent_tokens =questions
    ans_sent_tokens=answers

    
    # answers=re.sub("[^a-zA-Z]", " ",str(answers))
    # questions=re.sub("[^a-zA-Z]", " ",str(questions))
    ques=[]
    ans=[]
    for i in questions:
        ques.append(i)
    for j in answers:
        ans.append(j)

    sent_tokens =ques
    ans_sent_tokens=ans



    return ques,ans_sent_tokens,sent_tokens






app.static_folder = 'static'



#Greeting messages
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
greetings = ['hi','hey', 'hello', 'heyy', 'hi', 'hey', 'good evening', 'good morning', 'good afternoon', 'good', 'fine', 'okay', 'great', 'could be better', 'not so great', 'very well thanks', 'fine and you', "i'm doing well", 'pleasure to meet you', 'hi whatsup']
happy_emotions = ['i feel good', 'life is good', 'life is great', "i've had a wonderful day", "i'm doing good"]
goodbyes = ['thank you', 'thank you', 'yes bye', 'bye', 'thanks and bye', 'ok thanks bye', 'goodbye', 'see ya later', 'alright thanks bye', "that's all bye", 'nice talking with you', 'i’ve gotta go', 'i’m off', 'good night', 'see ya', 'see ya later', 'catch ya later', 'adios', 'talk to you later', 'bye bye', 'all right then', 'thanks', 'thank you', 'thx', 'thx bye', 'thnks', 'thank u for ur help', 'many thanks', 'you saved my day', 'thanks a bunch', "i can't thank you enough", "you're great", 'thanks a ton', 'grateful for your help', 'i owe you one', 'thanks a million', 'really appreciate your help', 'no', 'no goodbye']

#function to generate random greetings
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
def response(user_response,sent_tokens,ans_sent_tokens):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    # print(sent_tokens)
    print(len(sent_tokens))
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    print(req_tfidf)
    if(req_tfidf>=0.5):
        robo_response = robo_response+sent_tokens[idx]
        if idx<(len(sent_tokens)-1):
            print(ans_sent_tokens[idx])
            return ans_sent_tokens[idx]
        
    else:
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response

def data(userText,who):
    
    flag=True

    while(flag==True):
        # user_response = input()
        user_response=userText.lower()
        if(user_response!='bye'):
            if(user_response=='thanks' or user_response=='thank you' ):
                flag=False
                return "You are welcome.."
                # print("BOT: You are welcome..")
            else:
                if(greeting(user_response)!=None):
                    return greeting(user_response)
                else:
                    # print("BOT: ",end="")
                    if who =="par":
                        ques,ans_sent_tokens,sent_tokens=parents()
                    else:
                        ques,ans_sent_tokens,sent_tokens=stud()
                    res=response(user_response,ques,ans_sent_tokens)
                    sent_tokens.remove(user_response)
                    return res
        else:
            flag=False
            return "BOT: Bye! take care.."
        
@app.route("/about")
def about():
    return render_template("about.html")            

@app.route("/parent")
def home():
    return render_template("index.html")

@app.route("/student")
def student():
    return render_template("student.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    # cleanText = clean_text(str(userText))
    # user_response=userText.lower()
    #check sentiment 
    blob = TextBlob(userText, analyzer=PatternAnalyzer())
    polarity = blob.sentiment.polarity

    if userText in greetings:
        return "Hello! How may I help you today?"
    elif polarity>0.7:
        return "That's great! Do you still have any questions for me?"
    elif userText in happy_emotions:
        return "That's great! Do you still have any questions for me?"  
    elif userText in goodbyes:
        return "Hope I was able to help you today! Take care, bye!"
    topic = data(userText,who="par")
    print (topic)
    # res = random.choice(dictionary[topic])
    # print (res)
    return topic

@app.route("/get_student")
def get_student():
    userText = request.args.get('msg')
    # cleanText = clean_text(str(userText))
    # user_response=userText.lower()
    #check sentiment 
    blob = TextBlob(userText, analyzer=PatternAnalyzer())
    polarity = blob.sentiment.polarity

    if userText in greetings:
        return "Hello! How may I help you today?"
    elif polarity>0.7:
        return "That's great! Do you still have any questions for me?"
    elif userText in happy_emotions:
        return "That's great! Do you still have any questions for me?"  
    elif userText in goodbyes:
        return "Hope I was able to help you today! Take care, bye!"
    topic = data(userText,who="stud")
    print (topic)
    # res = random.choice(dictionary[topic])
    # print (res)
    return topic

if __name__ == "__main__":
    app.run(debug=True) 
