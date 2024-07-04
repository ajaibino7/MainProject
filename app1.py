from flask import Flask,render_template,request
import transformers
transformers.logging.set_verbosity_error()
from transformers import T5ForConditionalGeneration, T5Tokenizer
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi as ytt
from flask import Flask, render_template, request
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch
from flaskext.mysql import MySQL
from flask import (Flask, request, session, g, redirect, url_for, abort, render_template, flash, Response)
import os
import re
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

def check(email):
 
    if(re.fullmatch(regex, email)):
        return True
 
    else:
        return False
        
mysql = MySQL()
app = Flask(__name__,static_folder='static')
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'password123'
app.config['MYSQL_DATABASE_DB'] = 'chat'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)
app = Flask(__name__)

# Load the pretrained model
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')

# initialize the model architecture and weights
model_t = T5ForConditionalGeneration.from_pretrained("t5-small")
# initialize the model tokenizer
tokenizer_t = T5Tokenizer.from_pretrained("t5-small")
# define your resource endpoints

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Function to load and preprocess text from file
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Split text into lines
    lines = text.split('\n')
    # Preprocess each line
    lines = [preprocess_text(line) for line in lines]
    return lines

# Function to find answer to a question
def find_answer(question, lines):
    # Preprocess question
    question = preprocess_text(question)
    # Add question to text for comparison
    lines.append(question)
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Fit transform the text
    tfidf_matrix = vectorizer.fit_transform(lines)
    # Calculate cosine similarity between question and each line
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    # Find the index of the most similar line
    most_similar_index = np.argmax(similarity_scores)
    # Return the most similar line as the answer
    answer = lines[most_similar_index].split(" and ")[0]
    return answer



@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        cursor = mysql.connect().cursor()
        cursor.execute("SELECT * from user_details where email='" + email + "' and password='" + password + "'")
        print("SELECT * from user_details where email='" + email + "' and password='" + password + "'")
        data = cursor.fetchone()
        if data is None:
            return "Username or Password is wrong"
        else:
            return redirect(url_for('home'))

        return render_template('index.html')

    else:
        return render_template('index.html')

@app.route('/addUser',methods = ["GET","POST"])
def addUser():
    if request.method == 'POST':
        User_name = request.form['user_name']
        email = request.form['email']
        Phone_number = request.form['phone_number']
        age = request.form['age']
        Password = request.form['password']
        address = request.form['address']
        res=check(email)
        if User_name=="" or email=="" or Phone_number=="" or age=="" or Password=="" or address=="":
            return "Please Fill all fileds"
        if not res:
            return "Please Enter a valid Email"
        qry = " INSERT INTO `user_details` ( user_name,email, Password, age, phone_number,address ) values "
        qry += "('"+User_name +"','"+email +"','"+Password +"','"+age +"','"+Phone_number +"','"+address +"')"
        conn = mysql.connect()
        cursor = conn.cursor()
        cursor.execute(qry)
        conn.commit()
        return redirect(url_for('index'))
    return render_template('addUser.html')

@app.route("/home",methods = ["GET","POST"])
def home():
    return render_template('home.html')


#extracts id from url
def extract_video_id(url:str):
    # Examples:
    # - http://youtu.be/SA2iWivDJiE
    # - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    # - http://www.youtube.com/embed/SA2iWivDJiE
    # - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    query = urlparse(url)
    if query.hostname == 'youtu.be': return query.path[1:]
    if query.hostname in {'www.youtube.com', 'youtube.com'}:
        if query.path == '/watch': return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/embed/': return query.path.split('/')[2]
        if query.path[:3] == '/v/': return query.path.split('/')[2]
    # fail?
    return None
#text summarizer
def summarizer(script):
    # encode the text into tensor of integers using the appropriate tokenizer
    input_ids = tokenizer_t("summarize: " + script, return_tensors="pt", max_length=2048, truncation=True).input_ids
    # generate the summarization output
    outputs = model_t.generate(
        input_ids, 
        max_length=1500, 
        min_length=500, 
        length_penalty=2.0, 
        num_beams=6, 
        early_stopping=True)

    summary_text = tokenizer_t.decode(outputs[0])
    return(summary_text)

def write_text(text):
    with open("text.txt", "a") as file:
        file.write('\n' + text)
@app.route('/summarize',methods=['GET','POST'])
def video_transcript():
    if request.method == 'POST':
        url = request.form['youtube_url']
        video_id = extract_video_id(url)
        data = ytt.get_transcript(video_id,languages=['de', 'en'])
        print("data--------",data)
        
        scripts = []
        for text in data:
            for key,value in text.items():
                if(key=='text'):
                    scripts.append(value)
        transcript = " ".join(scripts)
        print("transcript---------------",transcript)
        summary = summarizer(transcript)
        summary = summary.replace("</s>","")
        summary = summary.replace("<pad>","")
        try:
            summary = summary.split("...")[0]
        except Exception as e:
            print(e)


        write_text(summary)

        return render_template('summery.html', summary=summary)
    else:
        return render_template('summery.html')

@app.route('/answer',methods=['GET','POST'])
def answer_question():
    # Get the question from the UI
    if request.method == 'POST':
        question = request.form['question']

        lines = load_text("text.txt")

        answer = find_answer(question, lines)

        conversation = [{'question': question, 'answer': answer}]

        return render_template('chat.html', conversation=conversation)
    else:
        return render_template('chat.html', conversation=[])

   
    
# server the app when this file is run
if __name__ == '__main__':
    app.run()