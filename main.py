
import uvicorn ##ASGI       
from fastapi import FastAPI, Request as RQ
from nltk.corpus import stopwords
import numpy as np
import networkx as nx
import regex
import nltk
import requests
import urllib.parse
nltk.download('stopwords')
from send_email import *


##############GOOGLE API AUTH FLOW######################################################
########################################################################################
# Request all access (permission to read/send/receive emails, manage the inbox, and more)
SCOPES = ['https://mail.google.com/']
our_email = 'docsummarizer@gmail.com'

def gmail_authenticate():
    creds = None
    # the file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time
    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)
    # if there are no (valid) credentials availablle, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # save the credentials for the next run
        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)
    return build('gmail', 'v1', credentials=creds)


##############END OF GOOGLE API AUTH FLOW######################################################
###############################################################################################

def read_article(data):

    article = data.split(". ")
    sentences = []
    for sentence in article:
        review = regex.sub("[^A-Za-z0-9]",' ', sentence)
        sentences.append(review.replace("[^a-zA-Z]", " ").split(" "))        
    sentences.pop()     
    return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words) #makes a vector of len all_words
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - nltk.cluster.util.cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(file_name, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text anc split it
    sentences =  read_article(file_name)
    
    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
#     print("\n\n---------------\nIndexes of top ranked_sentence order are ", ranked_sentence)    

    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    # print("\n")
    # print("*"*140)
    # print("\n\nSUMMARY: \n---------\n\n", ". ".join(summarize_text))
    a = ". ".join(summarize_text)
    return a


app = FastAPI()


@app.get('/')
def home(request: RQ):
   print("Beginning of logging")
   return "docs-summarizer-server-home"


@app.get("/get_summary/input{doc}/email{email_id}") #
async def get_summary(doc: str, to_email_id: str): 
   summarize=urllib.parse.unquote_plus(doc)
   to_email_id=urllib.parse.unquote_plus(to_email_id)
   summary=generate_summary(summarize)
   send_message(service, to_email_id, "HURRAY! Your Document has been summarized!", 
               f"Hello,\n\nYour document has been summarized.\n\n{summary}\n\n\nBest,\ndocsummarizer.io", []) #"test.txt", "anyfile.png"
   return generate_summary(doc)


service = gmail_authenticate()

if __name__ == '__main__':
    print("Server starting!")
   # get the Gmail API service
    service = gmail_authenticate()
    print(service)
    print("Here")
   # test send email
    send_message(service, "akanksh.belchada@gmail.com", "DOC-SUMMARIZER SERVICE INITIATED", 
               "Hello DEV-TEAM,\n\ndocsummarizer.io is now up and running.\nServices for Heroku Postgres, Heroku Data and for Redis and https://docsummarizer.herokuapp.com/ API services are now running.\n\n\nBest,\nmain.py.", []) #"test.txt", "anyfile.png"
    print("initiation-message-delivered")
    print("There")
    uvicorn.run(app, host='127.0.0.1', port=7000)
