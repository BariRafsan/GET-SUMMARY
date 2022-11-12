from distutils.debug import DEBUG
from flask import Flask, render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import xml.etree.ElementTree as ET
import pandas as pd
import spacy
nlp = spacy.load('en_core_web_sm') 
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
#from Making_pdf import pdf_write
from cosine_similarity import cosine_similarity
#from Doc2Vec import word2Vec,transferData,appendCSV
stopwords = list(STOP_WORDS)

import os
import csv
app = Flask(__name__)
new_docs = {
    'topic': [],
    'file': [],
    'text': [],
    'cnt': []
}
docs = {
    'topic': [],
    'text': [],
    #'cnt': [],
    'summary': [],
   # 'cnt2': []
}
# documents ={
#     'documents':[],
#     'documents_cleaned':[]
# }
summarized_df = pd.DataFrame(data=new_docs)
summarized_df.to_csv('summarized.csv', index=False)


#data=transferData()

# def readCSV():    
#     with open('document.csv') as csv_file:
#             cnt=0
#             max=-999
#             csv_reader = csv.reader(csv_file)
#             for row in csv_reader:
#                 if row[0]=='documents':
#                     continue
#                 documents['documents'].append(row[0])
#     documents_df=pd.DataFrame(data=documents)
#     return documents_df
# def CSVTransfer():
#     document = readCSV()
#     return document


topic=''
cnt=0
cnt2=0

num = 0




    

def remove_slash_n(s: str) -> str:
    """remove newline form a string"""
    return s.replace('\n', '')

def getSummary(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]

    punctuation1 = punctuation + '\n' +'\n\n'

    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation1:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] +=1
                
    max_frequencys = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word]/max_frequencys

    sentence_tokens = [sent for sent in doc.sents]

    sentence_score = {}
    for sent in sentence_tokens: #create a dictionary for sen tokens
        for word in doc:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_score.keys():
                    sentence_score[sent]= word_frequencies[word.text.lower()]
                else:
                    sentence_score[sent] += word_frequencies[word.text.lower()]

    select_length = int(len(sentence_tokens)*0.1)
    summary  = nlargest(select_length, sentence_score,key= sentence_score.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    #print(summary)
    return summary
def fileRead(file_p,topic):             #read the file and save it to csv file
    t=''
    global cnt
    with open(file_p, 'r') as f:
        data = f.read()
        tree = ET.parse(file_p)
        try:
            headline_tag = tree.getroot().find('HEADLINE')
            headline = remove_slash_n(headline_tag.text)
        except:
            headline = None
        try:
            date_line_tag = tree.getroot().find('DATELINE')
            date_line = remove_slash_n(date_line_tag.text)
        except:
            date_line = ' '
        paragraphs = [ remove_slash_n(p.text) for p in tree.getroot().find('TEXT').findall('P') ]
        text = ' '.join(paragraphs)
        
        cnt=len(text)
        #documents['documents'].append(text)
        #appendCSV(text)
        #m=word2Vec(text)        #callling Doc2Vec
        #m=float(m)
        m=cosine_similarity(text)           #call the cosine similarity function
        print('Cosine Similarity :'+str(m))

        #elif m>0.8:
        if m > 0.8:
            g='Similarity found'
            print(g)     
        else:
            g='No Similarity found'
            print(g)
            t +=  text
            print(topic)
            new_docs['topic'].append(topic)
            new_docs['file'].append(file_p)
            new_docs['text'].append(text)
            new_docs['cnt'].append(cnt)
            summarized_df = pd.DataFrame(data=new_docs)
            summarized_df.to_csv('summarized.csv', index=False)
    return t

            
def getSummery2(t): #joint the text of all the files and Summarize it and save it to csv file
    msg=''
    global topic
    global cnt,cnt2
    docs['topic'].append(topic)
    docs['text'].append(t)
    cnt=len(t)
    msg = getSummary(t)

    docs['summary'].append(msg)
    summarized_df_new = pd.DataFrame(data=docs)
    summarized_df_new.to_csv('summarized_new.csv', index=False)
    return msg
@app.route('/filee', methods=['GET', 'POST'])
def filee():                #get how many files to upload
    global num
    if request.method == 'POST':
        num = request.form['num']
        if num is None:
            num=2
        else :
            num=int(num)
        print(num)
        num=int(num)
        return render_template('file_upload.html',num=num)

@app.route('/fileu', methods=['GET', 'POST'])
def file():             #upload files
    global topic
    global num
    c='' 
    m=''
    msg=''      #msg is the summary of the text
    
      
    target = os.path.join(os.getcwd(), 'files/')
    if not os.path.isdir(target):
        os.mkdir(target)
    if request.method == 'POST':
        topic = request.form['topic']
        num=int(num)
        for i in range(0,num):
            
            file = request.files['file'+str(i)]
            filename = secure_filename(file.filename)
            destination = "/".join([target, filename])
            file.save(destination)
            file_p=destination
            c+=fileRead(file_p,topic)       #read files and save the text of all files in c
            #print(c)
            if c=='':
                m=" Similer Document Found"
            #msg='ALL Similer Document'
            else:
                msg = getSummery2(c)    
            
        
        #print("the ffffffffff"+t)
        print(topic)
    #pdf_write(msg)
        
        
    return render_template('file_upload.html', m=m,msg=msg,num=num)



if __name__ == '__main__':
    app.run(debug=True)