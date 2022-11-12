import csv
#from updateSummarizer import getSummary
# import pandas module 
import spacy
nlp = spacy.load('en_core_web_sm') 
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import rouge
import numpy as np
stopwords = list(STOP_WORDS)

import pandas as pd 
df = pd.read_csv("merged.csv")
n=len(df['sum1'][82])
d=len(df['text'][82])
print("d,n",d,n)
#df2=pd.read_csv("merged2.csv")
#print(text1)
def getSummary(text):
    x=len(text)
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
    if(x<10000):
        select_length = int(len(sentence_tokens)*0.09)
    else:
        select_length = int(len(sentence_tokens)*0.019)
    summary  = nlargest(select_length, sentence_score,key= sentence_score.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)
    #print(summary)
    return summary

# y=getSummary(df['text'][82])
# print(len(y))
def summary_csv():
    df = pd.read_csv("merged.csv")
    cnt=0
    for index, row in df.iterrows():
        text1 = row['text']
        #print(text1)
        summary = getSummary(text1)
        cnt+=1
        #print(len(summary))
        print(cnt)
        row['text']=summary
        #df.at[index, 'summary'] = summary
    #print(df)
    df.to_csv('merged2.csv', index=False)
    
    # x=len(df2['text'][0])
    # print(x)

cnt=0
df2 = pd.read_csv("merged2.csv")
def evaluate_summary(y_test, predicted):
    global cnt
    cnt+=1
    rouge_score = rouge.Rouge()    
    scores = rouge_score.get_scores(y_test, predicted, avg=True)       
    score_1 = round(scores['rouge-1']['f'], 2)
    
    score_2 = round(scores['rouge-2']['f'], 2)    
    score_L = round(scores['rouge-l']['f'], 2)
    avg_score = round(np.mean(
         [score_1,score_2,score_L]), 2)
    print("rouge1:", score_1, "| rouge2:", score_2, "| rougeL:",
         score_2, "--> avg rouge:", round(np.mean(
         [score_1,score_2,score_L]), 2))    
    #print("rouge"+str(cnt)+":", str(score_1))## Apply the function to predicted
    return avg_score
min=-999
def max_rouge():
    global min
    cnt2=0
    sum=0
    for index, row in df2.iterrows():

        score1 = evaluate_summary(row['text'],row['sum1'])
        score2 = evaluate_summary(row['text'],row['sum2'])
        score3 = evaluate_summary(row['text'],row['sum3'])
        score4 = evaluate_summary(row['text'],row['sum4'])
        #score=(score1+score2+score3+score4)/4
        #score = avg(score1,score2,score3,score4)
        score=max(score1,score2,score3,score4)
        #cnt2+=1
        #sum+=score
        print('Score Value:',score)
    #print(cnt2)
    #print(sum)
    #rouge=sum/cnt2
        if score> min:
             min=score
    print('max Value:',min)        #row['text']=summary
        
#summary_csv()
max_rouge()
