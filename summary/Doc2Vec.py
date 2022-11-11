from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

import pandas as pd
#from test import ReadCSV
#from updateSummarizer import CSVTransfer,readCSV
import numpy as np
import csv
import nltk
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
nltk.download('punkt')
# documents={
#     'documents':[],
# }
documents=[]
t='Machine learning is the study of computer algorithms that improve automatically through experience.\
 Machine learning algorithms build a mathematical model based on sample data, known as training data.\
 The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
 where no fully satisfactory algorithm is available.'

with open('summarized.csv') as csv_file:
            cnt=0
            max=-999
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if row[2]=='text':
                    continue
                documents.append(row[2])
#documents.append(t)
documents_df=pd.DataFrame(documents,columns=['documents'])


def appendCSV(text):
    global documents
    global documents_df
    documents.append(text)
    dict ={'documents':[text]}
    #documents_df=pd.DataFrame(documents,columns=['documents'])
    #documents_df=documents_df.append(dict,ignore_index=True)
    documents_df=pd.concat([documents_df, pd.DataFrame(data=dict)], ignore_index = True)
    documents_df.reset_index(inplace=True, drop=True)
    document_embeddings=clean_text()
    return document_embeddings
    #documents.append(text)
    

    



    #for row in d
#print(documents_df)
def transferData():
    return documents_df


#documents['documents'].append(t)
#documents_df=pd.DataFrame(data=documents)
#print(documents_df['documents'])



#Sample corpus
# documents = ['Machine learning is the study of computer algorithms that improve automatically through experience.\
# Machine learning algorithms build a mathematical model based on sample data, known as training data.\
# The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
# where no fully satisfactory algorithm is available.',
# 'Machine learning is the study of computer algorithms that improve automatically through experience.\
# Machine learning algorithms build a mathematical model based on sample data, known as training data.\
# The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
# where no fully satisfactory algorithm is available.',
# 'Machine learning is the study of computer algorithms that get better on their own over time.\
# Using training data, or sample data, machine learning algorithms create a mathematical model.\
# Machine learning is a field that uses a variety of techniques to train computers to do jobs.\
# when there is not an algorithm that is completely adequate.',
# 'Computer algorithms that develop automatically with use are the subject of machine learning research.\
# On the basis of training data—sample data—machine learning algorithms construct a mathematical model.\
# To train computers to carry out tasks, the field of machine learning uses a variety of techniques.\
# if there is not an algorithm that can totally satisfy the need.',
# 'A software engineer creates programs based on logic for the computer to execute. A software engineer has to be more concerned\
# about the correctness of the program in all the cases. Meanwhile, a data scientist is comfortable with uncertainty and variability.\
# Developing a machine learning application is more iterative and explorative process than software engineering.'
# ]
 
def clean_text():
    stop_words_l=stopwords.words('english')
    documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
    print('printing document: \n', documents_df)
    tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(documents_df.documents_cleaned)]
    model_d2v = Doc2Vec(vector_size=100,alpha=0.025, min_count=1)
    
    model_d2v.build_vocab(tagged_data)
    for epoch in range(100):
        model_d2v.train(tagged_data,
                    total_examples=model_d2v.corpus_count,
                    epochs=model_d2v.epochs)
    document_embeddings=np.zeros((documents_df.shape[0],100))
    for i in range(len(document_embeddings)):
        document_embeddings[i]=model_d2v.docvecs[i]
    return document_embeddings



#documents_df=pd.DataFrame(documents,columns=['documents'])
# 
# documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )



# #print(documents_df)

# # Tokenize the documents

# #print("the size of word2vec:" + str(documents_df.shape[0]))

# tagged_data = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(documents_df.documents_cleaned)]
# model_d2v = Doc2Vec(vector_size=100,alpha=0.025, min_count=1)
  
# model_d2v.build_vocab(tagged_data)



# for epoch in range(100):
#     model_d2v.train(tagged_data,
#                 total_examples=model_d2v.corpus_count,
#                 epochs=model_d2v.epochs)
    
# document_embeddings=np.zeros((documents_df.shape[0],100))



# for i in range(len(document_embeddings)):
#     document_embeddings[i]=model_d2v.docvecs[i]
# #print(document_embeddings)



def most_similar(doc_id,similarity_matrix,matrix):
    
    #print (f'Document: {documents_df.iloc[doc_id]["documents"]}')
    print ('\n')
    print ('Similar Documents:')
    if matrix=='Cosine Similarity':
        print(documents_df)
        print(similarity_matrix)
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
        print('\n')
        print(similar_ix)
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
        #print(ix)  
        print('\n')
        #print(similarity_matrix)
        #print (f'Document: {documents_df.iloc[ix]["documents"]}')
        print (f'{matrix} : {similarity_matrix[doc_id][ix]}')
        #print ()
#appendCSV(t)


#print(pairwise_similarities)
print('\n')
#pairwise_differences=euclidean_distances(document_embeddings)
cnt = 0
def word2Vec(text):
    global cnt
    cnt+=1
    max = -999
    #documents.append(text)
    #dict ={'documents':text}
    #documents_df=documents_df.append(dict,ignore_index=True)
    #documents_df['documents'].append(text)
    #documents_df=pd.DataFrame(documents,columns=['documents'])
    document_embeddings=appendCSV(text)
    pairwise_similarities=cosine_similarity(document_embeddings)
    print('Number of function called :'+ str(cnt))
    
    print('i Am in if')
    
    m=int(documents_df.shape[0])
    #print(documents)
    print(m)
    print('\n')
    print("the size of word2vec:" + str(documents_df.shape[0]))
    most_similar(m-1,pairwise_similarities,'Cosine Similarity')
    for i in range(m-1):
        similarity=pairwise_similarities[m-1][i]
        print(similarity)
        if similarity > max:
            max = similarity
            

    return max

#print(word2Vec(t))
    #print('pairwise similarity: '+ f'{pairwise_similarities[m-1][2]}')
    #return pairwise_similarities[m-1][m-1] 
#most_similar(0,pairwise_similarities,'Cosine Similarity')

#most_similar(1,pairwise_differences,'Euclidean Distance')
