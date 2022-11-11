import csv
import numpy as np
from numpy.linalg import norm
   

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


######################################Cosine Similarity############################################################
def cosine_similarity(Y):
    with open('summarized.csv') as csv_file:
        cnt=0
        max=-999
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[2]=='text':
                continue
            X=row[2]
            #print("print of x  :"+X)
            print('##############################################')
            # tokenization
            X=X.lower()
    #print("print of Y   :"+Y)
            Y=Y.lower()

            X_list = word_tokenize(X) 
            Y_list = word_tokenize(Y)

        # sw contains the list of stopwords
            sw = stopwords.words('english') 
            l1 =[];l2 =[]

            # remove stop words from the string
            X_set = {w for w in X_list if not w in sw} 
            Y_set = {w for w in Y_list if not w in sw}
            # form a set containing keywords of both strings 
            rvector = X_set.union(Y_set) 
            for w in rvector:
                if w in X_set: l1.append(1) # create a vector
                else: l1.append(0)
                if w in Y_set: l2.append(1)
                else: l2.append(0)
            c = 0
            #print(l1)
            #print(l2)
            # cosine formula 
            for i in range(len(rvector)):
                    c+= l1[i]*l2[i]

            cosine = c / float((sum(l1)*sum(l2))**0.5)
            if cosine > max:
                max=cosine
                
            print("similarity with "+"  :", cosine)

        #print(row[2])
            print('##############################################')
            cnt+=1
            print(cnt)
            print(max)
    return max

            



# v1='Machine learning is the study of computer algorithms that improve automatically through experience.\
# Machine learning algorithms build a mathematical model based on sample data, known as training data.\
# The discipline of machine learning employs various approaches to teach computers to accomplish tasks \
# where no fully satisfactory algorithm is available.'

# v2= 'Computer algorithms that develop automatically with use are the subject of machine learning research.\
# On the basis of training data—sample data—machine learning algorithms construct a mathematical model.\
# To train computers to carry out tasks, the field of machine learning uses a variety of techniques.\
# if there is not an algorithm that can totally satisfy the need.'
#cosine_similarity(v1,v2)
