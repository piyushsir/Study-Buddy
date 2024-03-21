import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Users/piyush gupta/OneDrive/Desktop/studyBuddy/Server/dataset/mbti_1.csv')
#print(data)
cnt_types = data['type'].value_counts()

"""plt.figure(figsize = (12,4))
sns.barplot(x=cnt_types.index,y=cnt_types.values,alpha = 0.8)
plt.ylabel("numer of occurances",fontsize=12)
plt.xlabel("types",fontsize=12)
plt.show()"""

#[p.split('|||') for p in data.head(2).posts.values]
def get_types(row):
    t = row['type']
    I=0;N=0
    T=0;J=0

    if t[0] == 'I' :I=1
    elif t[0]=='E' :E=0
    else : print('I-E incorrect')

    if t[1] == 'N' :N=1
    elif t[1]=='S' :S=0
    else : print('N-S incorrect')

    if t[2] == 'T' :T=1
    elif t[2]=='F' :F=0
    else : print('T-F incorrect')

    if t[3] == 'J' :J=1
    elif t[3]=='P' :P=0
    else : print('J-P incorrect')

    return pd.Series({'IE':I , 'NS':N , 'TF':T , 'JP':J})

#data  =  data.join(data.apply(lambda row :get_types(row),axis =1))
binary_p = {'I':0 ,'E':1 ,'N':0 , 'S':1 ,'F':0 ,'T':1 ,'J':0 ,'P':1}
binary_plist = [{0:'I' , 1:'E'},{0:'N',1:'S'},{0:'F',1:'T'},{0:'J',1:'P'}]
def translate_pers(pers):
    return [binary_p[l] for l in pers]

def translate_back(pers):
    s=""
    for i,l in enumerate(pers):
        s+=binary_plist[i][l]
    return s
d = data.head(4)
list_personality_bin = np.array([translate_pers(p) for p in d.type])
#print(list_personality_bin)
import nltk
#nltk.download('stopwords')

from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
#nltk.download('wordnet')

unique_lst = ['INFJ','ENTP','INTP','INTJ','ENTJ','ENFJ','INFP','ENFP','ISFP','ISTP','ISFJ','ISTJ','ESTP','ESFP','ESTJ','ESFJ']
unique_lst = [x.lower() for x in unique_lst]

#stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

cachedStopWords = stopwords.words("english")

def preprocessing(data,remove_stop_words=True,remove_mbti_profiles=True):
    list_personality=[]
    list_posts=[]
    len_data = len(data)
    i=0

    for row in data.iterrows():
        i+=1
        if (i%500 ==0 or i==1 or i==len_data):
            print("% s of %s rows" % (i,len_data))
        posts = row[1].posts
        tmp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ',posts)
        tmp = re.sub("[^a-zA-Z]"," ",tmp)
        tmp = re.sub(' +',' ',tmp).lower()
    
        if remove_stop_words:
            tmp = " ".join([lemmatiser.lemmatize(w) for w in tmp.split(' ') if w not in cachedStopWords])
        else:
            tmp = " ".join([lemmatiser.lemmatize(w) for w in tmp.split(' ')])
    
        if remove_mbti_profiles:
            for t in unique_lst:
                tmp = tmp.replace(t,"")
        type_labelized = translate_pers(row[1].type)
        list_personality.append(type_labelized)
        list_posts.append(tmp)
    list_posts=np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts,list_personality
    
list_posts,list_personality = preprocessing(data,remove_stop_words=True)
#print(list_posts[0])

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

cntizer = CountVectorizer(analyzer="word",max_features=1500,tokenizer=None,preprocessor=None,stop_words=None,max_df=0.7,min_df=0.1)

print("CountVectorizer...")

X_cnt = cntizer.fit_transform(list_posts)
tfizer = TfidfTransformer()

print("If-idf...")
X_tfidf = tfizer.fit_transform(X_cnt).toarray()

feature_names = list(enumerate(cntizer.get_feature_names_out()))
#feature_names

type_indicators = ["IE:Introversion(I)/Extroversion(E)","NS:Intuition(N)-Sensing(S)","FT:Feeling(F)-Thinking(T)","JP:Judging(J)-Preceiving(P)"]
from numpy import loadtxt 
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

X = X_tfidf

"""for i in range(len(type_indicators)):
    print("%s ..." % (type_indicators[i]))

    Y = list_personality[:,i]
    
    seed = 7
    test_size = 0.33
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
     
    model = XGBClassifier() 
    model.fit(x_train, y_train)
    
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions) 
    print(" %s Accuracy: %.2f2%%" % (type_indicators[i], accuracy*100.0))

for i in range(len(type_indicators)):
    print("%s ..." % (type_indicators[i]))

    Y = list_personality[:,i]
    
    seed = 7
    test_size = 0.33
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
     
    model = XGBClassifier() 
    eval_set = [(x_test,y_test)]
    model.fit(x_train, y_train,early_stopping_rounds=10,eval_metric="logloss",eval_set =eval_set,verbose=True)
    
    y_pred = model.predict(x_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions) 
    print(" %s Accuracy: %.2f2%%" % (type_indicators[i], accuracy*100.0))"""

"""
from xgboost import plot_importance
y = list_personality[:,0] 

model = XGBClassifier() 
model.fit(X, y)
ax = plot_importance (model, max_num_features=25)

fig = ax.figure

fig.set_size_inches (15, 20)

plt.show()
features = sorted(list (enumerate(model.feature_importances_)), key=lambda x: x[1], reverse=True)
for f in features[0:25]:
    print("%d\t%f\t%s" % (f[0],f[1],cntizer.get_feature_names_out()[f[0]]))

default_get_xgb_params = model.get_xgb_params()
"""
"""from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

X = X_tfidf
param={}
param['n_estimators']=200
param['max_depth'] = 2 
param['nthread'] = 8
param['learning_rate'] = 0.2

for l in range(len(type_indicators)):
    print("%s ..." % (type_indicators[l]))
    Y = list_personality[:,l]
    model = XGBClassifier(**param) 
    print("hello")
    param_grid = {
            'n_estimators' : [ 200, 300],'learning_rate': [ 0.2, 0.3]
    }

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, Y)
    print(" Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("* %f (%f) with: %r" % (mean, stdev, param))
"""

def psycho_test(Posts_array):

            my_posts = Posts_array

            mydata = pd.DataFrame(data={'type':['ENTP'],'posts':[my_posts]})
            my_posts,dummy = preprocessing(mydata,remove_stop_words=True)
            my_x_cnt = cntizer.transform(my_posts)
            my_x_tfidf = tfizer.transform(my_x_cnt).toarray()

            print(my_posts)
            param={}
            param['n_estimators']=200
            param['max_depth'] = 2 
            param['nthread'] = 8
            param['learning_rate'] = 0.2

            result=[]
            for i in range(len(type_indicators)):
                print("%s ..." % (type_indicators[i]))

                Y = list_personality[:,i]
                
                seed = 7
                test_size = 0.33
                x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
                
                model = XGBClassifier(**param) 
                model.fit(x_train, y_train)
                
                y_pred = model.predict(my_x_tfidf)
                result.append(y_pred[0])
                print(y_pred)
            finalres = translate_back(result)
            print(finalres)
            return finalres
