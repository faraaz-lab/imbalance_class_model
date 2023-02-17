# pip install tensorflow==2.10.0 tensorflow-hub==0.12.0 unidecode imblearn scikit-learn==1.0.1

import re
import boto3
import sagemaker
from sagemaker import get_execution_role

sess = sagemaker.Session()
print('sess:', sess)

region = boto3.Session().region_name
print('region:', region)

bucket = sess.default_bucket()
print('bucket:', bucket)

role = get_execution_role()
print('role:', role)
# Load train dataframe from S3
import pandas as pd
import io
import boto3

bucket = 'bucket_name'
import pickle
# s3 = boto3.client('s3')
# response = s3.get_object(Bucket=bucket, Key='key_name')
# df = pickle.load(io.BytesIO(response['Body'].read()))

s3 = boto3.resource('s3')
df=pickle.loads(s3.Bucket(bucket).Object('key_name').get()['Body'].read())

import pandas as pd
import numpy as np

#Data Visualization
# import matplotlib.pyplot as plt

#Text Color
# from termcolor import colored

#Train Test Split
from sklearn.model_selection import train_test_split

#Model Evaluation
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
# from mlxtend.plotting import plot_confusion_matrix

#Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow_hub import KerasLayer

from sklearn.model_selection import train_test_split
from unidecode import unidecode
import re

from  tensorflow.keras  import  backend  as  K

import tensorflow_hub as hub
import traceback

#some additional metrics, to-do: definition of true positives according to serious=yes o nly
def mcor(y_true, y_pred):
    try:
        #matthews_correlation
        y_pred_pos  =  K.round(K.clip(y_pred,  0,  1)) 
        y_pred_neg = 1 - y_pred_pos
        y_pos  =  K.round(K.clip(y_true,  0,  1)) 
        y_neg = 1 - y_pos
        tp  =  K.sum(y_pos  *  y_pred_pos) 
        tn  =  K.sum(y_neg  *  y_pred_neg) 
        fp  =  K.sum(y_neg  *  y_pred_pos) 
        fn  =  K.sum(y_pos  *  y_pred_neg) 
        numerator = (tp * tn - fp * fn)
        denominator  =  K.sqrt((tp  +  fp)  *  (tp  +  fn)  *  (tn  +  fp)  *  (tn  +  fn))
        return  numerator  /  (denominator  +  K.epsilon())
    except:
        traceback.print_exc()
        

def recall_dl(y_true, y_pred):
    try:
        y_pred_pos  =  K.round(K.clip(y_pred,  0,  1)) 
        y_pred_neg = 1 - y_pred_pos
        y_pos  =  K.round(K.clip(y_true,  0,  1)) 
        y_neg = 1 - y_pos
        tp  =  K.sum(y_pos  *  y_pred_pos) 
        tn  =  K.sum(y_neg  *  y_pred_neg) 
        fp  =  K.sum(y_neg  *  y_pred_pos) 
        fn  =  K.sum(y_pos  *  y_pred_neg) 
        return tp / (tp+fn)
    except:
        traceback.print_exc()
        
def precision_dl(y_true, y_pred):
    try:
        y_pred_pos  =  K.round(K.clip(y_pred,  0,  1)) 
        y_pred_neg = 1 - y_pred_pos
        y_pos  =  K.round(K.clip(y_true,  0,  1)) 
        y_neg = 1 - y_pos
        tp  =  K.sum(y_pos  *  y_pred_pos) 
        tn  =  K.sum(y_neg  *  y_pred_neg) 
        fp  =  K.sum(y_neg  *  y_pred_pos) 
        fn  =  K.sum(y_pos  *  y_pred_neg) 
        return tp / (tp+fp)
    except:
        traceback.print_exc()

def f1_dl(y_true,y_pred):
    try:
        return  2*((precision_dl(y_true,y_pred)*recall_dl(y_true,y_pred))/(precision_dl(y_true,y_pred)+recall_dl(y_true,y_pred)))
    except:
        traceback.print_exc()

# def single_class_precision(interesting_class_id):
#     def sc_precision(y_true, y_pred):
#         try:
#             class_id_true  =  K.argmax(y_true,  axis=-1) 
#             class_id_preds  =  K.argmax(y_pred,  axis=-1)
#             # Replace class_id_preds with class_id_true for recall here
#             accuracy_mask  =  K.cast(K.equal(class_id_preds,  interesting_class_id),  'int32') 
#             class_acc_tensor  =  K.cast(K.equal(class_id_true,  class_id_preds),  'int32')  *  acc
#             uracy_mask
#             class_acc  =  K.sum(class_acc_tensor)  /  K.maximum(K.sum(accuracy_mask),  1)
#             return class_acc
#         except:
#             traceback.print_exc()
#     return sc_precision


def recall_m(y_true, y_pred):
    try:
        #	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        true_positives= tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred)) 
        possible_positives  =  K.sum(K.round(K.clip(y_true,  0,  1)))
        recall  =  true_positives  /  (possible_positives  +  K.epsilon())
        return recall
    except:
        traceback.print_exc()

def precision_m(y_true, y_pred):
    try:
        true_positives  =  K.sum(K.round(K.clip(y_true  *  y_pred,  0,  1))) 
        predicted_positives  =  K.sum(K.round(K.clip(y_pred,  0,  1)))
        precision  =  true_positives  /  (predicted_positives  +  K.epsilon())
        return precision
    except:
        traceback.print_exc()

def  f1_m(y_true,  y_pred):
    try:
        precision = precision_m(y_true, y_pred) 
        recall = recall_m(y_true, y_pred)
        return  2*((precision*recall)/(precision+recall+K.epsilon()))
    except:
        traceback.print_exc()
def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
#     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)
def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

#     p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

#     f1 = 2*p*r / (p+r+K.epsilon())
# #     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(r)
def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
#     r = tp / (tp + fn + K.epsilon())

#     f1 = 2*p*r / (p+r+K.epsilon())
# #     f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
#     f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(p)

# def clean_text(raw_text):
#     try:
#         if type(raw_text)==None:
#             return 'NAN'
#         try:
#             text=unidecode(raw_text)
#         except:
#             text='NAN'
#         text=str(text).lower() #Normalization
#         text=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text) # Removing Unicode Characters
#         text=str(text).replace("\n","").replace("\r","").replace("\t","").replace(' \n ',' ')
#         text=str(text).strip()

#     #         text = re.sub("https?://.*[\t\r\n]*", "", text)
#         return text
#     except:
#         print(raw_text)
#         traceback.print_exc()
#         return 'NAN'


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline, make_pipeline
import  joblib
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

n_jobs=16

random_state=802
embed=hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def get_random_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    # max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    random_grid = {'clf__n_estimators': n_estimators,
               'clf__max_features': ['sqrt'],
               'clf__max_depth': max_depth,
               'clf__min_samples_split': min_samples_split,
               'clf__min_samples_leaf': min_samples_leaf,
               'clf__bootstrap': bootstrap
               #,'clf__n_features':n_features
    #               ,'vect__ngram_range': [(1,1), (1,2)]
    #            ,'vect__max_df':[0.95]
    #            ,'tfidf__use_idf': (True,False)
               ,"clf__criterion": ["gini", "entropy"]
              }
    # random_grid['vect__ngram_range']= [ ngram_tuple]
    from pprint import pprint

    pprint(random_grid)
    return random_grid

def evaluation_matrix(actual,predicted):
    print("accuracy_score",accuracy_score(actual,predicted))

    conf = confusion_matrix(actual,predicted)
    print(conf)
    tn, fp, fn, tp=confusion_matrix(actual,predicted).ravel()
    # tn, fp, fn, tp=confusion_matrix(list(map(lambda x: np.argmax(x), y_test)), list(map(lambda x: np.argmax(x), y_hat)))
    # evaluation_metrics(tn, fp, fn, tp)

    print(classification_report(actual,predicted))

def train_event_serious_model(df1,train_valid_flag,feature_name ,event_flagname,identifier):
    random_grid=get_random_grid()
    
    df=df1[[ train_valid_flag,feature_name ,event_flagname]]
    print("Input file shape",df.shape)
    df=df[np.logical_or(df[train_valid_flag]=="R",df[train_valid_flag]=="V")]
    print(df[df[train_valid_flag]=="R"][event_flagname].value_counts(dropna=False))
    print(df[df[train_valid_flag]=="V"][event_flagname].value_counts(dropna=False))
    print("Input file shape, after filter on column TRAINING==1",df.shape)
    df['text']=df[feature_name].apply(clean_text)
    print("df[df['text']=='NAN'].shape",df[df['text']=='NAN'].shape)
    
    df.fillna(0,  inplace=True)
    df=df.dropna()
    print("df.shape after dropping NA ",df.shape)

    df=df.reset_index()
    no_variants=['no','nO','No','NO']
    for no_variant in no_variants:
        df.loc[df[event_flagname] == no_variant, event_flagname] = 0
    yes_variants=['yes' ,'yeS' ,'yEs' ,'yES' ,'Yes' ,'YeS' ,'YEs' ,'YES']
    for yes_variant in yes_variants:
        df.loc[df[event_flagname] == yes_variant, event_flagname] = 1

    terms_train=df[df[train_valid_flag]=="R"]
    terms_test=df[df[train_valid_flag]=="V"]

    X_train=terms_train[feature_name]
    X_test=terms_test[feature_name]
    
        #binary_target
    y_train=terms_train[event_flagname].values
    y_train=y_train.astype(int)
    
    y_test=terms_test[event_flagname].values
    y_test=y_test.astype(int)
        
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    
    
    
    
    
#     train,test=train_test_split(df, test_size = 0.2, random_state = random_state,stratify=df[event_flagname])
#     train,val=train_test_split(train, test_size = 0.2, random_state = random_state,stratify=train[event_flagname])
    training_embeddings = embed(X_train.to_list())
    training_embeddings_np=training_embeddings.numpy()
    print("training_embeddings_np.shape",training_embeddings_np.shape)


    rf = Pipeline([
    ('smote', SMOTE(random_state=random_state)),
    ("clf", RandomForestClassifier())
    ])

    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                  n_iter = 5, scoring='f1_micro',
                                  cv = 5, verbose=2, n_jobs=n_jobs,random_state=random_state,
                                  return_train_score=True)

    rf_random.fit(training_embeddings_np
              ,y_train.astype(int));
    print("rf_random.best_params_")
    print(rf_random.best_params_)

    print("rf_random.best_score_")
    print(rf_random.best_score_)
    best_random = rf_random.best_estimator_
    
    joblib.dump(best_random, 'Model_Files/'+event_flagname+'_'+identifier+'.pkl')
    
    test_embedding = embed(X_test.to_list())
#     val_embedding = embed(val.text.to_list())

#     print("Evaluating Val data")
#     y_val=val[event_flagname].astype(int)
#     y_val_pred=best_random.predict(val_embedding)
#     evaluation_matrix(y_val, y_val_pred) 
    
#     y_test=test[event_flagname].astype(int)
    y_test_pred=best_random.predict(test_embedding)
    
    print("Evaluating test data")
    evaluation_matrix(y_test, y_test_pred) 

def clean_text(raw_text):
    try:
        if type(raw_text)==None:
            return 'NAN'
        if raw_text==None:
            return 'NAN'
        if len(raw_text.strip())==0:
            return 'NAN'
        try:
            text=unidecode(raw_text)
        except:
            print("Unidecode raw_text",raw_text)
            traceback.print_exc()
            text='NAN'
        text=str(text).lower() #Normalization
        text=re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text) # Removing Unicode Characters
        text=str(text).replace("\n","").replace("\r","").replace("\t","").replace(' \n ',' ')
        text=str(text).strip()

    #         text = re.sub("https?://.*[\t\r\n]*", "", text)
        return text
    except:
        print(raw_text)
        traceback.print_exc()
        return 'NAN'


import traceback
import numpy as np

event_flagnames=['flag_name']
feature_name= 'feature_name'
train_valid_flag='train_valid_flag'
identifier="identifier_v1"

for event_flagname in event_flagnames:
    train_event_serious_model(    df,train_valid_flag,feature_name ,event_flagname,identifier)
