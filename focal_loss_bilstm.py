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

print(df.shape)
print(df.columns)

#define input
#pre-process meddra training data for LSTM - not needed if pre-trained model is used #function for pre-processing the text input into numbers i.a. a word vector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import  json
import pandas as pd
import numpy as np
import sklearn
import random
import tensorflow as tf
#attention model for working with own embeddings    - but doesn't work if to be used in te nsorflow.js later
from tensorflow.keras.layers import InputSpec,Layer
from  tensorflow.keras  import  backend  as  K
from  tensorflow.keras  import initializers
#continue #attention model for working with own embeddings and not pre-trained model
from tensorflow.keras.optimizers import RMSprop
from  tensorflow.keras.layers  import  Input,  Embedding,  Dense,  LSTM,  Bidirectional,Conv1D,MaxPooling1D
from  tensorflow.keras.layers  import  concatenate,  Reshape,  SpatialDropout1D
from   tensorflow.keras.models   import Model
from  tensorflow.keras  import  backend  as  K
#from .AttentionWeightedAverage import AttentionWeightedAverage

import sklearn
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix,f1_score
from tensorflow.keras.models import model_from_json
import json
import pickle
# from imblearn.pipeline import Pipeline
# from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

n_jobs=25
random_state=802

def evaluate_model(y_test, y_pred):
    # Print the Confusion Matrix and slice it into four pieces

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)

    print('Confusion matrix\n\n', cm)
    # visualize confusion matrix with seaborn heatmap

    cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive', 'Actual Negative'], 
                                     index=['Predict Positive', 'Predict Negative'])

#     sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred))



def apply_str1(value):
    try:
#         print(type(value))
#         print(value[1:])
        return value[1:]
    except:
        traceback.print_exc()
        print("except",value)
        return value


def fill_not_available(dataset,fieldname):
#     print(fieldname)
#     print("before operation")
# #     print(dataset[fieldname].value_counts())
#     print("None",dataset[dataset[fieldname]==None].shape)
#     print("np.NaN",dataset[dataset[fieldname]==np.nan].shape)
#     print("empty string",dataset[dataset[fieldname]==""].shape)
#     print('dataset[fieldname]=="|Not available"]',dataset[dataset[fieldname]=="|Not available"].shape)
    dataset[fieldname]=dataset[fieldname].replace(r'^\s*$', "|Not available(M)", regex=True)
#     print(dataset[dataset[fieldname]=="|Not available(M)"].shape)
    dataset.loc[dataset[fieldname]=="",fieldname]="|Not available(M)"
    dataset.loc[dataset[fieldname]==np.nan,fieldname]="|Not available(M)"
    dataset.loc[dataset[fieldname]==None,fieldname]="|Not available(M)"
    dataset.loc[dataset[fieldname]=="nan",fieldname]="|Not available(M)"
#     print("after operation")
#     print(dataset[dataset[fieldname]=="|Not available(M)"].shape)
#     print(dataset[fieldname].value_counts())

    return dataset
def func_split_set(x,fieldname):
    try:

        split_x=set(str(x).split("|"))
        len_processed=len(split_x)
        len_raw=len(str(x).split("|"))
#         if "|" in str(x) and len_processed<len_raw:
#             print("*****\nraw,",x)
#             print('processed:',"|".join(list(split_x)))
            
        return "|".join(list(split_x))
    except:
        print(fieldname, 'in func')
        traceback.print_exc()

def func_split_set2(x):
    try:
#         print(type(x))
#         print(x)
        split_x=set(str(x).split("|"))
        len_processed=len(split_x)
        len_raw=len(str(x).split("|"))
#         if "|" in str(x) and len_processed<len_raw:
#             print("*****\nraw,",x)
#             print('processed:',"|".join(list(split_x)))
            
        return "|".join(list(split_x))
    except:
        print(x, 'func_split_set2 traceback')
        traceback.print_exc()
        return x
def  get_split_data_for_event_flag(df,event_flagname):

    X = df.drop([
           event_flagname+"_01"], axis=1)

    # for indicator in indicator_fieldname_list:
    # X = X.drop(indicator_fieldname_list, axis=1)

    y = df[event_flagname+"_01"]

    class_weight_full_df=get_class_weights(df,event_flagname)
    print("class_weight_full_df",class_weight_full_df)
    
    
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random_state,stratify=y)
    # check the shape of X_train and X_test
    
#     class_weight_train_df=get_class_weights(df,event_flagname)
    
    print(X_train.shape, X_test.shape)
    
    return X_train, X_test, y_train, y_test,class_weight_full_df

def get_class_weights(df,event_flagname):
    print(df.columns)
    print(df.shape)
    print(df[event_flagname+"_01"].value_counts())
    neg, pos = np.bincount(df[event_flagname+"_01"])
    print("neg, pos",neg, pos)
    total = neg + pos
    print('\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(total, pos, 100 * pos / total))
    
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight

def  get_split_data_for_event_flag(df,event_flagname):

    X = df.drop([
           event_flagname+"_01"], axis=1)

    # for indicator in indicator_fieldname_list:
    # X = X.drop(indicator_fieldname_list, axis=1)

    y = df[event_flagname+"_01"]

    class_weight_full_df=get_class_weights(df,event_flagname)
    print("class_weight_full_df",class_weight_full_df)
    
    
    
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = random_state,stratify=y)
    # check the shape of X_train and X_test
    
#     class_weight_train_df=get_class_weights(df,event_flagname)
    
    print(X_train.shape, X_test.shape)
    
    return X_train, X_test, y_train, y_test,class_weight_full_df



def get_glove_bilstm_score(terms_train,terms_val,terms_test, column_to_process,field_to_train,event_flagname,identifier):
    
    maxlen  =  25
    max_features = 30000

    X_pre  =  terms_train[column_to_process].apply(lambda  x:  str(x).lower())
    X_pre=X_pre.values
        
    val_X_pre=terms_val[column_to_process].apply(lambda  x:  str(x).lower())
    val_X_pre=val_X_pre.values 

    test_X_pre=terms_test[column_to_process].apply(lambda  x:  str(x).lower())
    test_X_pre=test_X_pre.values 

    
    print("X_pre",X_pre)

    #create the tokens
    tokenizer  =  Tokenizer(num_words=max_features,  oov_token='<UNK>') #tokenizer = Tokenizer(num_words=max_features, char_level=True) 
    tokenizer.fit_on_texts(list(X_pre))
    tokenizer_json=tokenizer.word_index
    with  open('Model_Files/dictionary_seriousness_'+event_flagname+'_'+column_to_process+'_'+identifier+'.json',  'w')  as  dictionary_file: 
        json.dump(tokenizer_json,  dictionary_file)

    X = tokenizer.texts_to_sequences(X_pre)
    X  =  pad_sequences(X, maxlen=maxlen)

    val_X = tokenizer.texts_to_sequences(val_X_pre)
    val_X = pad_sequences(val_X, maxlen=maxlen)

    test_X = tokenizer.texts_to_sequences(test_X_pre) 
    test_X = pad_sequences(test_X, maxlen=maxlen)
    

    with  open('Model_Files/tokenizer_seriousness_'+event_flagname+'_'+column_to_process+'_'+identifier+'_individual_v2.json',  'wb')  as  tokenizer_file: 
        pickle.dump(tokenizer, tokenizer_file, protocol=pickle.HIGHEST_PROTOCOL)        


    #binary_target
    Y=terms_train[field_to_train].values
    Y=Y.astype(int)

    val_Y=terms_val[field_to_train].values 
    val_Y=val_Y.astype(int)
    
    test_Y=terms_test[field_to_train].values 
    test_Y=test_Y.astype(int)
    

    maxlen  =  25
    max_features = 30000

    config = {
    'rnn_layers':  3,
    'rnn_size':  64,
    'rnn_bidirectional': True, 'max_length': maxlen,
    'max_words': max_features, 'dim_embeddings':  300,
    'word_level': True, 
        'single_text': False
    }
    default_config = config.copy()

    epochs=7
    batch_size=128

    model= textgenrnn_model(max_features,cfg=config,weights_path=None)
    print(model.summary())

    model.compile(loss=[focal_loss],  optimizer='adam',  metrics=['accuracy',f1,f1_m,precision_m, recall_m, precision_dl,recall_dl,f1_dl])
    
    print("X.shape,Y.shape,test_X.shape,test_Y.shape",X.shape,Y.shape,test_X.shape,test_Y.shape)
    
    model.fit(X,  Y,  validation_data=(val_X,  val_Y),  epochs=epochs,  batch_size=batch_size, verbose=1)


    fname='Model_Files/model_'+event_flagname+'_'+column_to_process+'_'+identifier+'_25input_3x64bi_attn_300emb_3epochs.h5'
    model.save(fname)

    y_hat  =  model.predict(test_X,batch_size=batch_size,verbose=1)
    print(type(y_hat))
    print(y_hat[0])
    y_pred=np.where(y_hat < 0.4, 0, 1)
    print(pd.DataFrame(y_pred).value_counts())

    print('Confusion matrix\n\n', sklearn.metrics.confusion_matrix(test_Y, y_pred))

    print('Classification Report\n\n',sklearn.metrics.classification_report(test_Y, y_pred))
    
    
    feature=column_to_process
    terms_train[feature+'_score']=model.predict(X)
    terms_test[feature+'_score']=y_hat

    return terms_train,terms_test

def get_random_grid():
    
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]
    # Number of features to consider at every split
    max_features = [ 'sqrt']
    #max_features = [int(x) for x in np.linspace(start = 5000, stop = 10000, num = 5)]
    # Maximum number of levels in tree
    #     max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth = [int(x) for x in np.linspace(10, 50, num = 5)]
    #     max_depth.append(None)
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
    #                   ,'vect__ngram_range': [(1,1), (1,2)]
    #                ,'vect__max_df':[0.95]
    #                ,'tfidf__use_idf': (True,False)
               ,"clf__criterion": ["gini", "entropy"]
              }
    # random_grid['vect__ngram_range']= [ ngram_tuple]
    

    pprint(random_grid)
    return random_grid

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

def single_class_precision(interesting_class_id):
    def sc_precision(y_true, y_pred):
        try:
            class_id_true  =  K.argmax(y_true,  axis=-1) 
            class_id_preds  =  K.argmax(y_pred,  axis=-1)
            # Replace class_id_preds with class_id_true for recall here
            accuracy_mask  =  K.cast(K.equal(class_id_preds,  interesting_class_id),  'int32') 
            class_acc_tensor  =  K.cast(K.equal(class_id_true,  class_id_preds),  'int32')  *  acc
            uracy_mask
            class_acc  =  K.sum(class_acc_tensor)  /  K.maximum(K.sum(accuracy_mask),  1)
            return class_acc
        except:
            traceback.print_exc()
    return sc_precision


def recall_m(y_true, y_pred):
    try:
        #   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
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

#custom loss function in order to put more weight on the class that is hard to predict
def focal_loss(y_true,  y_pred): 
    gamma = 2.0 #default 2.0
    alpha = 0.25 #default 0.25
    mult_1   = 1
    mult_0   = 1
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred)) 
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    # return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.po w( pt_0, gamma) * K.log(1. - pt_0))
    return  -(mult_1*K.sum(alpha  *  K.pow(1-  pt_1,  gamma)  *  K.log(pt_1)))-(mult_0*K.sum((1-alpha)  *  K.pow(  pt_0,  gamma)  *  K.log(1.  -  pt_0)))


class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for    a single timestep.
    """

    def __init__(self,  return_attention=False,  **kwargs): 
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def get_config(self):
        config = super().get_config().copy() 
        config.update({
        'return_attention': self.return_attention})
        return config
    def build(self, input_shape):
        self.input_spec  =  [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W  =  self.add_weight(shape=(input_shape[2],  1),
        name='{}_W'.format(self.name), initializer=self.init)
        self.trainable_weights2  =  [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)


    def  call(self,  x,  mask=None):
    # computes a probability distribution over the timesteps # uses 'max trick' for numerical stability
    # reshape is done to avoid issue with Tensorflow  # and 1-dimensional weights
        logits  =  K.dot(x,  self.W) 
        x_shape  =  K.shape(x)
        logits  =  K.reshape(logits,  (x_shape[0],  x_shape[1]))
        ai  =  K.exp(logits  -  K.max(logits,  axis=-1,  keepdims=True))

        # masked timesteps have zero weight
        if  mask  is  not  None:
            mask  =  K.cast(mask,  K.floatx()) 
            ai = ai * mask
        att_weights  =  ai  /  (K.sum(ai,  axis=1,  keepdims=True)  +  K.epsilon()) 
        weighted_input  =  x  *  K.expand_dims(att_weights)

        result  =  K.sum(weighted_input,  axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape): 
        output_len  =  input_shape[2]
        if self.return_attention:
            return  [(input_shape[0],  output_len),  (input_shape[0],
    input_shape[1])]
        return  (input_shape[0],  output_len)

    def  compute_mask(self,  input,  input_mask=None):
        if isinstance(input_mask, list):
            return  [None]  *  len(input_mask)
        else:
            return None
        
def  textgenrnn_model(num_classes,  cfg,  context_size=None,weights_path=None, dropout=0.3,
                      optimizer=RMSprop(learning_rate=4e-3,  rho=0.99)):
    '''
    Builds the model architecture for textgenrnn and loads the specified weights for the model.
    '''


    input  =  Input(shape=(cfg['max_length'],),  name='input')
    # input = Input(shape=(0,), name='input')
    embedded  =  Embedding(num_classes,  cfg['dim_embeddings']
                           ,input_length=cfg['max_length'], name='embedding')(input)

    if dropout > 0.0:
        embedded = SpatialDropout1D(dropout, name='dropout')(embedded)

    #custom convolution  layer  add #
    #conv1 =  Conv1D(filters=32, #  kernel_size=8,
    #   strides=1,
    #   activation='relu')(embedded)
    #embedded = MaxPooling1D(pool_size=4)(conv1) #end of addting custom convolution layer

    rnn_layer_list = []
    for i in range(cfg['rnn_layers']):
        prev_layer  =  embedded  if  i  is  0  else  rnn_layer_list[-1] 
        rnn_layer_list.append(new_rnn(cfg,  i+1)(prev_layer))

    seq_concat  =  concatenate([embedded]  +  rnn_layer_list,  name='rnn_concat') 
    attention = AttentionWeightedAverage(name='attention')(seq_concat)
    #output = Dense(le.classes_.shape[0], name='output', activation='softmax')(attention)
    output  =  Dense(1,  name='output',  activation='sigmoid')(attention)
    
    if  context_size  is  None:
        model  =  Model(inputs=[input],  outputs=[output])
        if  weights_path  is  not  None:
            model.load_weights(weights_path, by_name=True)
        #   model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    else:
        context_input = Input(
            shape=(context_size,), name='context_input') 
        context_reshape = Reshape((context_size,),
            name='context_reshape')(context_input) 
        merged = concatenate([attention, context_reshape], name='concat')
#   main_output = Dense(num_classes,  name='context_output', #      activation='softmax')(merged)
        main_output  =  Dense(1,  name='context_output',
                        activation='sigmoid')(merged)

        model  =  Model(inputs=[input,  context_input],
                    outputs=[main_output,  output])
        if  weights_path  is  not  None:
            model.load_weights(weights_path, by_name=True)
        #   model.compile(loss='categorical_crossentropy', optimizer=optimizer, #       loss_weights=[0.8, 0.2])

    return model


def new_rnn(cfg, layer_num):
    #use_cudnnlstm = K.backend() == 'tensorflow' and len(K.tensorflow_backend._get_avail able_gpus()) > 0
    #if use_cudnnlstm:
    #   from keras.layers import CuDNNLSTM #    if cfg['rnn_bidirectional']:
    #   return Bidirectional(CuDNNLSTM(cfg['rnn_size'],
    #   return_sequences=True),
    #   name='rnn_{}'.format(layer_num))
    #
    #   return CuDNNLSTM(cfg['rnn_size'],
    #   return_sequences=True,
    #   name='rnn_{}'.format(layer_num))                              #else:
    if cfg['rnn_bidirectional']:
        return  Bidirectional(LSTM(cfg['rnn_size'],
                return_sequences=True,
                recurrent_activation='sigmoid'), name='rnn_{}'.format(layer_num))

    return  LSTM(cfg['rnn_size'],
            return_sequences=True,
            recurrent_activation='sigmoid', name='rnn_{}'.format(layer_num))

from sklearn.model_selection import train_test_split
import re
from unidecode import unidecode
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

def train_classification_model(df1,train_valid_flag,feature_name ,event_flagname):
    df=df1[[ train_valid_flag,feature_name ,event_flagname]]
    print("Input file shape",df.shape)
    df=df[np.logical_or(df[train_valid_flag]=="R",df[train_valid_flag]=="V")]
    print(df[df[train_valid_flag]=="R"][event_flagname].value_counts(dropna=False))
    print(df[df[train_valid_flag]=="V"][event_flagname].value_counts(dropna=False))
    print("Input file shape, after filter on column TRAINING==1",df.shape)
    df['text']=df[feature_name].apply(clean_text)
    print(df[df['text']=='NAN'].shape)
    # df['text']=df.apply(append_event_verbatim,axis=1)
    df.fillna(0,  inplace=True)
    df=df.dropna()
    print(df.shape)

    df=df.reset_index()
    no_variants=['no','nO','No','NO']
    for no_variant in no_variants:
        df.loc[df[event_flagname] == no_variant, event_flagname] = 0
    yes_variants=['yes' ,'yeS' ,'yEs' ,'yES' ,'Yes' ,'YeS' ,'YEs' ,'YES']
    for yes_variant in yes_variants:
        df.loc[df[event_flagname] == yes_variant, event_flagname] = 1

    terms_train=df[df[train_valid_flag]=="R"]
    terms_test=df[df[train_valid_flag]=="V"]

    terms_train,terms_val =train_test_split(terms_train, test_size = 0.2, random_state = random_state,stratify=terms_train[event_flagname])

    print(terms_train.shape,terms_val.shape,terms_test.shape)

    feature='text'
    column_to_process=feature
    field_to_train=event_flagname
    print(feature)


    terms_train,terms_test= get_glove_bilstm_score(terms_train,terms_val,terms_test, column_to_process,field_to_train,event_flagname,identifier)


    print("terms_train.columns",terms_train.columns)
    print("terms_test.columns",terms_test.columns)
    terms_train.to_pickle("Model_Files/terms_train"+event_flagname+"_"+identifier+".pkl")
    terms_test.to_pickle("Model_Files/terms_test"+event_flagname+"_"+identifier+".pkl")

  
import traceback
import numpy as np
event_flagnames=['flag_name']
feature_name= 'feature_name'
train_valid_flag='train_valid_flag'
identifier="identifier_v1"
for event_flagname in event_flagnames:
    train_classification_model(df,train_valid_flag,feature_name ,event_flagname)
