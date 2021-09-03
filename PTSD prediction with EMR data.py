
#**********************************************************************************************************************
#*                            #    Loading the data, sampling and spliting                                            *
#**********************************************************************************************************************
import numpy as np
import pandas as pd


df_st_nt_PTSD = pd.read_csv('/PTSD.csv')
y=df_st_nt_PTSD[['PTSD']]



from sklearn.model_selection import train_test_split
#----------------------------------------- set aside a subset of the data as hold-out test dataset --------------------
X_tobe_RUS,X_test_holdout,y_tobe_RUS,y_test_holdout = train_test_split(df_st_nt_PTSD,y, test_size=0.15, random_state=40)

#----------------------------------------- create a balanced training dataset using development dataset ------------------------------
pid_sbsmple=X_tobe_RUS.idx # this is development dataset
list_of_idx = X_tobe_RUS['idx'].to_list()
df_tot_tobe_resampled=df_st_nt_PTSD[df_st_nt_PTSD.idx.isin(list_of_idx)]
df=df_tot_tobe_resampled
#1. Find Number of samples which are positive
PTSD_pos_cnt = len(df[df['PTSD'] == 1])
#2. Get indices of negative and positive samples
PTSD_neg_indices = df[df.PTSD == 0].index
PTSD_pos_indices = df[df.PTSD == 1].index
#3. Random sample negative indices
random_indices = np.random.choice(PTSD_neg_indices,PTSD_pos_cnt, replace=False)
#4. Concat positive indices with the sampled negative ones
under_sample_indices = np.concatenate([PTSD_pos_indices,random_indices])
#5. Get Balance Dataframe
under_sample = df.loc[under_sample_indices]
X_RUS_train = under_sample.loc[:,under_sample.columns != 'PTSD']
y_RUS_train = under_sample.loc[:,under_sample.columns == 'PTSD']
#--------------------- choose negative samples to create a skewed dataset --------------
# select those records that are not in the balanced sampled dataset
only_negative_sample=df_tot_tobe_resampled[~df_tot_tobe_resampled.index.isin(under_sample.index)]
PTSD_neg_indices_skewed = only_negative_sample.index
skewed_cnt=1700
random_indices_skewed = np.random.choice(PTSD_neg_indices_skewed,skewed_cnt, replace=False)
extra_negative_samples_for_test_crossvalidation=only_negative_sample.loc[random_indices_skewed]

#**********************************************************************************************************************
#*                                                   functions                                                        *
#**********************************************************************************************************************

from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

import spacy
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
spacy.load('en')
lemmatizer = spacy.lang.en.English()

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPool1D, Flatten, Input,GlobalMaxPooling1D,GlobalMaxPool1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from keras.layers.merge import concatenate
from numpy import array
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from keras import layers

def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens if len(token)>1])

def clean_all_but_alphabet(s):
  out = re.sub(r'[^a-zA-Z\s]', ' ', s)
  out=re.sub("\s\s+", " ", out)
  out=out.lower()
  return(out)

def wm2df(wm, feat_names):
    # create an index for each row
    doc_names = ['Doc{:d}'.format(idx) for idx, _ in enumerate(wm)]
    df = pd.DataFrame(data=wm.toarray(), index=doc_names,columns=feat_names)
    return(df)


def show_metrics(y_test, y_pred):
    #print('confusion matrix: \n', confusion_matrix(y_test, y_pred))
    auc_val= roc_auc_score(y_test, y_pred, average='macro')
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

    AUC= roc_auc_score(y_test, y_pred, average='macro')
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    # Sensitivity, hit rate, recall, or true positive rate
    SN = TP/(TP+FN)
    # Specificity or true negative rate
    SP = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)

    F_score=2*(PPV*SN)/(PPV+SN)

    metrics=[round(PPV,2),round(NPV,2),round(SN,2),round(SP,2),round(ACC,2),round(F_score,2),round(AUC,2)]
    return metrics
	
def getMaxThreshold(df_sum_vector):
  df_sum_vector_sorted = df_sum_vector.sort_values(['F1'], ascending=False)
  return df_sum_vector_sorted.iloc[0]['threshold']	

#**********************************************************************************************************************
#*                                                     ST_Data MLNN model                                             *
#**********************************************************************************************************************


cols=7 # number of mertrics
rows=85
sum_vector= [[0]*cols]*rows
fold=0
cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):
  fold+=1
  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note'],axis=1)
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note'],axis=1)
  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','PTSD'],axis=1)
  y_st_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD


  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_st_test_extra_neg_i = y_st_test_extra_neg
  y_test=pd.concat([y_test, y_st_test_extra_neg_i],axis=0)


  from sklearn.neural_network import MLPClassifier
  
  seed(1)
  random.set_seed(1)
  input_st = Input(shape=(399,))
  dense_layer_1 = Dense(50, activation='relu')(input_st)
  dense_layer_2 = Dense(50, activation='relu')(dense_layer_1)
  dense_layer_3 = Dense(50, activation='relu')(dense_layer_2)
  dense20_st = Dense(20, activation='relu')(dense_layer_3)
  dense1_st = Dense(1, activation='sigmoid')(dense20_st)
  
  X_st_train = np.asarray(X_st_train).astype(np.float32)

  model_mlp = Model(inputs=[input_st], outputs=dense1_st)
  # compile
  model_mlp.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

  model_mlp.fit(X_st_train, array(y_train), epochs=20, batch_size=16 ,verbose=1)

  
  y_pred = model_mlp.predict(X_test_st_skewed, verbose=False)
  
  threshould=0.05
  while(threshould<0.95):
		threshold+=0.01
        y_pred_rounded = (y_pred > threshould)
        cm=confusion_matrix(y_test, y_pred_rounded)  
        metrics=show_metrics(y_test, y_pred_rounded)
        sum_vector[i]=sum_vector[i]+np.array(metrics)
        print(metrics)
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(sum_vector)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()

threshould_max=getMaxThreshold(sum_vector) # threshould_max is the threshold value that gets the maximum F1-score value

# apply the model on the HOLDOUT test data 

X_test_holdout=X_test_holdout.drop(['idx','note'],axis=1)
y_test_holdout=y_test_holdout

y_pred_rf = RF_model_nt.predict_proba(X_test_holdout)[::,1]

y_pred_rounded = (y_pred_rf > threshould)
cm=confusion_matrix(y_test_holdout, y_pred_rounded)  
metrics=show_metrics(y_test_holdout, y_pred_rounded)
print(metrics)


#**********************************************************************************************************************
#*                                                    ST_Data RF model                                                *
#**********************************************************************************************************************

cols=7 # number of mertrics
rows=85
sum_vector= [[0]*cols]*rows
fold=0
cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):
  fold+=1
  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note'],axis=1)
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note'],axis=1)
  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','PTSD'],axis=1)
  y_st_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD


  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_st_test_extra_neg_i = y_st_test_extra_neg
  y_test=pd.concat([y_test, y_st_test_extra_neg_i],axis=0)


  from sklearn.ensemble import RandomForestClassifier
  RF= RandomForestClassifier(n_estimators=100)

  RF_model_st = RF.fit(X_st_train, y_train)
  y_pred = RF_model_nt.predict_proba(X_test_st_skewed)[::,1]
  
  threshould=0.1
  while(threshould<0.95):
		threshold+=0.01
        y_pred_rounded = (y_pred > threshould)
        cm=confusion_matrix(y_test, y_pred_rounded)  
        metrics=show_metrics(y_test, y_pred_rounded)
        sum_vector[i]=sum_vector[i]+np.array(metrics)
        print(metrics)
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(sum_vector)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()

threshould_max=getMaxThreshold(sum_vector) # threshould_max is the threshold value that gets the maximum F1-score value

# apply the model on the HOLDOUT test data 

X_test_holdout=X_test_holdout.drop(['idx','note'],axis=1)
y_test_holdout=y_test_holdout

y_pred_rf = RF_model_nt.predict_proba(X_test_holdout)[::,1]

y_pred_rounded = (y_pred_rf > threshould)
cm=confusion_matrix(y_test_holdout, y_pred_rounded)  
metrics=show_metrics(y_test_holdout, y_pred_rounded)
print(metrics)


#**********************************************************************************************************************
#*                                                 Note_Data BOW RF                                                   *
#**********************************************************************************************************************

cols=7 # number of mertrics
rows=len(threshoulds)
sum_vector= [[0]*cols]*rows
fold=0
cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):
  fold+=1
  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note'],axis=1)
  X_nt_train=X_RUS_train.iloc[train].note.apply(clean_all_but_alphabet)
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note'],axis=1)
  X_nt_test=X_RUS_train.iloc[test].note.apply(clean_all_but_alphabet)

  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','PTSD'],axis=1)
  X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note.apply(clean_all_but_alphabet)
  y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD


  X_test_nt_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)


  custom_vec = TfidfVectorizer(tokenizer=my_tokenizer,
                            analyzer='word',
                            ngram_range=(1,3),
                            stop_words='english',
                            min_df=0.005)
  wm_train = custom_vec.fit_transform(X_nt_train)
  wm_test = custom_vec.transform(X_test_nt_skewed)
  tokens = custom_vec.get_feature_names()
  df_wm_train=wm2df(wm_train,tokens)
  df_wm_test=wm2df(wm_test,tokens)

  from sklearn.ensemble import RandomForestClassifier
  RF= RandomForestClassifier(n_estimators=100)

  RF_model_nt = RF.fit(df_wm_train, y_train)
  y_pred = RF_model_nt.predict_proba(df_wm_test)[::,1]
  
  threshould=0.05
  while(threshould<0.95):
		threshold+=0.01
        y_pred_rounded = (y_pred > threshould)
        cm=confusion_matrix(y_test, y_pred_rounded)  
        metrics=show_metrics(y_test, y_pred_rounded)
        sum_vector[i]=sum_vector[i]+np.array(metrics)
        print(metrics)
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(threshoulds)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()

threshould_max=getMaxThreshold(sum_vector) # threshould_max is the threshold value that gets the maximum F1-score value

# apply the model on the HOLDOUT test data 

X_test_holdout=X_test_holdout.note.apply(clean_all_but_alphabet)
X_test_holdout=[note for note in  X_test_holdout]
y_test_holdout=y_test_holdout

wm_test_holdout = custom_vec.transform(X_test_holdout)
df_wm_test_holdout=wm2df(wm_test_holdout,tokens)
y_pred_rf = RF_model_nt.predict_proba(df_wm_test_holdout)[::,1]

y_pred_rounded = (y_pred_rf > threshould)
cm=confusion_matrix(y_test_holdout, y_pred_rounded)  
metrics=show_metrics(y_test_holdout, y_pred_rounded)
print(metrics)




#**********************************************************************************************************************
#*                                                 Note_Data CNN model                                                *
#**********************************************************************************************************************


fold=0
cols=7 # number of mertrics

sum_vector= [[0]*cols]*85

cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):

  print("----------",fold,"-----------")

  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note'],axis=1)
  X_nt_train=X_RUS_train.iloc[train].note.apply(clean_all_but_alphabet)
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note'],axis=1)
  X_nt_test=X_RUS_train.iloc[test].note.apply(clean_all_but_alphabet)

  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','PTSD'],axis=1)
  #X_nt_test_extra_neg=df_st_nt_just_negative_PTSD.note
  X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note.apply(clean_all_but_alphabet)
  y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD

  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  idx_test_extra=pd.concat([X_RUS_train.iloc[test].idx,extra_negative_samples_for_test_crossvalidation.idx])

  X_test_nt_skewed=pd.concat([X_nt_test, X_nt_test_extra_neg])

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test_holdout=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)

  max_len = max([len(s.split()) for s in X_nt_train])
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_nt_train)
  vocab_size = len(tokenizer.word_index) + 1
  X_train_encoded = tokenizer.texts_to_sequences(X_nt_train)
  X_test_encoded = tokenizer.texts_to_sequences(X_test_nt_skewed)
  # pad sequences
  X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len, padding='post')
  X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

  #--------------------------------------- note data: kernel size=1 -------------------------------
  inputs1 = Input(shape=(max_len,))
  embedding1 = Embedding(vocab_size, 100)(inputs1)
  conv1 = Conv1D(filters=64, kernel_size=1, activation='relu')(embedding1)
  drop1 = Dropout(0.5)(conv1)
  pool1 =layers.GlobalMaxPool1D()(drop1)
  #flat1 = Flatten()(pool1)
  #--------------------------------------- note data: kernel size=2 -------------------------------
  # channel 2
  inputs2 = Input(shape=(max_len,))
  embedding2 = Embedding(vocab_size, 100)(inputs2)
  conv2 = Conv1D(filters=64, kernel_size=2, activation='relu')(embedding2)
  drop2 = Dropout(0.5)(conv2)
  pool2 =layers.GlobalMaxPool1D()(drop2)
  #--------------------------------------- note data: kernel size=3 -------------------------------
  # channel 3
  inputs3 = Input(shape=(max_len,))
  embedding3 = Embedding(vocab_size, 100)(inputs3)
  conv3 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding3)
  drop3 = Dropout(0.5)(conv3)
  pool3 =layers.GlobalMaxPool1D()(drop3)
  # merge
  #--------------------------------------- merge all of the channels -------------------------------
  merged = concatenate([pool1, pool2, pool3])
  # interpretation
  dense1 = Dense(10, activation='relu')(merged)
  outputs = Dense(1, activation='sigmoid')(dense1)
  model_cnn_3k = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
  # compile
  model_cnn_3k.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

  #########################################################################################

  model_cnn_3k.fit([X_train_padded,X_train_padded,X_train_padded], array(y_train), epochs=9, batch_size=16 ,verbose=1)
  
  y_pred_cnn = model_cnn_3k.predict([X_test_padded,X_test_padded,X_test_padded], verbose=False)

  pid=X_RUS_train.iloc[test]['idx'].to_list()
  note_score = pd.DataFrame(
    {'fold':[fold for i in y_pred_cnn.tolist()],
     'idx': idx_test_extra,
     'score': [i[0] for i in y_pred_cnn.tolist()]})
  cnn_score_train=cnn_score_train.append(note_score)




 threshould=0.05
  while(threshould<0.95):
	threshold+=0.01
    y_pred_cnn_rounded = (y_pred_cnn > threshould)
    cm=confusion_matrix(y_test_holdout, y_pred_cnn_rounded)  
    metrics=show_metrics(y_test_holdout, y_pred_cnn_rounded)
    sum_vector[threshould]=sum_vector[threshould]+np.array(metrics)
    print(cm)
    metrics_rounded = [round(num, 2) for num in metrics]
    print(metrics_rounded)

  fold+=1
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(sum_vector)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()


threshould_max=getMaxThreshold(sum_vector) # threshould_max is the threshold value that gets the maximum F1-score value

#-------TEST with the holdout data--------------------

X_nt_holdout=X_test_holdout.note
X_st_holdout=X_test_holdout.drop(['idx','note'],axis=1)

X_test_encoded = tokenizer.texts_to_sequences(X_nt_holdout)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

y_pred_cnn_nt_holdout = model_cnn_3k.predict([X_test_padded,X_test_padded,X_test_padded], verbose=False)


y_pred_cnn_nt_holdout_rounded = (y_pred_cnn_nt_holdout > threshould_max)
cm=confusion_matrix(y_test_holdout, y_pred_cnn_nt_holdout_rounded)  
metrics=show_metrics(y_test_holdout, y_pred_cnn_nt_holdout_rounded)
print(metrics)

pid=X_test_holdout['idx'].to_list()
note_score_holdout = pd.DataFrame(
    {'idx': pid,
     'score': [i[0] for i in y_pred_cnn_nt_holdout.tolist()]})




#**********************************************************************************************************************
#*                                               Mixed_Data MLNN (Parallel)                                           *
#**********************************************************************************************************************
fold=0
cols=7 # number of mertrics
rows=85

sum_vector= [[0]*cols]*rows

cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):
  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note','recNo','rand','birthmonth'],axis=1)
  X_nt_train=X_RUS_train.iloc[train].note#.apply(clean_all_but_alphabet)
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note','recNo','rand','birthmonth'],axis=1)
  X_nt_test=X_RUS_train.iloc[test].note#.apply(clean_all_but_alphabet)

  X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','recNo','rand','birthmonth','PTSD'],axis=1)
  X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note#.apply(clean_all_but_alphabet)
  y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD

  X_test_st_skewed=pd.concat([X_st_test, X_st_test_extra_neg])
  idx_test_extra=pd.concat([X_RUS_train.iloc[test].idx,extra_negative_samples_for_test_crossvalidation.idx])

  X_test_nt_skewed=pd.concat([X_nt_test, X_nt_test_extra_neg])

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  y_nt_test_extra_neg_i = y_nt_test_extra_neg
  y_test_holdout=pd.concat([y_test, y_nt_test_extra_neg_i],axis=0)

  #--------------------------------------- st data -------------------------------
  #issue
  seed(1)
  random.set_seed(1)
  input_st = Input(shape=(339,))
  dense_layer_1 = Dense(50, activation='relu')(input_st)
  dense_layer_2 = Dense(50, activation='relu')(dense_layer_1)
  dense_layer_3 = Dense(50, activation='relu')(dense_layer_2)
  dense20_st = Dense(20, activation='relu')(dense_layer_3)
  dense1_st = Dense(1, activation='sigmoid')(dense20_st)


  ########################################### three kernels ###############################
  max_len = max([len(s.split()) for s in X_nt_train])
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_nt_train)
  vocab_size = len(tokenizer.word_index) + 1
  X_train_encoded = tokenizer.texts_to_sequences(X_nt_train)
  X_test_encoded = tokenizer.texts_to_sequences(X_test_nt_skewed)
  # pad sequences
  X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len, padding='post')
  X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

  #--------------------------------------- note data: kernel size=1 -------------------------------
  seed(1)
  random.set_seed(1)
  inputs1 = Input(shape=(max_len,))
  embedding1 = Embedding(vocab_size, 100)(inputs1)
  conv1 = Conv1D(filters=64, kernel_size=1, activation='relu')(embedding1)
  drop1 = Dropout(0.5)(conv1)
  pool1 =layers.GlobalMaxPool1D()(drop1)
  #--------------------------------------- note data: kernel size=2 -------------------------------
  # channel 2
  seed(1)
  random.set_seed(1)
  inputs2 = Input(shape=(max_len,))
  embedding2 = Embedding(vocab_size, 100)(inputs2)
  conv2 = Conv1D(filters=64, kernel_size=2, activation='relu')(embedding2)
  drop2 = Dropout(0.5)(conv2)
  pool2 =layers.GlobalMaxPool1D()(drop2)

  #--------------------------------------- note data: kernel size=3 -------------------------------
  # channel 3
  seed(1)
  random.set_seed(1)
  inputs3 = Input(shape=(max_len,))
  embedding3 = Embedding(vocab_size, 100)(inputs3)
  conv3 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding3)
  drop3 = Dropout(0.5)(conv3)
  pool3 =layers.GlobalMaxPool1D()(drop3)
  # merge
  #--------------------------------------- merge all of the channels -------------------------------
  merged_nt = concatenate([pool1, pool2, pool3])
  # interpretation
  dense10_nt = Dense(10, activation='relu')(merged_nt)
  output_nt = Dense(1, activation='sigmoid')(dense10_nt)
  merged_mixed = concatenate([dense1_st,output_nt])
  init_output = Dense(2, activation='relu')(merged_mixed)
  output = Dense(1, activation='sigmoid')(init_output)

  model_cnn_3k_plus_st = Model(inputs=[input_st,inputs1, inputs2, inputs3], outputs=output_nt)
  # compile
  model_cnn_3k_plus_st.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


  #########################################################################################
  X_test_st_skewed = np.asarray(X_test_st_skewed).astype(np.float32)
  X_st_train = np.asarray(X_st_train).astype(np.float32)

  model_cnn_3k_plus_st.fit([X_st_train,X_train_padded,X_train_padded,X_train_padded], array(y_train), epochs=10, batch_size=16 ,verbose=1)
  
  y_pred_cnn = model_cnn_3k_plus_st.predict([X_test_st_skewed,X_test_padded,X_test_padded,X_test_padded], verbose=False)


  for loop in range(len(threshoulds)):
    threshould=threshoulds[loop]
    print("------------------------------ threshould: ",threshould,"---------------------------------",fold,"-----------")
    y_pred_cnn_rounded = (y_pred_cnn > threshould)
    cm=confusion_matrix(y_test_holdout, y_pred_cnn_rounded)  
    metrics=show_metrics(y_test_holdout, y_pred_cnn_rounded)
    sum_vector[loop]=sum_vector[loop]+np.array(metrics)
    print(cm)

    metrics_rounded = [round(num, 2) for num in metrics]
    print(metrics_rounded)

  fold+=1
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(threshoulds)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()

threshould_max=getMaxThreshold(sum_vector) # threshould_max is the threshold value that gets the maximum F1-score value



#-------TEST with hold out data-------------------- parallel mixed CNN model with three kernels and st data , ten folds

X_nt_holdout=X_test_holdout.note
X_st_holdout=X_test_holdout.drop(['idx','note'],axis=1)

X_test_encoded = tokenizer.texts_to_sequences(X_nt_holdout)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

X_st_holdout = np.asarray(X_st_holdout).astype(np.float32)
y_pred_cnn_nt_holdout = model_cnn_3k_plus_st.predict([X_st_holdout,X_test_padded,X_test_padded,X_test_padded], verbose=False)
y_pred_cnn_nt_holdout_rounded = (y_pred_cnn_nt_holdout > threshould)
cm=confusion_matrix(y_test_holdout, y_pred_cnn_nt_holdout_rounded)  
metrics=show_metrics(y_test_holdout, y_pred_cnn_nt_holdout_rounded)
print(metrics)
print(cm)
  

#**********************************************************************************************************************
#*                                              Mixed_Data RF (Serial)                                                *
#**********************************************************************************************************************
fold=0
cols=7 # number of mertrics
rows=len(threshoulds)


cv = StratifiedKFold(n_splits=10)
for train,test in cv.split(X_RUS_train,y_RUS_train):

  print("-----------------------------",fold,"---------------------------------")
  fold+=1
  pid=X_RUS_train.iloc[train].idx
  X_st_train=X_RUS_train.iloc[train].drop(['idx','note'],axis=1)
  X_nt_train=X_RUS_train.iloc[train].note
  X_st_test=X_RUS_train.iloc[test].drop(['idx','note'],axis=1)
  X_nt_test=X_RUS_train.iloc[test].note

  y_train=y_RUS_train.iloc[train].PTSD
  y_test=y_RUS_train.iloc[test].PTSD
  

  ########################################### three kernels ###############################
  max_len = max([len(s.split()) for s in X_nt_train])
  tokenizer = Tokenizer()
  tokenizer.fit_on_texts(X_nt_train)
  vocab_size = len(tokenizer.word_index) + 1
  X_train_encoded = tokenizer.texts_to_sequences(X_nt_train)
  X_test_encoded = tokenizer.texts_to_sequences(X_nt_test)
  # pad sequences
  X_train_padded = pad_sequences(X_train_encoded, maxlen=max_len, padding='post')
  X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

  #--------------------------------------- note data: kernel size=1 -------------------------------
  # channel 1
  inputs1 = Input(shape=(max_len,))
  embedding1 = Embedding(vocab_size, 100)(inputs1)
  conv1 = Conv1D(filters=64, kernel_size=1, activation='relu')(embedding1)
  drop1 = Dropout(0.5)(conv1)
  pool1 =layers.GlobalMaxPool1D()(drop1)
  #--------------------------------------- note data: kernel size=2 -------------------------------
  # channel 2
  inputs2 = Input(shape=(max_len,))
  embedding2 = Embedding(vocab_size, 100)(inputs2)
  conv2 = Conv1D(filters=64, kernel_size=2, activation='relu')(embedding2)
  drop2 = Dropout(0.5)(conv2)
  pool2 =layers.GlobalMaxPool1D()(drop2)
  #flat2 = Flatten()(pool2)
  #--------------------------------------- note data: kernel size=3 -------------------------------
  # channel 3
  inputs3 = Input(shape=(max_len,))
  embedding3 = Embedding(vocab_size, 100)(inputs3)
  conv3 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding3)
  drop3 = Dropout(0.5)(conv3)
  pool3 =layers.GlobalMaxPool1D()(drop3)
  #--------------------------------------- merge all of the channels -------------------------------
  merged = concatenate([pool1, pool2, pool3])
  # interpretation
  dense1 = Dense(10, activation='relu')(merged)
  outputs = Dense(1, activation='sigmoid')(dense1)
  model_cnn_3k = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
  # compile
  model_cnn_3k.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

  #########################################################################################


  model_cnn_3k.fit([X_train_padded,X_train_padded,X_train_padded], array(y_train), epochs=10, batch_size=16 ,verbose=1)
  
  y_pred_cnn = model_cnn_3k.predict([X_test_padded,X_test_padded,X_test_padded], verbose=False)

  pid=X_RUS_train.iloc[test]['idx'].to_list()
  note_score = pd.DataFrame(
    {'fold':[fold for i in y_pred_cnn.tolist()],
     'idx': X_RUS_train.iloc[test].idx,
     'score': [i[0] for i in y_pred_cnn.tolist()]})
  cnn_score_train=cnn_score_train.append(note_score)
  
# end of cross validation loop
 
X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','PTSD'],axis=1)
X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note
y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD

X_nt_test_extra_neg_encoded = tokenizer.texts_to_sequences(X_nt_test_extra_neg)
X_nt_test_extra_neg_padded = pad_sequences(X_nt_test_extra_neg_encoded, maxlen=max_len, padding='post')

y_pred_cnn = model_cnn_3k.predict([X_nt_test_extra_neg_padded,X_nt_test_extra_neg_padded,X_nt_test_extra_neg_padded], verbose=False)


note_score_skewed = pd.DataFrame(
  {'fold':[-1 for i in y_pred_cnn.tolist()],
    'idx': extra_negative_samples_for_test_crossvalidation.idx,
    'score': [i[0] for i in y_pred_cnn.tolist()]})
cnn_score_train=cnn_score_train.append(note_score_skewed)
  
threshould_max=0.91 # threshould_max is the threshold value that gets the maximum F1-score value

####################################### join the calculated note_scores with other structured data ################
X_dev_plus_notescore=sqldf("""SELECT *  FROM X_test_st_skewed a inner join cnn_score_train b on a.idx=b.idx """)
##################### the st train model ####################
  
X_st_test_extra_neg = extra_negative_samples_for_test_crossvalidation.drop(['idx','note','PTSD'],axis=1)
#X_nt_test_extra_neg=df_st_nt_just_negative_PTSD.note
X_nt_test_extra_neg = extra_negative_samples_for_test_crossvalidation.note
y_nt_test_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD
y_st_extra_neg=extra_negative_samples_for_test_crossvalidation.PTSD
y_st_extra_neg_df=y_st_extra_neg.to_frame(name='PTSD')

skewed_part=X_dev_plus_notescore[X_dev_plus_notescore.fold == -1]
train_part=X_dev_plus_notescore[X_dev_plus_notescore.fold != -1]

skewed_part=X_dev_plus_notescore[X_dev_plus_notescore.fold == -1]
skewed_part=skewed_part.drop(['idx','fold'],axis=1)


##################### structured model ##########################



from sklearn.ensemble import RandomForestClassifier


fold=0
cols=7 # number of mertrics
rows=85

sum_vector= [[0]*cols]*rows

cls = RandomForestClassifier()
fold=1
cv = StratifiedKFold(n_splits=10)

train_part=train_part.drop(['idx','fold'],axis=1)
#X_st_test_skewed=X_st_test_skewed.drop(['Unnamed: 0', 'recNo', 'rand'],axis=1)
skewed_part=skewed_part.drop(['idx','fold'],axis=1)
for train,test in cv.split(train_part,y_RUS_train):
  fold+=1
  X_st_train=train_part.iloc[train]
  X_st_test=train_part.iloc[test]
  #
  X_st_test_skewed=pd.concat([X_st_test, skewed_part])

  y_train=y_RUS_train.iloc[train]
  y_test=y_RUS_train.iloc[test]
  y_test_holdout_dev=pd.concat([y_test, y_st_extra_neg_df])
  



  RF_model = cls.fit(X_st_train,y_train)
  y_pred = RF_model.predict_proba(X_st_test_skewed)[::,1]
   
  threshould=0.1
  while(threshould<0.95):
		threshold+=0.01
        y_pred_rounded = (y_pred > threshould)
        cm=confusion_matrix(y_test, y_pred_rounded)  
        metrics=show_metrics(y_test, y_pred_rounded)
        sum_vector[i]=sum_vector[i]+np.array(metrics)
        print(metrics)
#----------------------- calculate the average metrics for each threshould -----------
print("\n-------------------- average for each threshold -----------------")
print("\tPPV \t NPV \t SN \t SP \t ACC \t F1 \t AUC")
for loop in range(len(sum_vector)):
  v=sum_vector[loop]
  print(threshoulds[loop],end=':\t')
  for n in v:
    print(round(n/fold,2),end='\t')
  print()

		  

threshould_max=getMaxThreshold(sum_vector)# threshould_max is the threshold value that gets the maximum F1-score value

#-------TEST with the holdout data--------------------

X_nt_holdout=X_test_holdout.note
X_st_holdout=X_test_holdout.drop(['idx','note'],axis=1)

X_test_encoded = tokenizer.texts_to_sequences(X_nt_holdout)
X_test_padded = pad_sequences(X_test_encoded, maxlen=max_len, padding='post')

#y_pred_cnn = model_CNN_nt.predict(X_test_padded, verbose=False)
y_pred_cnn_nt_holdout = model_cnn_3k.predict([X_test_padded,X_test_padded,X_test_padded], verbose=False)
pid=X_test_holdout['idx'].to_list()
note_score_holdout = pd.DataFrame(
    {'idx': pid,
     'score': [i[0] for i in y_pred_cnn_nt_holdout.tolist()]})
	 
######### join the hold-ut note scores with other st data ###############
X_test_holdout_plus_notescore=sqldf("""SELECT *  FROM X_test_holdout a inner join note_score_holdout b on a.idx=b.idx """)

############### apply the trained RF model on the holdout data #################
y_pred = RF_model.predict_proba(X_test_holdout_plus_notescore)[::,1]
y_pred_rounded = (y_pred > threshould_max)
cm=confusion_matrix(y_test_holdout, y_pred_rounded)  
metrics=show_metrics(y_test_holdout, y_pred_rounded)
print(cm)
print(metrics)