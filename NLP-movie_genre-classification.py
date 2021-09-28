
"""# Preprocessing data"""

#import library 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import PorterStemmer #nltk.download('stopwords'),nltk.download('porter_test')
from nltk.tokenize import sent_tokenize, word_tokenize #nltk.download('punkt')

df = pd.read_csv('kaggle_movie_train.csv')

#one hot encoding 
genre = pd.get_dummies(df.genre)

#merge dataframe df with one hot encoding result
new_df = pd.concat([df, genre], axis=1)

#drop genre column 
new_df.drop(['genre'], axis=1)

#Apply text Preprocessing steps
porter = PorterStemmer()

def stemSentence(sentence):
    token_words=word_tokenize(sentence)
    token_words
    stem_sentence=[]
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)

stopwords = nltk.corpus.stopwords.words('english')

for i in range(df.shape[0]):

    #converting to lower case
    new_df.text[i] = new_df.text[i].lower()
    
    #cleaning special character
    new_df.text[i] = re.sub(r"[^a-zA-Z0-9]"," ",new_df.text[i])


    #removing stopwords
    text_tokens = word_tokenize(new_df.text[i])
    tokens_without_sw = [word for word in text_tokens if not word in stopwords]
    new_df.text[i] = (" ").join(tokens_without_sw)

    #stemming 
    new_df.text[i] = stemSentence(new_df.text[i])

#convert dataframe column into feature and label 
feature = new_df['text'].values
label = new_df[['action','adventure','comedy','drama','horror','other','romance','sci-fi','thriller']]

# data split for train dan validation
feature_train, feature_test, label_train, label_test = train_test_split(feature, label, test_size=0.2)

#convert string to numeric values
tokenizer = Tokenizer(num_words=5000, oov_token='x')
tokenizer.fit_on_texts(feature_train) 
tokenizer.fit_on_texts(feature_test) 

 
#convert into sequences
sequences_train = tokenizer.texts_to_sequences(feature_train)
sequences_test = tokenizer.texts_to_sequences(feature_test)

#padding
padded_train = pad_sequences(sequences_train) 
padded_test = pad_sequences(sequences_test)


#modelling
#use callback to prevent overfitting
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy') > 0.8):
      print("\ Stop training!")
      self.model.stop_training = True

callbacks = myCallback()

#build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=16),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(9, activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(
    padded_train,
    label_train, 
    epochs=40, 
    validation_data=(padded_test,label_test), 
    verbose=2,
    callbacks=[callbacks]
    )



