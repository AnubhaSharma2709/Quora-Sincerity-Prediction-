#!/usr/bin/env python
# coding: utf-8

# <table class='center'>
#     <tr style="background-color:white">
#       <td>
#           <center><h1> To Predict whether Question Asked on Quora are Sincere or Insincere</h1></center>
#           <hr>
#           <center><h2>Interestship 5.0 by Clique Community</h2></center>
#           <hr>
#           <center><h3>Mentee Name: Anubha Sharma</h3></center>
#       </td>
#      </tr>
# </table>
# 
# #### Understanding the Project First
# <ul>
#     <li>An Insincere question is defined as Questios Intended to make a statement rather than look for helpful answers</li>
# </ul>
# 
# #### Some Characteristics that can Signify that a Question is Insincere:
# <ol>
#     <li>Has a Non Neutral Tone:
#         <ul>
#             <li>Has an exaggerated tone to underscore a point about a group of people</li>
#             <li>Is rhetorical and meant to imply a statement about a group of people</li>
#         </ul>
#         <br>
#     <li>Is disparaging or inflammatory:
#         <ul>
#             <li>Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype</li>
#             <li>Makes disparaging attacks/insults against a specific person or group of people</li>
#             <li>Based on an outlandish premise about a group of people</li>
#             <li>Disparages against a characteristic that is not fixable and not measurable</li>  
#         </ul>
#         <br>
#     <li>Isn't grounded in reality:
#         <ul>
#             <li>Based on false information, or contains absurd assumptions</li>
#         </ul>
#         <br>
# </ol>
# 
# #### About the Data Set
# <ul>
#     <li>The training data includes the question that was asked, and whether it was identified as insincere (target = 1).</li>
#     <li>The ground-truth labels contain some amount of noise: they are not guaranteed to be perfect.</li>
# </ul>
#     

# In[1]:


import tensorflow as tf
print("GPU Availabilty" if tf.config.list_physical_devices("GPU") else "not available")


# In[2]:


#Importing the basic Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


# Reading the CSV File

data=pd.read_csv('train.csv')
data


# #### Basics Insights about the Dataset

# In[5]:


data.describe()


# In[6]:


data.info()


# In[7]:


print(data.shape)


# #### Dataset has more than 13 Lakh rows and 3 colums
# 
# #### The columns are :
# >qid - unique question identifier
# 
# >question_text - Quora question text
# 
# >target - a question labeled "insincere" has a value of 1, otherwise 0
# 
# ### What is NLP (Natural Language Processing)?
# <ul>
#     <li>NLP is a subfield of computer science and artificial intelligence concerned with interactions between computers and human (natural) languages. It is used to apply machine learning algorithms to text and speech.</li></ul>
# <ol>Examples of NLP:
#     <ol>
#         <li>Speech Recognition</li>
#         <li>Document Summarization</li>
#         <li>Machine Translation</li>
#     </ol>
# </ol>
# 
# ## Text Preprocessing
# 
# #### In NLP, text preprocessing is the first step in the process of building a model.
# 
# ### Step 1: Punctutation Removal
# 

# In[8]:


import string
string.punctuation

def removing_punctuation(text):
    ptfree="".join([i for i in text if i not in string.punctuation])
    return ptfree
#storing the puntuation free text
data['cleaned_ques']= data['question_text'].apply(lambda x:removing_punctuation(x))
data.head()


# ### Step 2: Lowering Text
# 
# >Converting a word to lower case (NLP -> nlp).
# Words like Book and book mean the same but when not converted to the lower case those two are represented as two different words in the vector space model (resulting in more dimensions).
# 

# In[9]:


data['ques_lower']= data['cleaned_ques'].apply(lambda x: x.lower())


# ### Step 3: Tokenization
# 
# >NLTK contains a module called tokenize() which further classifies into two sub-categories: Sentence Tokenization and Word Tokenization.<ul><li>NLTK — The Natural Language ToolKit is one of the best-known and most-used NLP libraries, useful for all sorts of tasks from tokenization, stemming, tagging, parsing, and beyond</li>
# </ul>
# 
# 
# >Tokenization is the process of tokenizing or splitting a string, text into a list of tokens.</li>
# 
# ><ul>Text into sentences tokenization
# > <li>Sentences into words tokenization</li>
# > <li>Sentence Tokenization - We use the sent_tokenize() method to split a document or paragraph into sentences</li>
#  </ul>
# 
# <ul><li>
# Punkt tokenizer divides a text into a list of sentences by using an unsupervised algorithm to build a model for abbreviation words, collocations, and words that start sentences.
#     </li></ul>
# 
# <ul><li>
# NLTK — The Natural Language ToolKit is one of the best-known and most-used NLP libraries, useful for all sorts of tasks from tokenization, stemming, tagging, parsing, and beyond</li>
# </ul>
# 

# In[10]:


#Installing all the required packages
get_ipython().system('pip install nltk')
get_ipython().system('pip install spacy')


# In[11]:


import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def tokenization(text):
    tokens = word_tokenize(text)
    return tokens
#applying function to the column
data['ques_tokenized']= data['ques_lower'].apply(lambda x: tokenization(x))


# ### Step 4: Stop Words Removal
# 
# >Stop words are very commonly used words (a, an, the, etc.) in the documents. These words do not really signify any importance as they do not help in distinguishing two documents.
# 

# In[12]:


import nltk
import spacy
from nltk.corpus import stopwords
nltk.download('stopwords')
import nltk
#Stop words present in the library
stopwords = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

data['no_stopwords']= data['ques_tokenized'].apply(lambda x:remove_stopwords(x))
data['no_stopwords'][0]


# ### Step 5: Lemmatization
# 
# <ul><li>Lemmatization reduces the words to a word existing in the language.</li></ul>
# 

# In[13]:


import nltk
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()


def lemmatizer(text):
  lemm_text = [wordnet_lemmatizer.lemmatize(word) for word in text]
  lemm_text=" ".join(lemm_text)
  return lemm_text

data['ques_lemmatized']=data['no_stopwords'].apply(lambda x:lemmatizer(x))
data['ques_lemmatized'][0]


# In[14]:


preprocessed_data= data[['qid','question_text','ques_lemmatized','target']]
preprocessed_data.to_csv('quora_preprocessed.csv',index=False)


# <center>
# <hr>
#     <h2>Exploratory Data Analysis</h2>
#     </center>
# 

# In[15]:


get_ipython().system('pip install wordcloud')


# In[16]:


preprocessed_data.head()


# In[17]:


preprocessed_data.describe()


# In[18]:


preprocessed_data.info()


# In[19]:


preprocessed_data.isna().sum()


# In[20]:


preprocessed_data["target"].value_counts()


# In[21]:


#Text Word startistics: min.mean, max and interquartile range
# This is mainly done to find out more about the Text Length

txt_length = preprocessed_data.ques_lemmatized.str.split().str.len()
txt_length.describe()


# In[22]:


# Defining Sincere and Insincere Words
sincere_words = preprocessed_data[preprocessed_data['target']==0]['ques_lemmatized']
insincere_words = preprocessed_data[preprocessed_data['target']==1]['ques_lemmatized']


# <center><h1 style="color:Blue;">Graph-1: See the distribution of Target Value in DataSet</h1></center>
# 

# In[23]:


preprocessed_data["target"].value_counts().plot(kind="pie",autopct='%0.2f',radius=1.5)


# <center><h3 style="color:red;">From the graph we can conclude that the data is highly imbalanced with 93.81 % of Sincere Data and Only 6.19% of Insincere Data</h3></center><hr>
# <ul><h3><li>Target1: Insincere Data</li></h3>
#         <h3><li>Target0: Sincere Data</li></h3>
#     </ul>
#     
# <center><h1 style="color:Blue;">Graph-2: Word Cloud</h1></center>
# <hr>
# <h1> What is a Word Cloud?</h1>
# <ul>
#     <li><h4>A word cloud is a collection, or cluster, of words depicted in different sizes.</h4></li>
#     <li><h4>The bigger and bolder the word appears, the more often it’s mentioned within a given text and the more important it is.</h4></li>
#     </ul>
# 

# In[24]:


from wordcloud import WordCloud, STOPWORDS
# initialize the word cloud

wordcloud = WordCloud( background_color='black', width=400, height=200)
# generate the word cloud by passing the corpus
text_cloud = wordcloud.generate(' '.join(preprocessed_data['ques_lemmatized']))

# plotting the word cloud
plt.figure(figsize=(10,20))
plt.imshow(text_cloud)
plt.axis('off')
plt.show()


# <center><h2>Word Cloud-1: All the words present in the DataSet</h2></center>
# <h3> Conclusion from Word Cloud-1</h3>
# <ol>Some of the maximum used words in the dataset are:
#     <li>india</li>
#     <li>one</li>
#     <li>make</li>
#     <li>best</li>
# </ol>

# In[25]:


sincere_word = ' '.join(sincere_words)
wc = wordcloud.generate(sincere_word)
plt.figure(figsize=(10,20))
plt.imshow(wc)
plt.axis('off')
plt.show()


# <center><h2>Word Cloud-2: All the Sincere words present in the DataSet</h2></center>
# <h3> Conclusion from Word Cloud-2</h3>
# <ol>Some of the maximum used words in the dataset are:
#     <li>one</li>
#     <li>india</li>
#     <li>good</li>
#     <li>best</li>
# </ol>

# In[26]:


insincere_word = ' '.join(insincere_words)
wc = wordcloud.generate(insincere_word)
plt.figure(figsize=(10,20))
plt.imshow(wc)
plt.axis('off')
plt.show()


# <center><h2>Word Cloud-3: All the Insincere words present in the DataSet</h2></center>
# <h3> Conclusion from Word Cloud-3</h3>
# <ol>Some of the maximum used words in the dataset are:
#     <li>women</li>
#     <li>people</li>
#     <li>muslim</li>
#     <li>trump</li>
# </ol>
# <center><h1 style="color:Blue;">Graph-3: Frequency of Top 20 Words in the DataSet</h1></center>
# <h2> What is Count Vectorizer</h2>
# <ul><li>CountVectorizer is a great tool provided by the scikit-learn library in Python.</li>
# <li>It is used to transform a given text into a vector on the basis of the frequency (count) of each word that occurs in the entire text.</li>
#     <li>This functionality makes it a highly flexible feature representation module for text.</li>
#     <hr>
#     <li><h3>Bag Of Words:</h3></li><br>
#     Bag of Words model is used to preprocess the text by converting it into a bag of words, which keeps a count of the total occurrences of most frequently used words.
#     </ul>

# In[27]:


from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1],
                        reverse=True)
    return words_freq[:n]


# In[28]:


fig =plt.figure(figsize = (15,5))
plt.subplot(1, 2, 1)
plot_2=get_top_n_words(sincere_words,10)
all_sincere_words=pd.DataFrame(plot_2,columns=['Words','TotalCount'])
all_sincere_words.groupby('Words').sum()['TotalCount'].plot.barh(color='green')
plt.title("Top 10 Sincere Words")

plt.subplot(1, 2, 2)
plot_3=get_top_n_words(insincere_words,10)
all_insincere_words=pd.DataFrame(plot_3,columns=['Words','TotalCount'])
all_insincere_words.groupby('Words').sum()['TotalCount'].plot.barh(color='red')
plt.title("Top 10 Insincere Words")
plt.show()

fig =plt.figure(figsize = (12,10))
plt.subplot(2, 2, 1)
plot_2=get_top_n_words(sincere_words,10)
all_sincere_words=pd.DataFrame(plot_2,columns=['Words','TotalCount'])
all_sincere_words.groupby('Words').sum()['TotalCount'].plot(kind="pie",autopct='%0.2f')
plt.title("Top 10 Sincere Words")

plt.subplot(2, 2, 2)
plot_3=get_top_n_words(insincere_words,10)
all_insincere_words=pd.DataFrame(plot_3,columns=['Words','TotalCount'])
all_insincere_words.groupby('Words').sum()['TotalCount'].plot(kind="pie",autopct='%0.2f',startangle = 90)
plt.title("Top 10 Insincere Words")
plt.show()


# <center><h1 style="color:Blue;">Graph-4: More Insights on Questions in the DataSet </h1></center>
# <h3>Question Length</h3>

# In[29]:


def question_size(question):
    return len(question.split(" "))

data['question_size'] = data["question_text"].apply(question_size)


# In[30]:


plt.hist(data.question_size, bins=100, range=[0, 100],label='train');


# <ul><li><h3>Inference</h3>
#     <p>Mostly questions have length in range (10-20)</p>
#     </li>
# <hr>
# <h3>Number of Unique words in the text</h3>

# In[31]:


data["num_unique_words"] = data["question_text"].apply(lambda x: len(set(str(x).split())))
plt.figure(figsize=(12,8))
sns.violinplot(data=data['num_unique_words'])
plt.show()


# <h3>Average length of the words in the text</h3>

# In[32]:


data["mean_word_len"] = data["question_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
plt.figure(figsize=(12,8))
sns.violinplot(data=data['mean_word_len'])
plt.show()


# 
# <hr><center>
#     <h1>Model Training</h1>
#     </center>
#     <hr>

# In[33]:


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (precision_score, recall_score, f1_score,accuracy_score,confusion_matrix)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


# In[34]:


data = pd.read_csv("quora_preprocessed.csv")


# In[35]:


data=data.dropna()


# In[36]:


x = data['ques_lemmatized']
y = data['target']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)


# <h3>TfidfVectorizer</h3><hr>
# <p>We are using the TfidfVectorizer to convert textual data into a numerical representation suitable for machine learning algorithms.</p>
# <p>TfidfVectorizer stands for "Term Frequency-Inverse Document Frequency Vectorizer" which is a popular technique used in natural language processing (NLP).</p>

# In[37]:


vectoriser = TfidfVectorizer(ngram_range=(1,2),max_features=100000)
vectoriser.fit(x_train)

x_train = vectoriser.transform(x_train)
x_test  = vectoriser.transform(x_test)


# <h3>Logistic Regression</h3>
# <p>Binary classification algorithm. Calculates probabilities using sigmoid function.</p>
# <hr>
# <h3>Accuracy</h3>
# <p>Measure of correct predictions. Ratio of correct predictions to total predictions.</p>

# In[38]:


model = LogisticRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")


# <h3>F1 Score</h3>
# <p>Harmonic mean of precision and recall. Balance between precision and recall for binary classification evaluation</p>

# In[39]:


f1score=f1_score(y_test,y_pred)
print(f"F1 Score:{f1score:.4f}")


# <h1>Confusion Matrix</h1>
# <p>Confusion matrix is a table that visualizes the performance of a classification model.</p>
# <table border="1">
# <tr>
#     <td></td>
#             <td>Predicted Positive</td>
#             <td>Predicted Negative</td>
#         </tr>
#         <tr>
#             <td>Actual Positive</td>
#             <td>True Positive (TP)</td>
#             <td>False Negative (FN)</td>
#         </tr>
#         <tr>
#             <td>Actual Negative</td>
#             <td>False Positive (FP)</td>
#             <td>True Negative (TN)</td>
#         </tr>
#     </table>
#  <p>It helps in assessing the performance of a classifier by showing the count of true positive, false positive, true negative, and false negative predictions.</p>
# 

# In[40]:


confusion_lg = confusion_matrix(y_test, y_pred) #confusion metrics
sns.heatmap(confusion_lg, linewidths=0.01, annot=True,fmt= '.1f', color='red')


# <h3>Multinomial Naive Bayes</h3>
# <p>Multinomial Naive Bayes is a probabilistic classifier based on Bayes' theorem, suitable for discrete data (e.g., word counts in text classification)</p>

# In[41]:


nb = MultinomialNB()

# Train the model
nb.fit(x_train, y_train)
# Evaluate the model on the test set
y_p = nb.predict(x_test)
accuracy = accuracy_score(y_test, y_p)
f1score=f1_score(y_test,y_p)
print(f"Test accuracy: {accuracy:.4f}")
print(f"F1 Score:{f1score:.4f}")


# In[42]:


from sklearn.model_selection import StratifiedKFold


# In[43]:


x = data['ques_lemmatized'].values
y = data['target'].values


# In[44]:


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []
lst_f1score_stratified = []
for train_index, test_index in skf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    tfidf = TfidfVectorizer(ngram_range=(1,2),max_features=100000)
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)

    classifier = LogisticRegression(class_weight = "balanced", C=0.5, solver='sag')
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    lst_accu_stratified.append(accuracy)
    F1score = f1_score(y_test,y_pred)
    lst_f1score_stratified.append(F1score)


# <p>Stratified K-Fold Cross-Validation with 10 splits is applied to ensure that each fold has a balanced representation of the target classes.</p><p>The dataset is split into training and testing sets for each fold, and the TF-IDF vectorization is performed to convert text data to numerical features.</p><p> Then, a Logistic Regression classifier is trained on the training data and evaluated on the testing data for each fold.</p><p> Accuracy and F1 score are calculated for each fold and stored in separate lists for further analysis.</p>

# In[45]:


print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:',
      max(lst_accu_stratified)*100, '%')
print('\nMinimum Accuracy:',
      min(lst_accu_stratified)*100, '%')
print('\nOverall Accuracy:',
      np.mean(lst_accu_stratified)*100, '%')


# In[46]:


print('List of possible F1 Scores:', lst_f1score_stratified)
print('\nMaximum f1 score That can be obtained from this model is:',
      max(lst_f1score_stratified)*100, '%')
print('\nMinimum f1 score:',
      min(lst_f1score_stratified)*100, '%')
print('\nOverall f1score:',
      np.mean(lst_f1score_stratified)*100, '%')


# In[47]:


get_ipython().system('pip install transformers --quiet')


# In[48]:


get_ipython().system('pip install torch')


# In[49]:


data = pd.read_csv('quora_preprocessed.csv')
data.head()


# In[50]:


import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from collections import defaultdict
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup


# <h3>We are importing specific modules from the "transformers" library for natural language processing (NLP) tasks:</h3>
# <ol>
#     <li><h4>BertModel</h4>
#         <p> This module represents the pre-trained BERT (Bidirectional Encoder Representations from Transformers) model. BERT is a widely used transformer-based model for various NLP tasks.</p>
#     </li>
#     <li><h4>BertTokenizer</h4>
#         <p>This module provides the tokenizer for processing text data, which is used to convert text into numerical inputs suitable for the BERT model.</p>
#     </li>
#       <li><h4>AdamW</h4>
#         <p>This module is an extension of the Adam optimization algorithm. It is often used for fine-tuning pre-trained transformer models like BERT, with added weight decay to prevent overfitting.</p>
#     </li>
#       <li><h4>get_linear_schedule_with_warmup</h4>
#         <p>This module is used to create a learning rate scheduler that adjusts the learning rate during training. It linearly increases the learning rate during the warm-up phase and then linearly decreases it during the rest of the training process.</p>
#     </li>
#     </ol>

# In[51]:


pretrained_model_name = 'bert-base-cased'
maxlen = 160
batch_size = 16
epochs = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# <ol>
#     <li><h4>pretrained_model_name = 'bert-base-cased</h4>
#         <p>Specifies the BERT pre-trained model to use (cased version)</p>
#     </li>
#     <li><h4>maxlen = 160</h4>
#         <p>Sets the maximum input sequence length for training/inference</p>
#     </li>
#       <li><h4>batch_size = 16</h4>
#         <p> Defines the number of samples processed in one training iteration (batch).</p>
#     </li>
#       <li><h4>epochs = 1</h4>
#         <p>Number of times the whole dataset is passed through during training</p>
#     </li>
#     <li><h4>device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")</h4>
#         <p>Sets the device (GPU if available, else CPU) for computations.</p>
#     </li>
#     </ol>

# In[52]:


data['processed_text'] = data['ques_lemmatized'].astype('str')


# In[53]:


tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)


# In[54]:


token_lens = []
for txt in data.processed_text:
    tokens = tokenizer.encode(txt, max_length=512, truncation = True)
    token_lens.append(len(tokens))


# <ul><li>Calculates the token lengths of processed text using a tokenizer, which encodes the text into tokens with a maximum length of 512.</li><li><hr> It iterates through the dataset, appends the token lengths to "token_lens" list.</li></ul>

# In[55]:


class QuestionDataset(Dataset):

    def __init__(self, questions, targets, tokenizer, max_len):
        self.questions = questions
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.questions)
    def __getitem__(self, item):
        question = str(self.questions[item])
        target = self.targets[item]
        encoding = self.tokenizer.encode_plus(
          question,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=False,
          pad_to_max_length=True,
          return_attention_mask=True,
          return_tensors='pt',
          truncation=True
        )
        return {
          'question_text': question,
          'input_ids': encoding['input_ids'].flatten(),
          'attention_mask': encoding['attention_mask'].flatten(),
          'targets': torch.tensor(target, dtype=torch.long)
      }


# <ul><li>Defining a custom dataset class, QuestionDataset, to preprocess and tokenize a list of questions and their corresponding targets (labels) for NLP tasks.</li><li><hr>The class inherits from PyTorch's Dataset class, which allows it to be used with PyTorch's DataLoader for efficient data loading during training and inference.</li></ul>

# In[56]:


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(data,test_size = 0.4,random_state = 42)
val_data, test_data = train_test_split(test_data,test_size = 0.6,random_state = 42)


# In[57]:


def data_loader(data, tokenizer, max_len, batch_size):
    ds = QuestionDataset(
      questions = data.processed_text.to_numpy(),
      targets = data.target.to_numpy(),
      tokenizer = tokenizer,
      max_len = max_len
    )
    return DataLoader(
      ds,
      batch_size = batch_size,
      num_workers = 0
    )


# <ul><li>Function that creates a DataLoader for NLP tasks.</li><hr><li>It uses the QuestionDataset class to preprocess data and tokenize using the provided tokenizer.</li><hr><li>The DataLoader loads data in batches with the specified batch size.</li>

# In[58]:


train_data_loader = data_loader(train_data, tokenizer, maxlen, batch_size)
val_data_loader = data_loader(val_data, tokenizer, maxlen, batch_size)
test_data_loader = data_loader(test_data, tokenizer, maxlen, batch_size)

data = next(iter(train_data_loader))
data.keys()

print (len(train_data_loader))
print (len(val_data_loader))
print (len(test_data_loader))


# <ul><li>Creating data loaders for training, validation, and testing using the data_loader function.</li><hr><li> It fetches the keys (features) of a single batch from the training data.</li><hr><li> It then prints the number of batches in each data loader: train, validation, and test.</li>

# In[59]:


# Shape of the torch
print(data['input_ids'].shape)
print(data['attention_mask'].shape)
print(data['targets'].shape)


# <ol>
#     <li><h4>input_ids</h4>
#         <p>This tensor represents the tokenized and encoded input sequence of the questions. It contains numerical representations of the tokens, which are used as input to the NLP model.</p>
#     </li>
#     <li><h4>attention_mask</h4>
#         <p>This tensor indicates which elements of the input sequence are valid and which ones are padding. It has the same shape as input_ids and helps the model focus on relevant information.</p>
#     </li>
#       <li><h4>targets</h4>
#         <p> This tensor holds the labels or target values corresponding to the input questions. It is used during training to compute the model's loss and evaluate its performance.</p>
#     </li>
#     </ol>

# In[60]:


data


# In[61]:


class QuestionClassifier(nn.Module):
    def __init__(self, n_classes):
        super(QuestionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          return_dict=False,
        )
        output = self.drop(pooled_output)
        return self.out(output)


# <p>Defining a Question Classifier using BERT as a base model for NLP tasks.<p>

# In[62]:


model = QuestionClassifier(2)
model = model.to(device)


# In[63]:


input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

print(input_ids.shape) # batch size x seq length
print(attention_mask.shape) # batch size x seq length


# In[64]:


model


# In[65]:


model(input_ids, attention_mask)


# In[66]:


optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_data_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)


# In[67]:


def training_func(
    model,
    data_loader,
    loss_fn,
    optimizer,
    device,
    scheduler,
    n_examples
):
    # Putting the model in the training mode
    model = model.train()

    losses = []
    correct_predictions = 0
    for dl in data_loader:
        input_ids = dl["input_ids"].to(device)
        attention_mask = dl["attention_mask"].to(device)
        targets = dl["targets"].to(device)
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)
        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


# <ul><li>Sets the model to training mode.</li>
#     <li>Iterates through the data_loader to process batches of data.</li>
#     <li>Moves data and model to the specified device (e.g., GPU).</li>
#     <li>Calculates model predictions and loss using the specified loss function.</li>
#     <li>Performs backpropagation and gradient updates using the optimizer.</li>
#     <li>Clips gradients to prevent exploding gradients.</li>
#     <li>Updates the learning rate using the scheduler.</li>
# </ul>

# In[68]:


def evaluate_model(model, data_loader, loss_fn, device, n_examples):

    # Putting the model in the Evaluation mode
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for dl in data_loader:
            input_ids = dl["input_ids"].to(device)
            attention_mask = dl["attention_mask"].to(device)
            targets = dl["targets"].to(device)
            outputs = model(
              input_ids=input_ids,
              attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)


# <ul><li>Code defines an evaluation function for a PyTorch model.</li><hr><li> It calculates accuracy and loss on a given data_loader using a specified loss function and device, then returns the accuracy and mean loss.</li></ul>

# In[ ]:


history = defaultdict(list)
best_accuracy = 0
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 30)
    train_acc, train_loss = training_func(
      model,
      train_data_loader,
      loss_fn,
      optimizer,
      device,
      scheduler,
      len(train_data)
    )
    print(f'Train loss {train_loss} accuracy {train_acc}')
    val_acc, val_loss = evaluate_model(
      model,
      val_data_loader,
      loss_fn,
      device,
      len(val_data)
    )
    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()
    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)
    if val_acc > best_accuracy:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_accuracy = val_acc


# <ul><li>The code trains a neural network model for multiple epochs, monitoring and storing training and validation accuracy/loss.</li><hr><li>The best model is saved based on the highest validation accuracy achieved.</li><ul>

# In[ ]:


def predictions(model, data_loader):
    model = model.eval()
    question_texts = []
    predictions = []
    prediction_probs = []
    real_values = []
    with torch.no_grad():
        for dl in data_loader:
            texts = dl["question_text"]
            input_ids = dl["input_ids"].to(device)
            attention_mask = dl["attention_mask"].to(device)
            targets = dl["targets"].to(device)
            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)
            question_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return question_texts, predictions, prediction_probs, real_values


# In[ ]:


y_question_texts, y_pred, y_pred_probs, y_test = predictions(model,test_data_loader)


# In[ ]:


i = 0
for text, pred, prob in zip(y_question_texts, y_pred, y_pred_probs):
    print(text, end = "   ")
    print(pred, end = "   ")
    print(prob)
    i+=1
    if i == 10:
        break


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:


report = classification_report(y_test, y_pred, output_dict=True)
report= pd.DataFrame.from_dict(report)
report


# In[ ]:


report_df=report.drop(['weighted avg','macro avg'], axis=1)
report_df


# In[ ]:


new_report_df=report_df.drop(['accuracy'],axis=1)
new_report_df

