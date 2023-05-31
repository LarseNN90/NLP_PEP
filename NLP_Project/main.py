#Credit to:
#https://www.kaggle.com/code/blurredmachine/bag-of-words-meets-random-forest/notebook
#1
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import nltk
#nltk.download() #ONLY RUN THIS LINE ONCE; Or you can keep running it if you believe in your computer
from nltk.corpus import stopwords # Import the stop word list

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

###2
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

###3
df_train = pd.read_csv(r"C:\Users\Thoma\Downloads\archive\labeledTrainData.tsv",
                              header=0,
                              delimiter="\t",
                              quoting=3)

df_test = pd.read_csv(r"C:\Users\Thoma\Downloads\archive\testData.tsv",
                             header=0,
                             delimiter="\t",
                             quoting=3)

####4
print(df_train.shape)
print(df_test.shape)


####5
df_train.info()

###6
###Handling the HTML markers in the reviews.
bs_data = BeautifulSoup(df_train["review"][0])
###7
###Removing numbers and such
letters_only = re.sub("[^a-zA-Z]", " ", bs_data.get_text() )

###8
###Splitting the text down into word tokens
lower_case = letters_only.lower()
words = lower_case.split()

###9
###Assembling the words into a bag of words
words = [w for w in words if not w in stopwords.words("english")]

###10
###Setting size for future processing
training_data_size = df_train["review"].size
testing_data_size = df_test["review"].size

###11
##Setting up function to clean data
def clean_text_data(data_point, data_size):
    review_soup = BeautifulSoup(data_point,features="html.parser")
    review_text = review_soup.get_text()
    review_letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    review_lower_case = review_letters_only.lower()
    review_words = review_lower_case.split()
    stop_words = stopwords.words("english")
    meaningful_words = [x for x in review_words if x not in stop_words]

    if ((i) % 2000 == 0):
        print("Cleaned %d of %d data (%d %%)." % (i, data_size, ((i) / data_size) * 100))

    return (" ".join(meaningful_words))
###12
###Show header for training dataset
df_train.head()

###13
###Cleaning dataframe
for i in range(training_data_size):
    df_train["review"][i] = clean_text_data(df_train["review"][i], training_data_size)
print("Cleaning training completed!")

###14
###Cleaning dataframe
for i in range(testing_data_size):
    df_test["review"][i] = clean_text_data(df_test["review"][i], testing_data_size)
print("Cleaning validation completed!")

###15
#Getting features ready to be trained; extracting features
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)
###16
###Extracting a validation testset from the training testset, equal to 30% of the data.
X_train, X_cv, Y_train, Y_cv = train_test_split(df_train["review"], df_train["sentiment"], test_size = 0.3, random_state=42)

###17

###Converting train set to vectors
X_train = vectorizer.fit_transform(X_train)
X_train = X_train.toarray()
print(X_train.shape)

###18
###Converting validation set to vectors
X_cv = vectorizer.transform(X_cv)
X_cv = X_cv.toarray()
print(X_cv.shape)

###19
###Converting test set to vectors
X_test = vectorizer.transform(df_test["review"])
X_test = X_test.toarray()
print(X_test.shape)
###20

distribution = np.sum(X_train, axis=0)

##21
#Training random forest model
forest = RandomForestClassifier()
forest = forest.fit( X_train, Y_train)

###22
###Validating the model and using the model on the dataset
predictions = forest.predict(X_cv)
print("Accuracy: ", accuracy_score(Y_cv, predictions))

result = forest.predict(X_test)
output = pd.DataFrame( data={"id":df_test["id"], "sentiment":result} )
output.to_csv( "submission.csv", index=False, quoting=3 )
