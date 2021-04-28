import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, classification_report
import seaborn as sns
sns.set()

reviews_df = pd.read_csv('./tripadvisor_hotel_reviews.csv')

reviews_df.head()


def mapping_sentiments(rating):
    if rating <= 2:
        return -1
    elif rating == 3:
        return 0
    else:
        return 1

reviews_df['Rating'] = [mapping_sentiments(x) for x in reviews_df['Rating']]


sns.countplot(x = "Rating", data = reviews_df)



X = reviews_df['Review']
y = reviews_df['Rating']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 17)


vectorizing = TfidfVectorizer(stop_words='english', ngram_range = (1, 5), analyzer = 'word')
X_train_transformed = vectorizing.fit_transform(X_train)


X_test_transformed = vectorizing.transform(X_test)


classifier = LinearSVC(C = 20, class_weight = 'balanced', random_state = 17, tol = 1e-07)


accuracy_score = cross_val_score(classifier, X_train_transformed, y_train, cv = 5, scoring = 'accuracy')
print(np.mean(accuracy_score))

classifier.fit(X_train_transformed, y_train)

y_pred = classifier.predict(X_test_transformed)


print(classification_report(y_test, y_pred))

test_accuracy = cross_val_score(classifier, X_test_transformed, y_test, cv = 5, scoring = 'accuracy')
print(np.mean(test_accuracy))




