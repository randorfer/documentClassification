#%%
def load_data():
    import os
    import io
    import numpy as np

    NEWLINE = '\n'
    SKIP_FILES = {'cmds'}


    def read_files(path):
        for root, dir_names, file_names in os.walk(path):
            for path in dir_names:
                read_files(os.path.join(root, path))
            for file_name in file_names:
                if file_name not in SKIP_FILES:
                    file_path = os.path.join(root, file_name)
                    if os.path.isfile(file_path):
                        past_header, lines = False, []
                        with io.open(file_path, encoding="latin-1") as f:
                            for line in f:
                                if past_header:
                                    lines.append(line)
                                elif line == NEWLINE:
                                    past_header = True
                        content = NEWLINE.join(lines)
                        yield file_path, content


    from pandas import DataFrame

    def build_data_frame(path, classification):
        rows = []
        index = []
        for file_name, text in read_files(path):
            rows.append({'text': text, 'class': classification})
            index.append(file_name)

        data_frame = DataFrame(rows, index=index)
        return data_frame


    SOURCES = [
        ('data/course', 'course'),
        ('data/department', 'department'),
        ('data/faculty', 'faculty'),
        ('data/other', 'other'),
        ('data/project', 'project'),
        ('data/staff', 'staff'),
        ('data/student', 'student')
    ]

    data = DataFrame({'text': [], 'class': []})
    for path, classification in SOURCES:
        data = data.append(build_data_frame(path, classification))

    from sklearn.utils import shuffle
    data = data.reindex(shuffle(data.index, random_state=42))
    return data
data = load_data()
data.describe()

#%%
data.head(1)

#%%
print "Setup build and evaluate"
import os
import time
import string
import pickle
from operator import itemgetter
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split as tts
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import confusion_matrix

def build_and_evaluate(
        X, y, classifier=SGDClassifier,
        verbose=True, ngram_range=(1,1), test_size=0.2, max_features=None
    ):
    """
    Builds a classifer for the given list of documents and targets in two
    stages: the first does a train/test split and prints a classifier report,
    the second rebuilds the model on the entire corpus and returns it for
    operationalization.

    X: a list or iterable of raw strings, each representing a document.
    y: a list or iterable of labels, which will be label encoded.

    Can specify the classifier to build with: if a class is specified then
    this will build the model with the Scikit-Learn defaults, if an instance
    is given, then it will be used directly in the build pipeline.

    If outpath is given, this function will write the model as a pickle.
    If verbose, this function will print out information to the command line.
    """

    def build(classifier, X, y=None, ngram_range=(1,1), max_features=None):
        """
        Inner build function that builds a single model.
        """
        if isinstance(classifier, type):
            classifier = classifier()

        model = Pipeline([
            ('vectorizer', TfidfVectorizer(
                ngram_range=ngram_range,
                stop_words='english',
                max_features=max_features
            )),
            ('classifier', classifier),
        ])

        model.fit(X, y)
        return model

    # Label encode the targets
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # Begin evaluation
    if verbose: print("Building for evaluation")
    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size)
    
    model = build(classifier, 
        X_train, 
        y_train, 
        ngram_range=ngram_range, 
        max_features=max_features
    )

    model.labels_ = labels

    if verbose: print("Classification Report:\n")

    y_pred = model.predict(X_test)
    print(clsr(y_test, y_pred, target_names=labels.classes_))
    print(confusion_matrix(y_test, y_pred))

    return model
print "Completed setup build and evaluate"
#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.79      0.75      0.77       269
 department       0.76      0.44      0.56        72
    faculty       0.78      0.71      0.74       334
      other       0.85      0.83      0.84      1126
    project       0.54      0.63      0.58       147
      staff       0.30      0.17      0.22        42
    student       0.72      0.86      0.78       495

avg / total       0.78      0.78      0.77      2485

[[202   0   8  39   5   1  14]
 [  2  32   8  19   2   0   9]
 [  6   2 236  33   8   4  45]
 [ 37   6  26 932  48   6  71]
 [  1   1  10  24  93   3  15]
 [  0   0   5   9   5   7  16]
 [  7   1   9  39  10   2 427]]
'''
model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = SGDClassifier(
        alpha=0.000001,
        penalty='elasticnet'
    ),
    ngram_range=(1,2),
    test_size=0.3,
    max_features = 5000
)

#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.72      0.79      0.75       370
 department       0.55      0.26      0.36        68
    faculty       0.78      0.71      0.74       469
      other       0.86      0.81      0.83      1494
    project       0.61      0.47      0.53       199
      staff       0.41      0.13      0.20        52
    student       0.69      0.89      0.77       661

avg / total       0.77      0.77      0.76      3313

[[ 294    0   10   56    2    0    8]
 [   7   18   10   14    5    2   12]
 [  19    1  333   22   16    2   76]
 [  74   10   43 1216   29    3  119]
 [   8    4   16   48   93    2   28]
 [   2    0    8    6    3    7   26]
 [   5    0    9   57    4    1  585]]
'''
model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = SGDClassifier(n_jobs=4),
    ngram_range=(1,3),
    test_size=0.4
)

#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.79      0.79      0.79       362
 department       0.41      0.19      0.26        75
    faculty       0.71      0.74      0.73       449
      other       0.85      0.82      0.84      1494
    project       0.67      0.44      0.53       217
      staff       0.70      0.12      0.21        57
    student       0.67      0.85      0.75       659

avg / total       0.76      0.76      0.75      3313

[[ 286    1    9   50    1    0   15]
 [   7   14   11   27    4    1   11]
 [  12    3  333   22   10    0   69]
 [  48   12   46 1232   24    1  131]
 [   4    3   34   51   95    0   30]
 [   0    1   12    9    3    7   25]
 [   5    0   22   63    5    1  563]]
'''
model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = SGDClassifier(n_jobs=4, loss='hinge'),
    ngram_range=(1,3),
    test_size=0.4
)

#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.73      0.79      0.76       369
 department       0.48      0.14      0.22        76
    faculty       0.71      0.74      0.72       452
      other       0.85      0.83      0.84      1500
    project       0.65      0.39      0.49       205
      staff       0.42      0.11      0.17        47
    student       0.70      0.87      0.77       664

avg / total       0.76      0.77      0.75      3313

[[ 291    0   10   57    0    0   11]
 [  10   11   19   21    5    1    9]
 [  14    1  334   27    7    0   69]
 [  62    9   53 1238   26    4  108]
 [  11    1   30   50   80    2   31]
 [   0    0   10    5    3    5   24]
 [   8    1   14   61    3    0  577]]
'''
model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = SGDClassifier(n_jobs=4, loss='hinge', penalty='elasticnet'),
    ngram_range=(1,3),
    test_size=0.4
)


#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.97      0.51      0.67       356
 department       0.90      0.28      0.43        64
    faculty       0.83      0.65      0.73       452
      other       0.69      0.96      0.80      1505
    project       1.00      0.10      0.18       197
      staff       0.33      0.02      0.04        50
    student       0.69      0.65      0.67       689

avg / total       0.76      0.73      0.70      3313

[[ 181    0    4  155    0    0   16]
 [   2   18    0   31    0    0   13]
 [   1    1  293   82    0    0   75]
 [   1    1   16 1443    0    0   44]
 [   2    0   14  129   19    0   33]
 [   0    0   11   21    0    1   17]
 [   0    0   15  223    0    2  449]]
'''

from sklearn.ensemble import RandomForestClassifier
Model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = RandomForestClassifier(n_estimators=50),
    ngram_range=(1,3),
    test_size=0.4
)

#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.00      0.00      0.00       357
 department       0.00      0.00      0.00        72
    faculty       0.00      0.00      0.00       457
      other       0.46      1.00      0.63      1526
    project       0.00      0.00      0.00       201
      staff       0.00      0.00      0.00        52
    student       0.00      0.00      0.00       648

avg / total       0.21      0.46      0.29      3313

[[   0    0    0  357    0    0    0]
 [   0    0    0   72    0    0    0]
 [   0    0    0  457    0    0    0]
 [   0    0    0 1526    0    0    0]
 [   0    0    0  201    0    0    0]
 [   0    0    0   52    0    0    0]
 [   0    0    0  648    0    0    0]]
'''
from sklearn.svm import SVC
Model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = SVC(kernel='rbf'),
    ngram_range=(1,3),
    test_size=0.4
)

#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.79      0.75      0.77       396
 department       0.63      0.25      0.36        68
    faculty       0.74      0.69      0.71       458
      other       0.82      0.86      0.84      1491
    project       0.69      0.42      0.52       206
      staff       0.58      0.11      0.19        63
    student       0.68      0.86      0.76       631

avg / total       0.77      0.77      0.76      3313

[[ 296    0    6   78    2    0   14]
 [   5   17   18   19    0    0    9]
 [  13    1  314   34    9    0   87]
 [  45    5   49 1281   23    2   86]
 [   8    3   22   57   87    2   27]
 [   0    1   10   15    2    7   28]
 [   6    0    4   72    3    1  545]]
'''
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
Model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = OneVsRestClassifier(LinearSVC()),
    ngram_range=(1,3),
    test_size=0.4
)

#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.79      0.75      0.77       396
 department       0.63      0.25      0.36        68
    faculty       0.74      0.69      0.71       458
      other       0.82      0.86      0.84      1491
    project       0.69      0.42      0.52       206
      staff       0.58      0.11      0.19        63
    student       0.68      0.86      0.76       631

avg / total       0.77      0.77      0.76      3313

[[ 296    0    6   78    2    0   14]
 [   5   17   18   19    0    0    9]
 [  13    1  314   34    9    0   87]
 [  45    5   49 1281   23    2   86]
 [   8    3   22   57   87    2   27]
 [   0    1   10   15    2    7   28]
 [   6    0    4   72    3    1  545]]
'''
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
Model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = OneVsOneClassifier(LinearSVC()),
    ngram_range=(1,3),
    test_size=0.4
)

#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.80      0.66      0.73       384
 department       0.55      0.15      0.24        73
    faculty       0.71      0.63      0.67       430
      other       0.80      0.86      0.82      1511
    project       0.69      0.42      0.52       205
      staff       0.44      0.07      0.12        55
    student       0.69      0.88      0.78       655

avg / total       0.75      0.75      0.74      3313

[[ 255    0   20   95    0    0   14]
 [   6   11   10   30    4    0   12]
 [   7    1  272   66    4    2   78]
 [  44    6   36 1294   24    1  106]
 [   2    2   23   64   86    2   26]
 [   0    0   10   16    6    4   19]
 [   4    0   12   61    1    0  577]]
'''
from sklearn.multiclass import MultiOutputRegressor
from sklearn.svm import LinearSVC
Model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = MultiOutputRegressor(LinearSVC()),
    ngram_range=(1,3),
    test_size=0.4
)
#%%
'''
Building for evaluation
Classification Report:
precision    recall  f1-score   support

     course       0.77      0.73      0.75       382
 department       0.25      0.17      0.20        76
    faculty       0.72      0.74      0.73       448
      other       0.83      0.83      0.83      1465
    project       0.68      0.42      0.52       216
      staff       0.46      0.11      0.18        55
    student       0.70      0.85      0.77       671

avg / total       0.75      0.76      0.75      3313

[[ 278   18    5   73    1    0    7]
 [  13   13    9   25    6    0   10]
 [   8    3  331   24   11    1   70]
 [  49   15   49 1221   19    2  110]
 [   9    0   37   52   91    2   25]
 [   1    1   10    6    4    6   27]
 [   5    1   19   70    2    2  572]]
'''
from sklearn.multiclass import OutputCodeClassifier 
from sklearn.svm import LinearSVC
Model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = OutputCodeClassifier(LinearSVC()),
    ngram_range=(1,3),
    test_size=0.4
)

#%%
'''

'''
from sklearn.neural_network import MLPClassifier 
Model = build_and_evaluate(
    X = data['text'],
    y = data['class'],
    classifier = MLPClassifier(),
    ngram_range=(1,3),
    test_size=0.4
)

#%%
'''
Best score: 0.796
Best parameters set:
clf__alpha: 1e-05
	clf__penalty: 'elasticnet'
	vect__max_df: 1.0
	vect__max_features: 5000
	vect__ngram_range: (1, 2)
Performing grid search...
pipeline: ['vect', 'tfidf', 'clf']
parameters:
{'clf__alpha': (1e-05, 1e-06),
 'clf__penalty': ('l2', 'elasticnet'),
 'vect__max_df': (0.75, 1.0),
 'vect__max_features': (None, 5000, 10000, 50000),
 'vect__ngram_range': ((1, 3), (1, 2))}
'''
from __future__ import print_function

from pprint import pprint
from time import time
import logging

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'vect__max_df': (0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 3), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    #'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    #'clf__penalty': 'l2',
    #'clf__n_iter': (10, 50, 80)
}
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

print("Performing grid search...")
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time()
grid_search.fit(data['text'],data['class'])
print("done in %0.3fs" % (time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))