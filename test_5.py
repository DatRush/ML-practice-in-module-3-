import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

adult = pd.read_csv('/Users/tair/Library/Mobile Documents/com~apple~CloudDocs/Code/курсы/modul_3_all/adult/adult.data',
                    names=['age', 'workclass', 'fnlwgt', 'education',
                           'education-num', 'marital-status', 'occupation',
                           'relationship', 'race', 'sex', 'capital-gain',
                           'capital-loss', 'hours-per-week', 'native-country', 'salary'])

counts = adult['native-country'].value_counts()
values_to_replace = counts[counts < 100].index

adult['native-country'] = adult['native-country'].replace(values_to_replace, 'other')
adult.dropna
adult['salary'] = (adult['salary'] != ' <=50K').astype('int32')
adult = pd.get_dummies(adult, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])


a_features = adult[['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']].values
norm_features = (a_features - a_features.mean(axis=0)) / a_features.std(axis=0)
adult.loc[:, ['age', 'education-num', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss']] = norm_features


X = adult[list(set(adult.columns) - set(['salary']))].values
y = adult['salary'].values

X = np.hstack([np.ones(X.shape[0])[:, np.newaxis], X])
m = X.shape[1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(penalty='l2', C=0.88)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

def print_logisitc_metrics(y_test, predictions):
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print(f'acc = {acc:.2f} F1-score = {f1:.2f}')
    
def calc_and_plot_roc(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.title('Receiver Operating Characteristic', fontsize=15)
    plt.xlabel('False positive rate (FPR)', fontsize=15)
    plt.ylabel('True positive rate (TPR)', fontsize=15)
    plt.legend(fontsize=15)
    plt.show() 
    
print_logisitc_metrics(y_test, predictions)

y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

print(conf_matrix)

y_pred_proba = model.predict_proba(X_test)[:, 1] 
print(calc_and_plot_roc(y_test, y_pred_proba))

scores = cross_val_score(model, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average score:", np.mean(scores))