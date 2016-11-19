import numpy as np 
import pandas as pd 
from pandas import DataFrame
import re
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt 

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print '**** Dimension of train data {}'.format(train_df.shape)
print '**** Dimension of test data {}'.format(test_df.shape)

print '**** Basic statistical description:'
# train_df.info()
# print train_df.head()
# print train_df.tail()
# print train_df.describe()

###
print train_df.isnull().sum()
print test_df.isnull().sum()

###
print 'the nan embarked and other info:'
print train_df[train_df.Embarked.isnull()][['Pclass', 'Fare', 'Cabin', 'Sex']]
# sns.boxplot(x='Pclass', y='Fare', hue='Embarked', data=train_df)
# plt.axhline(y='80', color='red')
# plt.show()

train_df['Embarked'] = train_df['Embarked'].fillna('C')
# another way
# _ = train_df.set_value(train.Embarked.isnull(), 'Embarked', 'C')

###
# Fare
#
print 'the nan Fare and other info:'
print test_df[test_df.Fare.isnull()][['Pclass','Cabin','Embarked']]  #3, NaN, S

# plot fare, pclass_3, embarked_5
# sns.countplot(x= 'Fare', data=test_df[(test_df.Pclass==3) & (test_df.Embarked=='S')])
# plt.title('Histogram of Fare, Pclass_3 and Embarked_S')
# plt.show()

print test_df[(test_df.Pclass==3) & (test_df.Embarked=='S')].Fare.value_counts().head()
test_df['Fare'] = test_df['Fare'].fillna(8.05)


df = pd.concat([train_df, test_df], axis=0)
train_size = train_df.shape[0]


## Normalized the fare
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Norm_fare'] = pd.Series(scaler.fit_transform(df['Fare'].reshape(-1,1)).reshape(-1), index=df.index)


df['Cabin'] = df['Cabin'].fillna('U0')
Cabin_type = df[~df['Cabin'].isnull()]['Cabin'].map( lambda x: re.compile('([A-Z]+)').search(x).group())
# Cabin_type = pd.factorize(Cabin_type)[0]
df['Cabin_type'] = Cabin_type
del Cabin_type
# sns.countplot(x='Cabin_type', hue='Pclass', data=df)
# plt.show()

## Group_size
df['Group_num'] = df['Parch'] + df['SibSp'] + 1

train_df = df.iloc[: train_size, :]
# plot family
# fig, (axis1, axis2) = plt.subplots(1, 2)
# sns.countplot(x='Group_num', data=train_df, ax = axis1)

# family_perc = train_df[['Group_num', 'Survived']].groupby(['Group_num'], as_index=False).mean()
# sns.barplot(x='Group_num', y='Survived', data=family_perc, ax = axis2)
# plt.show()

df['Group_size'] = pd.Series('M', index=df.index)
df = df.set_value(df['Group_num']>4, 'Group_size', 'L')
df = df.set_value(df['Group_num']==1, 'Group_size', 'S')

### Age
#
average_age_df = df['Age'].mean()
std_age_df = df['Age'].std()
count_nan_age_df = df['Age'].isnull().sum()
rand_age = np.random.randint(average_age_df - std_age_df, average_age_df + std_age_df, size = count_nan_age_df)
df['Age'][np.isnan(df['Age'])] = rand_age
# conver age from float to int
df['Age'] = df['Age'].astype(int)


### Person
#
def get_person(passenger):
	age, sex = passenger
	return 'child' if age < 16 else sex

df['Person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
df.drop(['Sex'], axis=1, inplace=True)

# df.info()
df.drop(labels=['Fare','Cabin', 'Name', 'Parch', 'SibSp', 'Ticket', 'Group_num'], axis=1, inplace=True)
# df.info()
df = pd.get_dummies(df, columns=['Embarked', 'Pclass', 'Cabin_type', 'Group_size', 'Person'])
df.info()
# print df.head()

train_df = df.iloc[: train_size, :]
train_df.drop('PassengerId', axis=1, inplace=True)
train_df['Survived'] = train_df['Survived'].astype(int)

test_df = df.iloc[train_size : , :]
test_df.drop('Survived', axis=1, inplace=True)
print '****************'
test_df.info()

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import make_scorer

def get_model(estimator, parameters, X_train, y_train, scoring):  
    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator_


from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

from sklearn.metrics import roc_curve, auc
def plot_roc_curve(estimator, X, y, title):
    # Determine the false positive and true positive rates
    fpr, tpr, _ = roc_curve(y, estimator.predict_proba(X)[:,1])

    # Calculate the AUC
    roc_auc = auc(fpr, tpr)
    print ('ROC AUC: %0.2f' % roc_auc)

    # Plot of a ROC curve for a specific class
    plt.figure(figsize=(10,6))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - {}'.format(title))
    plt.legend(loc="lower right")
    plt.show()


from sklearn.metrics import accuracy_score
scoring = make_scorer(accuracy_score, greater_is_better=True)

## CV check
from sklearn.cross_validation import train_test_split
x_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
x_train_train, x_train_val, y_train_train, y_train_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)



lg = LogisticRegression()
parameters={'C': [0.5]}
clf_lgl = get_model(lg, parameters, x_train_train, y_train_train, scoring)

print clf_lgl
print accuracy_score(y_train_val, clf_lgl.predict(x_train_val))
plot_learning_curve(clf_lgl, 'Logistic Regression', x_train, y_train, cv=4)
# plt.show()

## test_df prediction
X_test = test_df.drop('PassengerId',axis=1).copy()
Y_pred = clf_lgl.predict(X_test)
submission=pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': Y_pred})
submission.to_csv('titanic_predict.csv', index=False)
