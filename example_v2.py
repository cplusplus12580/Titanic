import numpy as np 
import pandas as pd 
import re
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt 

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('train.csv')

# print '**** Dimension of train data {}'.format(train_df.shape)

# print '**** Basic statistical description:'
# train_df.info()
# print train_df.head()
# print train_df.tail()
# print train_df.describe()

###
# print train_df.isnull().sum()

###
# print 'the nan embarked and other info:'
print train_df[train_df.Embarked.isnull()][['Pclass', 'Fare', 'Cabin', 'Sex']]
# sns.boxplot(x='Pclass', y='Fare', hue='Embarked', data=train_df)
# plt.axhline(y='80', color='red')
# plt.show()

train_df['Embarked'] = train_df['Embarked'].fillna('C')
# another way
# _ = train_df.set_value(train.Embarked.isnull(), 'Embarked', 'C')

# plot
# fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))
# sns.countplot(x='Embarked', data=train_df, ax=axis1)
# sns.countplot(x='Survived', hue='Embarked', data=train_df, order=[1,0], ax=axis2)
# embark_perc = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
# sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)
# sns.plt.show()

###
# Fare
#



train_df['Cabin'] = train_df['Cabin'].fillna('U0')
Cabin_type = train_df[~train_df['Cabin'].isnull()]['Cabin'].map( lambda x: re.compile('([A-Z]+)').search(x).group())
# Cabin_type = pd.factorize(Cabin_type)[0]
train_df['Cabin_type'] = Cabin_type
del Cabin_type
print train_df['Cabin_type'].value_counts()
# plot
# sns.countplot(x='Cabin_type', hue='Pclass', data=train_df)
# plt.show()

## Group_size
train_df['Group_num'] = train_df['Parch'] + train_df['SibSp'] + 1

# plot family
# fig, (axis1, axis2) = plt.subplots(1, 2)
# sns.countplot(x='Group_num', data=train_df, ax = axis1)

# family_perc = train_df[['Group_num', 'Survived']].groupby(['Group_num'], as_index=False).mean()
# sns.barplot(x='Group_num', y='Survived', data=family_perc, ax = axis2)
# plt.show()

train_df['Group_size'] = pd.Series('M', index=train_df.index)
train_df = train_df.set_value(train_df['Group_num']>4, 'Group_size', 'L')
train_df = train_df.set_value(train_df['Group_num']==1, 'Group_size', 'S')

# groupsize_perc = train_df[['Group_size', 'Survived']].groupby(['Group_size'], as_index=False).mean()
# sns.barplot(x='Group_size', y='Survived', data=groupsize_perc)
# plt.show()

### Age
#
average_age_df = train_df['Age'].mean()
std_age_df = train_df['Age'].std()
count_nan_age_df = train_df['Age'].isnull().sum()
rand_age = np.random.randint(average_age_df - std_age_df, average_age_df + std_age_df, size = count_nan_age_df)
train_df['Age'][np.isnan(train_df['Age'])] = rand_age

## Normalized the fare
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df['Norm_fare'] = pd.Series(scaler.fit_transform(train_df['Fare'].reshape(-1,1)).reshape(-1), index=train_df.index)



### Person
#
def get_person(passenger):
	age, sex = passenger
	return 'child' if age < 16 else sex

train_df['Person'] = train_df[['Age', 'Sex']].apply(get_person, axis=1)

# fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
# sns.countplot(x='Person', data=train_df, ax=axis1)
# person_perc = train_df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()
# sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])
# plt.show()

# train_df.info()
train_df.drop(labels=['PassengerId', 'Fare','Cabin', 'Name', 'Sex', 'Parch', 'SibSp', 'Ticket', 'Group_num'], axis=1, inplace=True)
# train_df.info()
train_df = pd.get_dummies(train_df, columns=['Embarked', 'Pclass', 'Cabin_type', 'Group_size', 'Person'])
# train_df.info()
# print train_df.head()

train_df['Survived'] = train_df['Survived'].astype(int)



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

# （1）中等、没妹子
x=[24, 0, 1, 0, 0, 0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1]
print clf_lgl.predict(x)
# （2）高级、没妹子
x=[24, 0, 1, 0, 0, 1,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1]
print clf_lgl.predict(x)
# （3）中等、有妹子
x=[24, 0, 1, 0, 0, 0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1]
print clf_lgl.predict(x)
# （4）高级、有妹子
x=[24, 0, 1, 0, 0, 1,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1]
print clf_lgl.predict(x)

