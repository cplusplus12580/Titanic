import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV,  RandomizedSearchCV

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
import warnings
warnings.filterwarnings("ignore")

train_df = pd.read_csv('train.csv', dtype={'Age': np.float64}, )
test_df = pd.read_csv('test.csv', dtype={'Age': np.float64}, )

# print train_df.head()
printtrain_df.isnull().sum()
test_df.isnull().sum()

train_size = train_df.shape[0]
# print train_size
df = pd.concat([train_df, test_df], axis=0)
df = df.drop(['Name', 'Ticket'], axis=1)

# Embarked
#
# print df['Embarked'].isnull().sum()


# print df['Embarked'].value_counts()
df['Embarked'] = df['Embarked'].fillna('S')

train_df = df.iloc[: train_size, :]

# plot
# sns.factorplot('Embarked', 'Survived', data=train_df, size=4, aspect=3)
# sns.factorplot('Embarked', 'Survived', data=train_df, size=4, aspect=1, kind='bar')

# fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))
# sns.countplot(x='Embarked', data=train_df, ax=axis1)
# sns.countplot(x='Survived', hue='Embarked', data=train_df, order=[1,0], ax=axis2)
# embark_perc = df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
# sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)
# sns.plt.show()

embark_dummies_df = pd.get_dummies(df['Embarked'])
df = df.join(embark_dummies_df)
df.drop(['Embarked'], axis=1, inplace=True)

# Fare
#
df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
# get fare for survived & didn't survive passengers
train_df = df.iloc[: train_size, :]
fare_not_survived = train_df['Fare'][train_df['Survived'] == 0]
fare_survived = train_df['Fare'][train_df['Survived'] == 1]
# get the average and std for fare of survived/not survived passengers
average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])
# plot
# train_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100) # xlim=(0,50)
# average_fare.index.names = std_fare.index.names = ['Survived']
# average_fare.plot(yerr=std_fare, kind='bar', legend=False)
# plt.show()

# Age
#
# get avereage, std and number of NaN values in df
average_age_df = df['Age'].mean()
std_age_df = df['Age'].std()
count_nan_age_df = df['Age'].isnull().sum()
# generate random numbers between (mean-std) and (mean+std)
rand_age = np.random.randint(average_age_df - std_age_df, average_age_df + std_age_df, size = count_nan_age_df)

# plot original Age values in train_df
# fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
# axis1.set_title('Original Age values - Titanic')
# axis2.set_title(' New Age values - Tinatic')

# train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)

# # fill NaN values in Age colum with random values generated
df['Age'][np.isnan(df['Age'])] = rand_age
# conver age from float to int
df['Age'] = df['Age'].astype(int)
train_df = df.iloc[: train_size, :]
# train_df['Age'].hist(bins=70, ax=axis2)
# plt.show()

# continue with plot Age column
# facet = sns.FacetGrid(train_df, hue='Survived', aspect=4)
# facet.map(sns.kdeplot, 'Age', shade=True)
# facet.set(xlim=(0, train_df['Age'].max()))
# facet.add_legend()

# average survived passengers by age
# fig, axis1 = plt.subplots(1,1, figsize=(18,6))
# average_age = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
# sns.barplot(x='Age', y='Survived', data=average_age)
# plt.show()

# Cabin
#
# has lots of NaN values, so it won't cause a remarkable impact on prediction
df.drop('Cabin', axis=1, inplace=True)

# Parch and SibSp
#
# merges into Family

df['Family'] = df['Parch'] + df['SibSp']
df.drop('Parch', axis=1, inplace=True)
df.drop('SibSp', axis=1, inplace=True)
df['Has_family'] = df['Family']
df['Has_family'].loc[df['Family'] > 0] = 1
df['Has_family'].loc[df['Family'] == 0] = 0

train_df = df.iloc[: train_size, :]

# plot Has_family
# fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize=(10, 5))
# sns.countplot(x='Has_family', data=train_df, order=[1,0], ax = axis1)

# family_perc = train_df[['Has_family', 'Survived']].groupby(['Has_family'], as_index=False).mean()
# sns.barplot(x='Has_family', y='Survived', data=family_perc, order=[1,0],ax = axis2)
# axis1.set_xticklabels(['With Family', 'Alone'], rotation=0)
# plt.show()

# plot family
# fig, (axis1, axis2) = plt.subplots(1, 2)
# sns.countplot(x='Family', data=train_df, ax = axis1)

# family_perc = train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
# sns.barplot(x='Family', y='Survived', data=family_perc, ax = axis2)
# plt.show()

# Sex
#
def get_person(passenger):
	age, sex = passenger
	return 'child' if age < 16 else sex

df['Person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
df.drop(['Sex'], axis=1, inplace=True)

train_df = train_df = df.iloc[: train_size, :]

# plot
# fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
# sns.countplot(x='Person', data=train_df, ax=axis1)
# # average of survived for each Person(male, female, or child)
# person_perc = train_df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()
# sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])
# plt.show()

# plot danshengou
def get_single_male(passenger):
	person, hasfamily = passenger
	return 1 if person == 'male' and hasfamily == 0 else 0

df['Single_male'] = df[['Person', 'Has_family']].apply(get_single_male, axis=1)
train_df = train_df = df.iloc[: train_size, :]

# fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
# sns.countplot(x='Single_male', data=train_df, ax=axis1)
# person_perc = train_df[['Single_male', 'Survived']].groupby(['Single_male'], as_index=False).mean()
# sns.barplot(x='Single_male', y='Survived', data=person_perc, ax=axis2)
# plt.show()


person_dummies_df = pd.get_dummies(df['Person'])
person_dummies_df.columns = ['Child', 'Female', 'Male']
person_dummies_df.drop(['Male'], axis=1, inplace=True)
df = df.join(person_dummies_df)
df.drop(['Person'], axis=1, inplace=True)

# Pclass
#
# print df['Pclass'].value_counts()
# sns.factorplot('Pclass', 'Survived', order=[1, 2, 3], data=df, size=5)
# plt.show()

# pclass_dummies_df  = pd.get_dummies(df['Pclass'])
# pclass_dummies_df.columns = ['Class_1','Class_2','Class_3']
# pclass_dummies_df.drop(['Class_3'], axis=1, inplace=True)
# df.drop(['Pclass'], axis=1, inplace=True)
# df = df.join(pclass_dummies_df)


# df.info()
# df.describe()
# print df.head()

df.drop(['Has_family'], axis=1, inplace=True)
df.drop(['Single_male'], axis=1, inplace=True)
df['Child'] = df['Child'].astype(int)
df['Female'] = df['Female'].astype(int)
df['C'] = df['C'].astype(int)
df['Q'] = df['Q'].astype(int)
df['S'] = df['S'].astype(int)
# df['Pclass_1'] = df['Pclass_1'].astype(int)
# df['Pclass_2'] = df['Pclass_2'].astype(int)

# df.info()
