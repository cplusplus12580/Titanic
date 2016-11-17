# Titanic

## 库
```python
import pandas as pd
from pandas import Series, DataFrame

import numpy as np 
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
```
## 导入数据
```python
train_df = pd.read_csv('train.csv', dtype={'Age': np.float64}, )
test_df = pd.read_csv('test.csv', dtype={'Age': np.float64}, )
```
## 数据
- 训练集信息
```python
train_df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 891 entries, 0 to 890
Data columns (total 12 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            714 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          204 non-null object
Embarked       889 non-null object
dtypes: float64(2), int64(5), object(5)
memory usage: 83.6+ KB
```
得到的训练集信息如下：
> * (1)共有891（0-890）条数据
> * (2)12字段， PassengerId、Survived、Pclass、Name、Sex、Age、SibSp、Parch、Ticket、Fare、Cabin、Embarked
> * (3)Age、Cabin和Embarked字段中含有空值
> * (4)数据类型有float、int和object
  
- 测试集信息
```python
test_df.info()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 418 entries, 0 to 417
Data columns (total 11 columns):
PassengerId    418 non-null int64
Pclass         418 non-null int64
Name           418 non-null object
Sex            418 non-null object
Age            332 non-null float64
SibSp          418 non-null int64
Parch          418 non-null int64
Ticket         418 non-null object
Fare           417 non-null float64
Cabin          91 non-null object
Embarked       418 non-null object
dtypes: float64(2), int64(4), object(5)
memory usage: 36.0+ KB
```
得到的测试集信息如下：
> * (1)共有418（0-417）条数据
> * (2)11个字段，PassengerId、Pclass、Name、Sex、Age、SibSp、Parch、Ticket、Fare、Cabin、Embarked，缺失的Survived字段是需要预测出来的
> * (3)Age、Fare、Cabin字段中含有空值
> * (4)数据类型有float、int和object

- 数据显示
```python
print train_df.head()
```
```python
   PassengerId  Survived  Pclass                                               Name     Sex   Age  SibSp   \
0            1         0       3                             Braund, Mr. Owen Harris    male  22.0      1
1            2         1       1   Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1
2            3         1       3                              Heikkinen, Miss. Laina  female  26.0      0 
3            4         1       1        Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1
4            5         0       3                            Allen, Mr. William Henry    male  35.0      0

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S 
```
## 数据清洗方案
### 方案一

将训练集和测试集的数据按照行的形式进行合并
```python
train_size = train.shape[0]
df = pd.concat([train_df, test_df], axis=0)
```
一些比较复杂的字段先不处理
```python
df = df.drop(['Name', 'Ticket'], axis=1)
```
Embarked字段，在df中含有2个空值
```python
print df['Embarked'].isnull().sum()
```
```python
2
```
由于缺失值的数量非常小，所以用最多的一个类别代替，类别统计如下：
```python
print titanic_df['Embarked'].value_counts()
```
```python
S    914
C    270
Q    123
```
所以用类别“S”来代替缺失值
```python
df['Embarked'] = df['Embarked'].fillna('S')
```

画图查看一下不同的Embarked幸存率情况，折线图形式
```python
train_df = df.iloc[: train_size, :]
sns.factorplot('Embarked', 'Survived', data=train_df, size=4, aspect=3)
sns.plt.show()
```
![](raw/figure_1.png?raw=true)

柱状图形式
```python
sns.factorplot('Embarked', 'Survived', data=train_df, size=4, aspect=1, kind='bar')
sns.plt.show()
```
![](raw/figure_3.png?raw=true)

Embarked更详细的图表信息
```python
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=train_df, order=[1,0], ax=axis2)
embark_perc = df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)
sns.plt.show()
```
![](raw/figure_2.png?raw=true)

#### Fare
Fare在训练集中含有空值，只有一个，用均值替换
```python
df['Fare'].filllna(test_df['Fare'].mean(), inplace=True)
```
票价分布直方图
```python
train_df = df.iloc[: train_size, :]
train_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100) # xlim=(0,50)
plt.show()
```
![](raw/figure_4.png?raw=true)

分析存活和非存活乘客的票价情况,均值和方差
```python
fare_not_survived = train_df['Fare'][train_df['Survived'] == 0]
fare_survived = train_df['Fare'][train_df['Survived'] == 1]
# 均值和误差
average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])
average_fare.index.names = std_fare.index.names = ['Survived']
average_fare.plot(yerr=std_fare, kind='bar', legend=False)
plt.show()
```
![](raw/figure_5.png?raw=true)

#### Age
Age字段在训练集和测试集中均含有空值，用均值和标准差限定的随机数代替，
```python
average_age_df = df['Age'].mean()
std_age_df = df['Age'].std()
count_nan_age_df = df['Age'].isnull().sum()
rand_age = np.random.randint(average_age_df - std_age_df, average_age_df + std_age_df, size = count_nan_age_df)
```
在1*2的图中画出训练集的原始Age和新Age
```
# plot original Age values in train_df
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
axis1.set_title('Original Age values - Titanic')
axis2.set_title(' New Age values - Tinatic')
train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
```
```python
# 用随机数替换NaN
df['Age'][np.isnan(df['Age'])] = rand_age
# conver age from float to int
df['Age'] = df['Age'].astype(int)
train_df = df.iloc[: train_size, :]
train_df['Age'].hist(bins=70, ax=axis2)
plt.show()
```
![6](raw/figure_6.png?raw=true)
