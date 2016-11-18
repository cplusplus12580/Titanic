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

字段说明：
> - Survived: 是否幸存。0:否；1:是
> - Pclass: 船舱等级。1:高级；2:中级；3:低级
> - Name: 乘客姓名
> - Sex: 乘客性别
> - Age: 乘客年龄
> - SibSp: 随行的兄妹、配偶的数量
> - Parch: 随行的父母、子女的数量
> - Ticket: 船票的号码
> - Fare: 船票价格
> - Cabin: 类似于火车的座位
> - Embarked: 登船港口。C=Cherbourg; Q=Queenstown; S=Southampton
  
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
### 查看缺省值
训练集：
```python
print train_df.isnull().sum()

PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
dtype: int64
```
测试集：
```python
print test_df.isnull().sum()

PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
Embarked         0
dtype: int64
```

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
由于缺失值的数量非常小，可以查看一下这两个乘客的其他属性：
```python
print train_df[train_df.Embarked.isnull()][['Pclass', 'Fare', 'Cabin', 'Sex']]
```
```python
     Pclass  Fare Cabin     Sex
61        1  80.0   B28  female
829       1  80.0   B28  female
```
```python
sns.boxplot(x='Pclass', y='Fare', hue='Embarked', data=train_df)
plt.axhline(y='80', color='red')
plt.show()
```
![](raw/figure_14.png?raw=true)
从上图中可以看出，Pclass为1、船票费用80的乘客都集中在从C（Cherbourg）港口登船。所以将缺失值替换为“C”
```python
train_df['Embarked'] = train_df['Embarked'].fillna('C')
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

把Embarked的值转换为三个新的字段
```python
embark_dummies_df = df.get_dummies(df['Embarked'])
df = df.join(embark_dummies_df)
df.drop(['Embarked'], axis=1, inplace=True)
```

#### Fare
Fare在训练集中含有空值，查看相关信息：
```python
print test_df[test_df.Fare.isnull()][['Pclass','Cabin','Embarked']]
```
```python
     Pclass Cabin Embarked
152       3   NaN        S
```
该乘客从S港口登船的，船舱等级为3，Cabin未知，以前面两点信息进行统计：
```python
sns.countplot(x= 'Fare', data=test_df[(test_df.Pclass==3) & (test_df.Embarked=='S')])
plt.title('Histogram of Fare, Pclass_3 and Embarked_S')
plt.show()
```
![](raw/figure_15.png?raw=true)

票价最多的类别：
```python
print test_df[(test_df.Pclass==3) & (test_df.Embarked=='S')].Fare.value_counts().head()
```
```python
8.0500    17
7.7750    10
7.8958    10
7.8542     8
8.6625     8
```
所以，将缺失值替换为8.05：
```python
test_df['Fare'] = test_df['Fare'].fillna(8.05)
```
票价正则化：
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Norm_fare'] = pd.Series(scaler.fit_transform(df['Fare'].reshape(-1,1)).reshape(-1), index=df.index)
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

#### Cabin
在训练集和测试集中均含有大量的缺失值，将两个数据集合并为df
```python
df = pd.concat([train_df, test_df], axis=0)
```
先将缺失值填充为U0，然后从Cabin字段中提取Cabin的类别，存放在新的Cabin_type字段中

```python
df['Cabin'] = df['Cabin'].fillna('U0')
Cabin_type = df[~df['Cabin'].isnull()]['Cabin'].map( lambda x: re.compile('([A-Z]+)').search(x).group())

Cabin_type = pd.factorize(Cabin_type)[0]
df['Cabin_type'] = Cabin_type
del Cabin_type
```
factorize能够将类别分别对应成数字
```python
print df['Cabin_type'].value_counts()

U    1014
C      94
B      65
D      46
E      41
A      22
F      21
G       5
T       1
```
factor后：
```python
print df['Cabin_type'].value_counts()

0    1014
1      94
6      65
4      46
2      41
5      22
7      21
3       5
8       1
```



#### Title

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
# Age类型从float转换成int
df['Age'] = df['Age'].astype(int)
train_df = df.iloc[: train_size, :]
train_df['Age'].hist(bins=70, ax=axis2)
plt.show()
```
![](raw/figure_6.png?raw=true)

幸存和非幸存乘客的年龄分布情况
```python
facet = sns.FacetGrid(train_df, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plt.show()
```
![](raw/figure_7.png?raw=true)

各个年龄的幸存率
```python
fig, axis1 = plt.subplots(1,1, figsize=(18,6))
average_age = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
plt.show()
```
![](raw/figure_8.png?raw=true)


#### Parch和SibSp字段，合并成Group_size字段
Group_num字段为Parch和SibSp的和，并添加一个字段Has_family用来判断是否有Family
```python
df['Group_num'] = df['Parch'] + df['SibSp'] + 1
```
Group_num的幸存率
```python
train_df = df.iloc[: train_size, :]

fig, (axis1, axis2) = plt.subplots(1, 2)
sns.countplot(x='Group_num', data=train_df, ax = axis1)
family_perc = train_df[['Group_num', 'Survived']].groupby(['Group_num'], as_index=False).mean()
sns.barplot(x='Group_num', y='Survived', data=family_perc, ax = axis2)
plt.show()
```
![](raw/figure_16.png?raw=true)

数量在2～4之间幸存率明显高于其他的，因此将Group分成三类：
```python
df['Group_size'] = pd.Series('M', index=df.index)
df = df.set_value(df['Group_num']>4, 'Group_size', 'L')
df = df.set_value(df['Group_num']==1, 'Group_size', 'S')
```

左图中可以看出单独出游的乘客数量多于和家人一起出游的乘客数量，但是和家人一起出游的乘客的幸存均值明显高于相对的群体

再看看不同家人数量的分布情况
```python
fig, (axis1, axis2) = plt.subplots(1, 2)
sns.countplot(x='Family', data=train_df, ax = axis1)

family_perc = train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, ax = axis2)
plt.show()
```
![](raw/figure_10.png?raw=true)
携带家人数量在1～3之间，幸存均值远远高于其他的情况

#### Sex
根据“儿童、女士优先”的原则，将乘客划分为男士(male)、女士(female)和儿童(child)存放在新的字段Person中
```python
def get_person(passenger):
	age, sex = passenger
	return 'child' if age < 16 else sex

df['Person'] = df[['Age', 'Sex']].apply(get_person, axis=1)
df.drop(['Sex'], axis=1, inplace=True)
```
不同人群的数量分布及幸存均值
```python
train_df = train_df = df.iloc[: train_size, :]

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(x='Person', data=train_df, ax=axis1)

person_perc = train_df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])
plt.show()
```
![](raw/figure_11.png?raw=true)
儿童和女士的优势好明显。

看看单身狗的幸存情况
```python
def get_single_male(passenger):
	person, hasfamily = passenger
	return 1 if person == 'male' and hasfamily == 0 else 0

df['Single_male'] = df[['Person', 'Has_family']].apply(get_single_male, axis=1)
train_df = train_df = df.iloc[: train_size, :]

fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(x='Single_male', data=train_df, ax=axis1)
person_perc = train_df[['Single_male', 'Survived']].groupby(['Single_male'], as_index=False).mean()
sns.barplot(x='Single_male', y='Survived', data=person_perc, ax=axis2)
plt.show()
```
![](raw/figure_12.png?raw=true)
幸存率不到1/5.

将child和female转换成field，male劣势太大了，就不考虑在内了
```python
person_dummies_df = pd.get_dummies(df['Person'])
person_dummies_df.columns = ['Child', 'Female', 'Male']
person_dummies_df.drop(['Male'], axis=1, inplace=True)

df = df.join(person_dummies_df)
df.drop(['Person'], axis=1, inplace=True)
```

### Pclass
查看Pclass的种类
```python
print df['Pclass'].value_counts()

3    1163
1     524
2     458
Name: Pclass, dtype: int64
```
船舱的等级有三类，分为1、2、3等，档次依次降低。
```python
sns.factorplot('Pclass', 'Survived', order=[1, 2, 3], data=df, size=5)
plt.show()
```
![](raw/figure_13.png?raw=true)
Pclass为3时，幸存率只有25%左右，非常低

```python
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1309 entries, 0 to 417
Data columns (total 16 columns):
Age            1309 non-null int64
Cabin          1309 non-null object
Embarked       1309 non-null object
Fare           1309 non-null float64
Name           1309 non-null object
Parch          1309 non-null int64
PassengerId    1309 non-null int64
Pclass         1309 non-null int64
SibSp          1309 non-null int64
Survived       891 non-null float64
Ticket         1309 non-null object
Norm_fare      1309 non-null float64
Cabin_type     1309 non-null object
Group_num      1309 non-null int64
Group_size     1309 non-null object
Person         1309 non-null object
dtypes: float64(3), int64(6), object(7)
memory usage: 173.9+ KB
```
```python
df.drop(labels=['Fare','Cabin', 'Name', 'Parch', 'SibSp', 'Ticket', 'Group_num'], axis=1, inplace=True)
df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1309 entries, 0 to 417
Data columns (total 9 columns):
Age            1309 non-null int64
Embarked       1309 non-null object
PassengerId    1309 non-null int64
Pclass         1309 non-null int64
Survived       891 non-null float64
Norm_fare      1309 non-null float64
Cabin_type     1309 non-null object
Group_size     1309 non-null object
Person         1309 non-null object
dtypes: float64(2), int64(3), object(4)
memory usage: 102.3+ KB
```
```python
df = pd.get_dummies(df, columns=['Embarked', 'Pclass', 'Cabin_type', 'Group_size', 'Person'])
df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
Int64Index: 1309 entries, 0 to 417
Data columns (total 25 columns):
Age              1309 non-null int64
PassengerId      1309 non-null int64
Survived         891 non-null float64
Norm_fare        1309 non-null float64
Embarked_C       1309 non-null float64
Embarked_Q       1309 non-null float64
Embarked_S       1309 non-null float64
Pclass_1         1309 non-null float64
Pclass_2         1309 non-null float64
Pclass_3         1309 non-null float64
Cabin_type_A     1309 non-null float64
Cabin_type_B     1309 non-null float64
Cabin_type_C     1309 non-null float64
Cabin_type_D     1309 non-null float64
Cabin_type_E     1309 non-null float64
Cabin_type_F     1309 non-null float64
Cabin_type_G     1309 non-null float64
Cabin_type_T     1309 non-null float64
Cabin_type_U     1309 non-null float64
Group_size_L     1309 non-null float64
Group_size_M     1309 non-null float64
Group_size_S     1309 non-null float64
Person_child     1309 non-null float64
Person_female    1309 non-null float64
Person_male      1309 non-null float64
dtypes: float64(23), int64(2)
memory usage: 265.9 KB
```
```python
print df.head()
   Age  PassengerId  Survived  Norm_fare  Embarked_C  Embarked_Q  Embarked_S  \
0   22            1       0.0  -0.503176         0.0         0.0         1.0   
1   38            2       1.0   0.734809         1.0         0.0         0.0   
2   26            3       1.0  -0.490126         0.0         0.0         1.0   
3   35            4       1.0   0.383263         0.0         0.0         1.0   
4   35            5       0.0  -0.487709         0.0         0.0         1.0   

   Pclass_1  Pclass_2  Pclass_3     ...       Cabin_type_F  Cabin_type_G  \
0       0.0       0.0       1.0     ...                0.0           0.0   
1       1.0       0.0       0.0     ...                0.0           0.0   
2       0.0       0.0       1.0     ...                0.0           0.0   
3       1.0       0.0       0.0     ...                0.0           0.0   
4       0.0       0.0       1.0     ...                0.0           0.0   

   Cabin_type_T  Cabin_type_U  Group_size_L  Group_size_M  Group_size_S  \
0           0.0           1.0           0.0           1.0           0.0   
1           0.0           0.0           0.0           1.0           0.0   
2           0.0           1.0           0.0           0.0           1.0   
3           0.0           0.0           0.0           1.0           0.0   
4           0.0           1.0           0.0           0.0           1.0   

   Person_child  Person_female  Person_male  
0           0.0            0.0          1.0  
1           0.0            1.0          0.0  
2           0.0            1.0          0.0  
3           0.0            1.0          0.0  
4           0.0            0.0          1.0 
```


