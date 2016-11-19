# Titanic
## 目录
- 一、库
- 二、数据
    - 2.1 导入数据
    - 2.2 数据初认识
    - 2.3 数据示例
    - 2.4 缺失值情况
- 三、清洗数据
    - 3.1 Embarked字段
    - 3.2 Age字段
    - 3.3 Age和Sex字段
    - 3.4 Cabin字段
    - 3.5 Fare字段
    - 3.6 Parch和SibSp字段
    - 3.7 Pclass字段
    - 3.8 类型化处理
- 四、机器学习
    - 4.1 CV数据集
    - 4.2 CV检验
    - 4.3 测试集预测
- 五、尚需优化的问题

## 一、库
```python
import pandas as pd
from pandas import Series, DataFrame

import numpy as np 
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)

import warnings
warnings.filterwarnings("ignore")
```
## 二、数据

- 2.1 导入数据
```python
train_df = pd.read_csv('train.csv')
```

- 2.2 数据初认识
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
  
- 2.3 数据示例

```python
print train_df.head()
```
```python
   PassengerId  Survived  Pclass  \
0            1         0       3   
1            2         1       1   
2            3         1       3   
3            4         1       1   
4            5         0       3   

                                                Name     Sex   Age  SibSp  \
0                            Braund, Mr. Owen Harris    male  22.0      1   
1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   
2                             Heikkinen, Miss. Laina  female  26.0      0   
3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   
4                           Allen, Mr. William Henry    male  35.0      0   

   Parch            Ticket     Fare Cabin Embarked  
0      0         A/5 21171   7.2500   NaN        S  
1      0          PC 17599  71.2833   C85        C  
2      0  STON/O2. 3101282   7.9250   NaN        S  
3      0            113803  53.1000  C123        S  
4      0            373450   8.0500   NaN        S 
```

- 2.4 缺失值情况
```python
print train_df.isnull().sum()
```
```python
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
```

## 三、数据清洗

### 3.1 Embarked
Embarked字段，在df中含有2个空值

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

Embarked更详细的图表信息
```python
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))
sns.countplot(x='Embarked', data=train_df, ax=axis1)
sns.countplot(x='Survived', hue='Embarked', data=train_df, order=[1,0], ax=axis2)
embark_perc = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
sns.barplot(x='Embarked', y='Survived', data=embark_perc, order=['S', 'C', 'Q'], ax=axis3)
sns.plt.show()
```
![](raw/figure_17.png?raw=true)

### 3.2 Age

- 缺失的值用随机值代替
```python
average_age_df = train_df['Age'].mean()
std_age_df = train_df['Age'].std()
count_nan_age_df = train_df['Age'].isnull().sum()
rand_age = np.random.randint(average_age_df - std_age_df, average_age_df + std_age_df, size = count_nan_age_df)
train_df['Age'][np.isnan(train_df['Age'])] = rand_age
```

- 不同年龄的幸存率分布情况
```python
fig, axis1 = plt.subplots(1,1, figsize=(18,6))
average_age = train_df[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=average_age)
plt.show()
```
![](raw/figure_8.png?raw=true)

### 3.3 Age和Sex

为了检验女士和儿童是否有优势，添加一个由Age和Sex生成的字段Person，值为child、female或male。
```python
def get_person(passenger):
    age, sex = passenger
    return 'child' if age < 16 else sex
train_df['Person'] = train_df[['Age', 'Sex']].apply(get_person, axis=1)
```

不同人群的数量分布及幸存率
```python
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
sns.countplot(x='Person', data=train_df, ax=axis1)

person_perc = train_df[['Person', 'Survived']].groupby(['Person'], as_index=False).mean()
sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male', 'female', 'child'])
plt.show()
```
![](raw/figure_18.png?raw=true)

儿童和女士的优势好明显，尤其是女士，幸存率接近80%。

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

### 3.4 Cabin

训练集中含有大量的缺失值。
先将缺失值填充为U0，然后从Cabin字段中提取Cabin的类别，存放在新的Cabin_type字段中

```python
df['Cabin'] = df['Cabin'].fillna('U0')
Cabin_type = df[~df['Cabin'].isnull()]['Cabin'].map( lambda x: re.compile('([A-Z]+)').search(x).group())
df['Cabin_type'] = Cabin_type
del Cabin_type
```

```python
print df['Cabin_type'].value_counts()
```
```python
U    687
C     59
B     47
D     33
E     32
A     15
F     13
G      4
T      1
```

### 3.5 Fare

票价分布直方图
```python
train_df = df.iloc[: train_size, :]
train_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100) # xlim=(0,50)
plt.show()
```
![](raw/figure_4.png?raw=true)

幸存和非幸存乘客的年龄分布情况
```python
facet = sns.FacetGrid(train_df, hue='Survived', aspect=4)
facet.map(sns.kdeplot, 'Age', shade=True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()
plt.show()
```
![](raw/figure_7.png?raw=true)

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

票价正则化：
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train_df['Norm_fare'] = pd.Series(scaler.fit_transform(train_df['Fare'].reshape(-1,1)).reshape(-1), index=train_df.index)
```

### 3.6 Parch和SibSp

这两个字段都和家属有关，所以考虑把它们转换成一个字段。转换过程如下：

- 第一步：Group_num字段，也就是Parch和SibSp的和，再加上自己
```python
df['Group_num'] = df['Parch'] + df['SibSp'] + 1
```

- 查看Group_num的幸存率
```python
fig, (axis1, axis2) = plt.subplots(1, 2)
sns.countplot(x='Group_num', data=train_df, ax = axis1)
family_perc = train_df[['Group_num', 'Survived']].groupby(['Group_num'], as_index=False).mean()
sns.barplot(x='Group_num', y='Survived', data=family_perc, ax = axis2)
plt.show()
```
![](raw/figure_16.png?raw=true)

由于数量在2～4之间幸存率明显高于其他的，因此将Group_num分成三类。

- Group_size字段，1对应于S，2～4对应于M，5及以上对应于L
```python
df['Group_size'] = pd.Series('M', index=df.index)
df = df.set_value(df['Group_num']>4, 'Group_size', 'L')
df = df.set_value(df['Group_num']==1, 'Group_size', 'S')
```

```python
groupsize_perc = train_df[['Group_size', 'Survived']].groupby(['Group_size'], as_index=False).mean()
sns.barplot(x='Group_size', y='Survived', data=groupsize_perc)
plt.show()
```
![](raw/figure_20.png?raw=true)

### 3.7 Pclass
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

### 3.8 类型化处理

```python
train_df.into()
```
```python
RangeIndex: 891 entries, 0 to 890
Data columns (total 17 columns):
PassengerId    891 non-null int64
Survived       891 non-null int64
Pclass         891 non-null int64
Name           891 non-null object
Sex            891 non-null object
Age            891 non-null float64
SibSp          891 non-null int64
Parch          891 non-null int64
Ticket         891 non-null object
Fare           891 non-null float64
Cabin          891 non-null object
Embarked       891 non-null object
Cabin_type     891 non-null object
Group_num      891 non-null int64
Group_size     891 non-null object
Norm_fare      891 non-null float64
Person         891 non-null object
dtypes: float64(3), int64(6), object(8)
memory usage: 118.4+ KB
```
```python
df.drop(labels=['PassengerId','Fare','Cabin', 'Name', 'Sex', 'Parch', 'SibSp', 'Ticket', 'Group_num'], axis=1, inplace=True)
df.info()
```
```python
Data columns (total 8 columns):
Survived       891 non-null int64
Pclass         891 non-null int64
Age            891 non-null float64
Embarked       891 non-null object
Cabin_type     891 non-null object
Group_size     891 non-null object
Norm_fare      891 non-null float64
Person         891 non-null object
dtypes: float64(2), int64(3), object(4)
memory usage: 62.7+ KB
```
```python
train_df = pd.get_dummies(train_df, columns=['Embarked', 'Pclass', 'Cabin_type', 'Group_size', 'Person'])
train_df.info()
```
```python
RangeIndex: 891 entries, 0 to 890
Data columns (total 24 columns):
Survived         891 non-null int64
Age              891 non-null float64
Norm_fare        891 non-null float64
Embarked_C       891 non-null float64
Embarked_Q       891 non-null float64
Embarked_S       891 non-null float64
Pclass_1         891 non-null float64
Pclass_2         891 non-null float64
Pclass_3         891 non-null float64
Cabin_type_A     891 non-null float64
Cabin_type_B     891 non-null float64
Cabin_type_C     891 non-null float64
Cabin_type_D     891 non-null float64
Cabin_type_E     891 non-null float64
Cabin_type_F     891 non-null float64
Cabin_type_G     891 non-null float64
Cabin_type_T     891 non-null float64
Cabin_type_U     891 non-null float64
Group_size_L     891 non-null float64
Group_size_M     891 non-null float64
Group_size_S     891 non-null float64
Person_child     891 non-null float64
Person_female    891 non-null float64
Person_male      891 non-null float64
dtypes: float64(23), int64(2)
memory usage: 174.1 KB
```
```python
print train_df.info()
memory usage: 174.1 KB
     Survived   Age  Norm_fare  Embarked_C  Embarked_Q  Embarked_S  \
0           0  22.0  -0.502445         0.0         0.0         1.0   
1           1  38.0   0.786845         1.0         0.0         0.0   
2           1  26.0  -0.488854         0.0         0.0         1.0   
3           1  35.0   0.420730         0.0         0.0         1.0   
4           0  35.0  -0.486337         0.0         0.0         1.0   

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

## 四、机器学习

### 4.1 CV数据集

为了进行Cross-Validation检验，将train_df数据按照7:3的比例分成训练集和检验集
```python
from sklearn.cross_validation import train_test_split
x_train = train_df.drop('Survived', axis=1)
y_train = train_df['Survived']
x_train_train, x_train_val, y_train_train, y_train_val = train_test_split(x_train, y_train, test_size=0.3, random_state=1)
```
### 4.2 CV检验
```python
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
```
中间含有一段CV检验的过程，可用下图表示：
![](raw/figure_19.png?raw=true)

从图中可以看出，该模型没有产生过拟合和欠拟合的问题，基本正确。

### 4.3 测试集预测
```python
lg.fit(x_train_train, y_train_train)

from sklearn.metrics import accuracy_score
print accuracy_score(y_train_val, lg.predict(x_train_val))
```
在测试集上的预测准确率为0.791044776119。


## 五、尚需优化的问题

- name字段中还有很多信息可以利用
- 年龄预测
- 多个模型的测试
