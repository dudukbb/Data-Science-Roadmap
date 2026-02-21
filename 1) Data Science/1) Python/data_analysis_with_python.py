
# data analysis with python

# numpy
# pandas
# veri görselleştirme: matplotlib & seaborn
# gelişmiş fonksiyonel keşifçi veri analizi
# ( advanced functional exploratory data analysis )

#################################################
# NUMPY
#################################################
# numerical python
# Why numpy ?
# Attributes of numpy arrays
# Reshaping
# Index Selection
# Slicing
# Fancy Index
# Conditions on numpy (numpy'da koşullu işlemler)
# Mathematical Operations
import numpy as np

a = [1, 2, 3]
b = [4, 5, 6]

ab = []

for i in range(0,len(a)):
    ab.append(a[i] * b[i])

for i in range(0,len(a)):
    print(ab[i])

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a*b

# create numpy array

np.array([[1, 2, 3], [4, 5, 6]])
type(np.array([1, 2, 3]))

np.zeros(10, dtype= int)
np.random.randint(0,10,8)
np.random.normal(10,4,(3,4))

########################################################
# Attributes of numpy arrays
########################################################
# ndim: boyut sayısı
# shape: boyut bilgisi
# size: toplam eleman sayısı
# dtype: array veri tipi

a = np.random.randint(10,size=5)

#########################################
# Reshaping
#########################################

import numpy as np

np.random.randint(0,10,(9))
np.random.randint(0,10,(9)).reshape(3,3)

ar = np.random.randint(0,10,9)
ar.reshape(3,3)

###########################################
# Index Selection
##########################################

a = np.random.randint(10,size=10)
a[0:5]
a[0] = 999

m = np.random.randint(10, size=(3,5))
m[0,0]

m[:,0]
m[1,:]
m[0:2, 0:3]


##########################################
# Fancy Index
#########################################
import numpy as np
v = np.arange(0,30,3)
v[1]
catch = [1,2,3]
v[catch]

# Conditions on Numpy ( Numpy'da koşullu işlemler )
import numpy as np
v = np.array([1,2,3])
ab =[]

for i in v:
    print(i)
    if i<3:
        ab.append(i)

##################
# Numpy ile
##################

v<3
v[v>3]

# mathematical operations
import numpy as np
v = np.array([1,2,3])
v/5*10
v**2
v-1


v = np.subtract(v,1)
np.add(v,1)
np.mean(v)
np.sum(v)
np.min(v)
np.max(v)
np.var(v)

# numpy ile iki bilinmeyenli denklem çözümü
# 5*x0 + x1 = 12
# x0 + 3*x1 = 10

a = np.array([[5,1],[1,3]])
b = np.array([12,10])

np.linalg.solve(a,b)


# PANDAS

# veri analizi denince en sık kullanılan kütüphanelerdendir
# pandas series
# reading data
# quiğck look at data
# selection in pandas
# aggregation & grouping
# apply ve lambda
# join işlemleri


# pandas series
# tek boyutlu
# index bilgisi barındırır


import pandas as pd

s = pd.Series(['A','B','C','D'])
s.index
s.dtype
s.size
s.ndim
type(s.values)
s.head()
s.tail(3)


# Reading Data

import pandas as pd

df = pd.read_csv("data_analysis_with_python/datasets/advertising.csv")
df.head()

# pandas cheatsheets ara elinde böyle bir cheatsheet bulunsun

# Quick look at data

import seaborn as sns
df = sns.load_dataset("titanic")
df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()    # veri setinde en az bir tane bile olsa eksiklik var mı
                            # df'e

df.isnull().sum()
df["sex"].head()
df["sex"].value_counts()

# Selection in Pandas ( pandasta seçim işlemleri )

df = sns.load_dataset("titanic")
df.head()

df.index
df[0:13]
df = df.drop([1,2,3],axis=0).head()



# Değişkeni Indexe çevirmek
df["age"]

df.index = df["age"]
df.head()

df.index
df["age"] = df.index
df.head()

df.reset_index().head()

# değişkenler üzerinde işlemler
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

"age" in df
type(df["age"].head())
type(df[["age"]].head())
df[["age", "alive"]]
sol_names = ["age","adult_male","alive"]
df[sol_names]


df.head()
df.head()

df.head()

df.loc[:, ~df.columns.str.contains("age")].head()

# iloc & loc
df = sns.load_dataset("titanic")
df.head()

# iloc : index bilgisi vererek seçim yapma işlemlerini ifade eder
# iloc: integer based selection
# loc : indexlerdeki labellara göre selection yapar
# loc : label based selection

# iloc
df.iloc[0:3]
df.iloc[0,0]


# loc
df.loc[0:3]
df.iloc[0:3,0:3]
df.head()
df.loc[0:3,"age2"]
print(df.columns)

df.loc[(df["age2"]>50)
       & (df["sex"] == "male"), ["age","class"]].head()

# Aggregation & Grouping

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

df["age"].mean()

# cinsiyete göre yaş ortalaması
df.groupby("sex")["age"].mean()

df.groupby("sex").agg({"age": "mean"})
df.groupby("sex").agg({"age": [ "mean", "sum"]})
df.groupby("sex").agg({"age": [ "mean", "sum"],
                       "embark_town":"count"})

df.groupby("sex").agg({"age":["mean","sum"],
                       "survived":"mean"})

df.groupby(["sex", "embark_town"]).agg({"age":["mean"],
                       "survived" : "mean"})


df.groupby(["sex", "embark_town","class"]).agg({"age":["mean"],
                       "survived" : "mean",
                    "sex":"count"})


# pivot table
# gorupby gibi
# verinin kırılım noktaları için vs kullnaırız
# Pivot table, veriyi satır (index), sütun (columns)
# ve değer (values) boyutlarında yeniden düzenleyip özet çıkarır.
# Yani bir nevi akıllı grup-by + tablo kombinasyonu gibidir.

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()


# df.pivot_table( kesişim( mean) , satır, sütun)
df.pivot_table("survived", "sex",["embarked","class"])

df.head()

# cut & qcut elinizdeki sayısal değişkeni kategorik değişkene çevirir.
# en yaygın kullanılan iki fonksiyon
# pd.cut : sayısal değişkenleri hangi kategorilere
# bölmek istediğimizi biliyorsak bu
#
# pd.qcut : sayısal değişkenleri tanımayıp çeyreklik olarak
# bölünsün dersek qcut fonksiyonunu kullanırız
# qcut değerleri küçükten büyüğe sıralar ve çeyreklere böler

df["new_age"] = pd.cut(df["age"],[0,10,18,25,40,90])
df.pivot_table("survived", "sex", ["new_age", "class"])
pd.set_option("display.width",500)


# Apply
# satır veya sütunlarda otomatik olarak fonksiyon çalıştırır
# bir df'e apply ile istediğimiz fonksiyonu uygulayabiliriz


#Lambda
# bir fonksiyon tanımlama şeklidir
# kod akışında bir defa kullanayım atayım gibi durumlarda
# fonksiyon tanımlamak yerine lamda kullanırız

import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()


df["age3"] = df["age"]*2
df["age4"] = df["age"]*3
df.head()

(df["age"]/10).head()
(df["age3"]/10).head()
(df["age4"]/10).head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        print((df[col]/10).head())


for col in df.columns:
    if "age" in col:
        df[col] = df[col]/10


df.head()
# değişkenleri seçtik
# apply dedik uygula ( neyi uygula )
# lambda ile tanımlı olan fonksiyonu uygula

df[["age","age3","age4"]].apply(lambda x: x**2).head()

# bir döngü yazmadan apply fonksiyonu bize değişkenlerde gemze imkanı sağladı
# x : değişken
df.loc[:,df.columns.str.contains("age")].apply(lambda x: x/10).head()

df.loc[:,df.columns.str.contains("age")].apply(lambda x: (x - x.mean())/ x.std()).head()

# aplly elimizdeki fonksiyonu satır veya sütunlara uygulama imkanı sunar
def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()

df.loc[:,["age","age3","age4"]] = df.loc[:,df.columns.str.contains("age")].apply(standart_scaler).head()

df.loc[:,df.columns.str.contains("age")] = df.loc[:,df.columns.str.contains("age")].apply(standart_scaler).head()

df.head()

# aplly : fonksiyon uygulamamızı sağlar döngü olmadan,
# lambda : kullan at, tek seferlik fonksiyondur.

###############################################################
# JOIN : birleştirme işlemleri
###############################################################

import numpy as np
import pandas as pd
m = np.random.randint(1,30, size=(5,3))
df1 = pd.DataFrame(m, columns=["var1","var2","var3"])
df2 = df1 + 99

#concat methodu dataframeleri birleştirir
pd.concat([df1, df2],ignore_index=True)


# merge ile birleştirme işlemleri

df1 = pd.DataFrame({"employees":["join", "dennis","mark","maria"],
                    "start_date": [2010,2009,2014,2019]})
df2 = pd.DataFrame({"employees":["join", "dennis","mark","maria"],
                    "group": ["accounting","engineering","engineering","hr"]})

# {} ------> sözlük ( dictionary )
# [] ------> list ( liste )

# on = " " ile hangi argümana göre sütunları merge edeceğimizi söylüyoruz.
pd.merge(df1, df2, on="employees")
df3 = pd.merge(df1, df2)

# amaç : her çalışanın müdürünün bilgisine erişmek istiyoruz

df4 = pd.DataFrame({"group":["accounting","engineering","engineering","hr"],
                    "manager": ["Caner","Mustafa","Berkcan","x"]})

# groupa göre birleştirdik
pd.merge(df3,df4)



# VERİ GÖRSELLEŞTİRME : MATPLOTLİB & SEABORN

# MATPLOTLİB
#low-level veri görselleştirme
# yani seaborn'a kıyasla daha fazla çaba ile veri görselleştirme demek

# categorik değişken : sütun grafik --------> countplot , barplot
# numeric değişken : histogram, boxplot ( aykırı değerleri de gösterir )

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["sex"].value_counts().plot(kind="bar")
plt.show()


# sayısal değişken görselleştirme

plt.hist(df["age"])
plt.show()

# boxplot: kutu grafiği
# veri setindeki aykırı değerleri
# çeyreklik değerler üzerinden gösterebilyor
# verinin kendi içindeki dağılımına bakarak
# genel dağılımın dışındaki gözlemleri yakalamaktır
# boxplottaki amaç ve işaretler aykırı değerleri

plt.boxplot(df["fare"])
plt.show()

# matplotlib'in özellikleri

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

# plot : veriyi görselleştiren fonksiyon

x = np.array([1,8])
y = np.array([0,150])

plt.plot(x,y,"o")
plt.show()

# marker : işaretleyici özelliği
# marker ilgili noktaları işaretler
# plot iki nokta arasını çizgi ile birleştirir

y = np.array([13,28,11,100])
plt.plot(y, marker="*")
plt.show()

# line : çizgi oluşturmamızı sağlar

y = np.array([13,28,11,100])
plt.plot(y,linestyle="--",marker="*",color="red")
plt.show()

# multiple lines

x = np.array([23,18,31,10])
y = np.array([13,28,31,100])
plt.plot(x)
plt.plot(y)
plt.show()

# labels

x = np.array([23,18,31,10])
y = np.array([13,28,31,100])
plt.plot(x,y)
plt.title("Hash Map")
plt.xlabel("Key")
plt.ylabel("Value")
plt.grid()
plt.show()

# subplots : birlikte birden fazla görselin gösterilmesi

x = np.array([23,18,31,10])
y = np.array([13,28,31,100])
 # 1 satırlık 2 sütunluk grafik oluştur ve 1.si ni oluşturduk
plt.subplot(1,2,1)
# 1 satırlık 2 sütunluk 2.grafiği oluştur
plt.subplot(1,2,2)
plt.title("1")
plt.plot(x,y)
plt.show()




# SEABORN : veri görselleştirme kütüphanesi
# high level : daha az çaba ile daha çok iş yapabilmek

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset("tips")
df.head()

# value_counts() : benzersiz (unique) değerlerin
# kaç kez tekrarlandığını (frekansını) gösteren bir metottur.
# Yani bir sütundaki her değerin sayısını çıkarır.
# Kısaca: “Bu sütunda her kategoriden kaç tane var?”
# sorusunun cevabıdır.

# seaborn ile
df["sex"].value_counts()
sns.countplot(x = df["sex"],data = df)
plt.show()

# matplotlib ile
df["sex"].value_counts().plot(kind="bar")
plt.show()

# sayısal değişken görselleştirme
sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()

# veri anlama , değişken anlama için 3 şey var
# 1) value_count & countplot
# 2) hist
# 3) boxplot

######################################################################
# ADVANCED FUNCTİONAL EDA ( gelişmiş fonksiyonel keşifçi veri analizi )
######################################################################
# gelişmiş fonksiyonel keşifçi veri analizi
# AMAÇ : elimize gelen veriyi(küçük ya da büyük) işleyebilmek
# fonksiyonel olarak ölçeklendirebilmek
# hızlı bir şekilde genel gonksiyonlar ile
# elimize gelen veriyi analiz etmek

# 1. Genel Resim
# 2. Analysis of categorical variables  (kategorik değişken analizi)
# 3. Analysis of numerical variables  (sayısal değişken analizi)
# 4. Analysis of target variables  (hedef değişken analizi)
# 5. Analysis of correlation  (korelasyon analizi)

# 1) veri setinin iç ve dış özelliklerinin
# genel hatları hakkında edinmek ( type , eksik value vs)


# 1) GENEL RESİM

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T   # sayısal değişkenleri betimleme
df.isnull().values.any()  # ekisk değer var mı yok mu
df.isnull().sum() # veri setindeki bütün değişkenlerdeki eksik değer sayısını verir.
                  # kaçar tane eksik


def check_df(df,head=5):
    print("###########Shape########")
    print(df.shape)
    print("###############Dtypes#################")
    print(df.dtypes)
    print("###############Head#################")
    print(df.head(head))
    print("###############Tail#################")
    print(df.tail(head))
    print("###############NA#######################")
    print(df.isnull().sum())
    print("##########Quantifies##########")
    print(df.describe([0,0.05,0.50,0.95,0.99,1]).T)

check_df(df)

df= sns.load_dataset("flights")
check_df(df)

df.head()
# Kategorik Değişken Analizi

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df["sex"].unique()       # bir sütundaki bensersiz değerleri listeler
df["class"].nunique()    # benzersiz değerlerin sayısını ( kaç tane oldugunu ? )  verir
                        # Yani: “Bu sütunda kaç farklı değer var?” sorusuna yanıt verir.




# Categorical Analysis

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()


df["survived"].value_counts()
df["sex"].unique()
df["class"].nunique()

#comprenhension
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category","bool"]]

num_but_cat = [ col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int","float"] ]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["int","float"] ]

cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]
df[cat_cols].nunique()
[ col for col in df.columns if col not in cat_cols]


def cat_summary(df,col_name,plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100*df[col_name].value_counts() / len(df)}))
    print("#####################################################")

for col in cat_cols:
    cat_summary(df,col)


    if plot:
        sns.countplot(x=df[col_name],data = df)
        plt.show(block = True)

cat_summary(df,"sex",plot=True)


for col in cat_cols:
    if df[col].dtypes == "bool":
        print("dkfvbdfkbdsk")
    else:
        cat_summary(df,col,plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df,col,plot=True)

# Sayısal Değişken Analizi
# analysis of numerical variables

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df[["age","fare"]].describe().T
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category","bool"]]

num_but_cat = [ col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int","float"] ]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["int","float"] ]

cat_cols = cat_cols + num_but_cat
num_cols = [ col for col in df.columns if df[col].dtypes in ["int","float"] ]
num_but_cat = [col for col in num_cols if col not in cat_cols]

def num_summary(df,num_col):
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    print(df[num_col].describe(quantiles).T)

num_summary(df,"age")

for col in num_cols:
    num_summary(df,col)

def num_summary(df,num_col, plot=False):
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    print(df[num_col].describe(quantiles).T)

num_summary(df,"age")

for col in num_cols:
    num_summary(df,col)

def num_summary(df,num_col, plot=False):
    quantiles = [0.5, 0.75, 0.9, 0.95, 0.99]
    print(df[num_col].describe(quantiles).T)

    if plot:
        df[num_col].hist()
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)

num_summary(df,"age",plot=True)

for col in num_cols:
    num_summary(df,col,plot=True)

# değişkenlerin yakalanması ve işlemlerin genelleştirilmesi

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

def grab_col_names(df, cat_th = 10, car_th=30):
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]

    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["int", "float"]]

    cat_cols = cat_cols+num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in df.columns if df[col].dtypes in ["int","float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]


    print(f"Observations: {df.shape[0]}")
    print(f"Variables: {df.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'cat_cols: {len(cat_cols)}')

    return cat_cols, num_cols, cat_but_car



cat_cols, num_cols, cat_but_car = grab_col_names(df)

print(cat_cols)
print(num_cols)
print(cat_but_car)

for col in cat_cols:
    cat_summary(df,col)

for col in num_cols:
    num_summary(df,col,plot=True)

# bool tipteki değişkenleri int yapmak , catsummaryi görsel özelliği eklşeekl

df = sns.load_dataset("titanic")
df.info()
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# hedef değişken analizi
# analysis of target variable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)



def cat_summary(df,col_name,plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100*df[col_name].value_counts() / len(df)}))
    print("#####################################################")

def grab_col_names(df,cat_th = 10, car_th=20):

 cat_cols, num_cols, cat_but_car = grab_col_names(df)

 df.head()

df["survived"].value_counts()
cat_summary(df,"survived")

# hedef değişkenin kategorik değişkenler ile analizi

df.groupby("sex")["survived"].mean()

def target_summary_with_cat(df,target,categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean()}))


target_summary_with_cat(df,"survived","sex")

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]

num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["int", "float"]]

cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
num_cols = [col for col in num_cols if col not in cat_cols]

for col in cat_cols:
    target_summary_with_cat(df,"survived",col)


#   Hedef değişkenin sayısal değişkenler ile analizi

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age":"mean"})


def target_summary_with_num(df,target,numerical_col):
    print(df.groupby(target).agg({numerical_col: "mean"}),end = "\n\n\n")

target_summary_with_num(df,"survived","age")

for col in num_cols:
    target_summary_with_num(df,"survived",col)


# korelasyon analizi
# analysis of correlation

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv(r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\1.Hafta - Python I\data_analysis_with_python\datasets\breast_cancer.csv")
df = df.iloc[:,1:-1]
df.head()

num_cols = [col for col in df.columns if df[col].dtype in [int, float]]
corr = df[num_cols].corr()

# correlation
# değişkenlerin birbiri ilşe ilişkisini ifade eder
# -1 ve +1 arasında değerler alır
# -1 ya da +1'e yaklaştıkça ilişki güçlenir
# ilişki pozitifse pozitif korelasyon ( bir artarken diğeri artar )
# ilişki negatifse negatif korelasyon ( biri artarken diğeri azalır )
# o demek korelasyon yok demektir

sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

# yüksek korelasyonlu değişkenlerin silinmesi

cor_matrix = df.corr().abs()

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k =1).astype(np.bool))
drop_list = [ col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90) ]
cor_matrix[drop_list]
df.drop(drop_list, axis=1)
