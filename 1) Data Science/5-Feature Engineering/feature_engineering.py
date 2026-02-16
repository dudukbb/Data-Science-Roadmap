#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def load_application_train():
    data = pd.read_csv("6.Hafta - Feature Engineering/datasets/application_train.csv")
    return data

df = load_application_train()
df.head()


def load():
    data = pd.read_csv("6.Hafta - Feature Engineering/datasets/titanic.csv")
    return data


df = load()
df.head()


#############################################
# 1. Outliers (Aykırı Değerler)
#############################################

#############################################
# Aykırı Değerleri Yakalama
#############################################

###################
# Grafik Teknikle Aykırı Değerler
###################

sns.boxplot(x=df["Age"])
plt.show()

###################
# Aykırı Değerler Nasıl Yakalanır?
###################

q1 = df["Age"].quantile(0.25)
q1
q3 = df["Age"].quantile(0.75)
q3
iqr = q3 - q1
iqr
up = q3 + 1.5 * iqr
up
low = q1 - 1.5 * iqr
low

df[(df["Age"] < low) | (df["Age"] > up)]

df[(df["Age"] < low) | (df["Age"] > up)].index

###################
# Aykırı Değer Var mı Yok mu?
###################
# aykırı değer var mı ? varsa = True
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[(df["Age"] < low)].any(axis=None)

# 1. Eşik değer belirledik.
# 2. Aykırılara eriştik.
# 3. Hızlıca aykırı değer var mı yok diye sorduk.

###################
# İşlemleri Fonksiyonlaştırmak
###################

# outlierlar için threshold belirliyoruz
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")
low
up

df[(df["Fare"] < low) | (df["Fare"] > up)].head() # outlierlar

df[(df["Fare"] < low) | (df["Fare"] > up)].index # outlierların indexleri

# outlier var mı yok mu ?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")

###################
# grab_col_names
###################

dff = load_application_train()
dff.head()

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# PassengerIdyi dıoşarda bırak num_colsda
num_cols = [col for col in num_cols if col not in "PassengerId"]

# numeric colonlarda gez, outlier var mı bak
for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

# SK_ID_CURR num_cols dışında bıraktık
num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

# dff colonlarında outlier var mı bakıyoruz
for col in num_cols:
    print(col, check_outlier(dff, col))

###################
# Aykırı Değerlerin Kendilerine Erişmek
###################
# shape(gözlem sayısı,feature sayısı)
# shape(0,...) = gözlem sayısı
# shape(...,1) = feature sayısı
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

# 10’dan fazla outlier varsa sadece ilk 5 tanesini göster
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
# 10'dan daha az outlier varsa hepsini göster
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])
# index=True ise yani outlierların indexlerini istiyorsak outlier_index listinde tut ve return et
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")

# index=True olduğu için indexleri de gelecek
grab_outliers(df, "Age", True)

# outlier indexlerini age_index'te saklıyoruz
age_index = grab_outliers(df, "Age", True)


outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", True)

#############################################
# Aykırı Değer Problemini Çözme
#############################################

###################
# Silme
###################

low, up = outlier_thresholds(df, "Fare")
low
up
df.shape  # kaç gözlem var

# outlier olmayan gözlemlerin shapei
# yani outlierları silersem ne kadar gözlem kalacak
df[~((df["Fare"] < low) | (df["Fare"] > up))].shape

# outlierları siler
def remove_outlier(dataframe, col_name):
    # alt ve üst limitleri getir
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

# gözlem sayısı
df.shape

# outlierları sildik
for col in num_cols:
    new_df = remove_outlier(df, col)

# outlierların sayısı
df.shape[0] - new_df.shape[0]

###################
# Baskılama Yöntemi (re-assignment with thresholds)
###################

low, up = outlier_thresholds(df, "Fare")

# outlierlarlar
df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

# loc = satır ve sütunlarda filtreleme yapar
df.loc[((df["Fare"] < low) | (df["Fare"] > up)), "Fare"]

# üst limite göre outlier olanlar ve üst limit
# üst limitten fazla olanları yani outlierları üst limit ile güncelle
df.loc[(df["Fare"] > up), "Fare"] = up

# alt limite göre outlier olanlar ve alt limit
# alt limitten daha az olanları yani  outlierları alt limit ile güncelle
df.loc[(df["Fare"] < low), "Fare"] = low

# outlier thresholdlara göre alt limitten az ve üst limitten fazla olanları
# yani outlierları alt limit ve üst limitle güncelle
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

# outlier var mı ?
for col in num_cols:
    print(col, check_outlier(df, col))

# outlierları baskıladık
for col in num_cols:
    replace_with_thresholds(df, col)

# outlier var mı ?
for col in num_cols:
    print(col, check_outlier(df, col))


###################
# Recap
###################

df = load()
outlier_thresholds(df, "Age")   # outlier thresholdlar
check_outlier(df, "Age")    # outlier var mı ?
grab_outliers(df, "Age", index=True)  # outlierları(index) yakala

remove_outlier(df, "Age").shape   # outlierları sildik
replace_with_thresholds(df, "Age")     # thresholdlarla outlierları alt ve üst limitle güncelledik
check_outlier(df, "Age")   # outlier var mı ? YOK, outlierlar temizlendi


#############################################
# Çok Değişkenli Aykırı Değer Analizi: Local Outlier Factor
#############################################
# age = 17
# marriage = 3
# yaşı 17 olup 3 defa evlenmek aykırı bir durumdur
# yani tek başına 17 yaş ya da 3 evlilik aykırı değilken birlikte
# yani çok değişkenli iken aykırı olabilir
# 17, 3
# local outlier factor = lof

df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()
df.shape

for col in df.columns:
    print(col, check_outlier(df, col))


low, up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low, up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape

# lof yöntemi
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)   # lof skorlarını getirir.

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# lof skorları - onu +'e çeviririyoruz
# df_scores = -df_scores

# skorları sırala
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# threshold = sıralanmış skorlarda 3.olan
th = np.sort(df_scores)[3]
th

df[df_scores < th]

df[df_scores < th].shape

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index

# inplace = True argümanını ekleseydik kalıcı drop ederdik
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

#############################################
# Missing Values (Eksik Değerler)
#############################################

#############################################
# Eksik Değerlerin Yakalanması
#############################################

df = load()
df.head()

# eksik gozlem var mı yok mu
df.isnull().values.any()

# degiskenlerdeki eksik deger sayisi
df.isnull().sum()

# degiskenlerdeki tam deger sayisi
df.notnull().sum()

# veri setindeki toplam eksik deger sayisi
df.isnull().sum().sum()

# en az bir tane eksik degere sahip olan gözlem birimleri
df[df.isnull().any(axis=1)]

# tam olan gözlem birimleri
df[df.notnull().all(axis=1)]

# Azalan şekilde missing valuları sıralamak
df.isnull().sum().sort_values(ascending=False)

# df.isnull().sum() = datadaki missing value frequency
# frequencyi / toplam gözlem sayısı * 100 = yüzdelik olarak nissing value oranları
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

# sadece missing valueya dahip featureları seçelim
# df.isnull().sum() = eksik değer sayısı
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
na_cols

# na_name = eksik değerleri olan featureların isimleri
# na_name parametresini True olarak verirsek
# fonksiyonu çağırdığımızda na_nameleri de alırız
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # eksik değer sayısı
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    # eksik değer oranı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # missing df
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns


missing_values_table(df)

missing_values_table(df, True)


#############################################
# Eksik Değer Problemini Çözme
#############################################

missing_values_table(df)

###################
# Çözüm 1: Hızlıca silmek
###################

# 1 satırda 1 tane bile eksik değer varsa siliyor
df.dropna().shape

###################
# Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
###################

# fillna
# avg, median ve herhangi bir değerle eksik değerleri dolduruyoruz
df["Age"].fillna(df["Age"].mean()).isnull().sum() # 0 demek eksik değer yok hallettik demek
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

# datadaki tüm featurelar için eksik değerleri doldurma
# apply = satırlarda veya sütunlarda gez (axis=0 olduğu için satır)
# lambda = tek satırlık kullan at fonksiyon
df.apply(lambda x: x.fillna(x.mean()), axis=0)

# featureun tipi Object değilse numeric ise mean ile eksik değerleri doldur
df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum().sort_values(ascending=False)

# # Categoric değişkenlerde mod ile doldururuz. Stringin 0.indexindeki ile eksik değerleri dolduruyoruz
df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()

# String bir missing ifadesi ile Embarked değişkenindeki eksik valuları doldur
df["Embarked"].fillna("missing")

# koşul sağlanıyorsa bir şey yap sağlanmıyorsa yapma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()

###################
# Kategorik Değişken Kırılımında Değer Atama
###################

# cinsiyete göre groupby alıp meanlerine bak
df.groupby("Sex")["Age"].mean()

df["Age"].mean()

# agei cinsiyetlerin avglarına göre doldur
# dfi sexe göre groupby al
# transform = bunu mean ile değiştir
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.groupby("Sex")["Age"].mean()["female"]

# kadınlardaki eksik değerleri kadınların meani ile doldurduk
df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]

df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################################
# Çözüm 3: Tahmine Dayalı Atama ile Doldurma
#############################################
# 1) eksik değeri silme
# 2) eksik değeri mean median vs ile doldurma
# 3) gelişmiş ml teknikleri ile çözme

# eksik değere sahip olan değişken = bağımlı değişken
# eksik değere sahip olmayan değişkene = bağımsız değişken
# kabul ederek modelleme gerçekleştirecegiz

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]
# 2 sınıfa sahip olan ilk sınıfını atacak , ikincisini atacak
# yani female male gibi şeyleri binary encode edilmiş şekilde olacak
# 2 sınıf veya daha fazla sınıfa sahip olan categoric değişkenleri numeric olarak ifade etmek
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dff.head()

# değişkenlerin standartlaştırılması = min max scaler
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()


# knn'in uygulanması.
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)

df["age_imputed_knn"] = dff[["Age"]]

df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]
df.loc[df["Age"].isnull()]


###################
# Recap
###################

df = load()
# missing table
missing_values_table(df)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma


#############################################
# Gelişmiş Analizler
#############################################

###################
# Eksik Veri Yapısının İncelenmesi
###################

# datadaki tam olan gözlemlerin sayısı
msno.bar(df)
plt.show()

# değişkenlerdeki eksikliklikler bir arada mı ortaya çıktı?
msno.matrix(df)
plt.show()

# korelasyon +1 ile -1 arasında değerler alır
# eksik değerler birlikte çıkmış olabilir 2 featureda
# ya da biri diğerinde eksik değer çıkmasına sebep oluyor olabilir
# nullity correlation değerlerini
# eksik değerlerin rassallığı
msno.heatmap(df)
plt.show()

###################
# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
###################

missing_values_table(df, True)
na_cols = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


# Survived = bağımlı değişken
# eksik değer(NaN)'e sahip olan değişkenlerin, targtet'ı nasıl etkilediğini analiz eder
missing_vs_target(df, "Survived", na_cols)


###################
# Recap
###################

df = load()
na_cols = missing_values_table(df, True)
# sayısal değişkenleri direk median ile oldurma
df.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0).isnull().sum()
# kategorik değişkenleri mode ile doldurma
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()
# kategorik değişken kırılımında sayısal değişkenleri doldurmak
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# Tahmine Dayalı Atama ile Doldurma
missing_vs_target(df, "Survived", na_cols)


#############################################
# 3. Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

#############################################
# Label Encoding & Binary Encoding
#############################################

df = load()
df.head()
df["Sex"].head()

le = LabelEncoder()
le.fit_transform(df["Sex"])[0:5]  # label encode ettik
le.inverse_transform([0, 1]) # 0 = female, 1 = male

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

df = load()

# eşsiz sınıf sayısı 2 olan
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

binary_cols

for col in binary_cols:
    label_encoder(df, col)

df.head()

df = load_application_train()
df.shape

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

binary_cols

df[binary_cols].head()

for col in binary_cols:
    label_encoder(df, col)  # label encoder NaN'lere de bir label atadı


df = load()
df["Embarked"].value_counts()  # bir değişkende her değerin kaç kez geçtiğini sayar
df["Embarked"].nunique()
len(df["Embarked"].unique()) # NaN'ı da categori olarak algılıyor, label atıyor

#############################################
# One-Hot Encoding
#############################################

df = load()
df.head()
df["Embarked"].value_counts()

# get_dummies = one-hot encoding yapar
pd.get_dummies(df, columns=["Embarked"]).head()

# ilk sınıf alfabetik sıraya göre drop edildi
pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()

# Embarked içindeki eksik değerler(NaN) için de bir sınıf oluşturur.
pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()

# Sex ve Embarked featurelarına opne hot encoding uygula
pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True).head()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = load()

# cat_cols, num_cols, cat_but_car = grab_col_names(df)

# one hot encidong yapacagımız kolonları seçiyoruz
# unique number sayısı 2 ile 10 arasında olanlara one hot encoding uygula
# Sex'e daha öncesinde label encoding uyguladık ona ve targeta dokunmuyoruz
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head() # kalıcı değişiklik istersek df'e atamamız gerekir

df.head()

#############################################
# Rare Encoding
#############################################

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
# 3. Rare encoder yazacağız.

###################################################################
# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
###################################################################

df = load_application_train()
df["NAME_EDUCATION_TYPE"].value_counts()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

###################
# 2. Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.
###################

df["NAME_INCOME_TYPE"].value_counts()

df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "TARGET", cat_cols)

#############################################
# 3. Rare encoder'ın yazılması.
#############################################

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(df, 0.01)

rare_analyser(new_df, "TARGET", cat_cols)

df["OCCUPATION_TYPE"].value_counts()


#############################################
# Feature Scaling (Özellik Ölçeklendirme)
#############################################

###################
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
###################

df = load()
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])
df.head()


###################
# RobustScaler: Medyanı çıkar iqr'a böl.
###################

rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])
df.describe().T

###################
# MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü
###################

# X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
# X_scaled = X_std * (max - min) + min

mms = MinMaxScaler()
df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

df.head()

age_cols = [col for col in df.columns if "Age" in col]

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in age_cols:
    num_summary(df, col, plot=True)

###################
# Numeric to Categorical: Sayısal Değişkenleri Kateorik Değişkenlere Çevirme
# Binning
###################

df["Age_qcut"] = pd.qcut(df['Age'], 5)

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

#############################################
# Binary Features: Flag, Bool, True-False
#############################################

df = load()
df.head()

df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')

df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})


from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                            df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SibSp'] + df['Parch']) == 0), "NEW_IS_ALONE"] = "YES"

df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})


test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                             df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],

                                      nobs=[df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                            df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#############################################
# Text'ler Üzerinden Özellik Türetmek
#############################################

df.head()

###################
# Letter Count
###################

df["NEW_NAME_COUNT"] = df["Name"].str.len()

###################
# Word Count
###################

df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))

###################
# Özel Yapıları Yakalamak
###################

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean","count"]})

###################
# Regex ile Değişken Türetmek
###################

df.head()

df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)


df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

#############################################
# Date Değişkenleri Üretmek
#############################################

dff = pd.read_csv("datasets/course_reviews.csv")
dff.head()
dff.info()

dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")

# year
dff['year'] = dff['Timestamp'].dt.year

# month
dff['month'] = dff['Timestamp'].dt.month

# year diff
dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year

# month diff (iki tarih arasındaki ay farkı): yıl farkı + ay farkı
dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month


# day name
dff['day_name'] = dff['Timestamp'].dt.day_name()

dff.head()

# date


#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################
df = load()
df.head()

df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]

df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1

df.loc[(df['SEX'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'

df.loc[(df['SEX'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'

df.loc[(df['SEX'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'

df.loc[(df['SEX'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'

df.loc[(df['SEX'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df.head()

df.groupby("NEW_SEX_CAT")["Survived"].mean()


#############################################
# Titanic Uçtan Uca Feature Engineering & Data Preprocessing
#############################################

df = load()
df.shape
df.head()

# featureların isimlerinin hepsini büyült
df.columns = [col.upper() for col in df.columns]

#############################################
# 1. Feature Engineering (Değişken Mühendisliği)
#############################################

# Cabin bool
df["NEW_CABIN_BOOL"] = df["CABIN"].notnull().astype('int')
# Name count
df["NEW_NAME_COUNT"] = df["NAME"].str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df['NEW_TITLE'] = df.NAME.str.extract(r' ([A-Za-z]+)\.', expand=False)
# family size
df["NEW_FAMILY_SIZE"] = df["SIBSP"] + df["PARCH"] + 1
# age_pclass
df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]
# is alone
df.loc[((df['SIBSP'] + df['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df['SIBSP'] + df['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age
df.loc[(df['SEX'] == 'male') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['SEX'] == 'male') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] > 21) & (df['AGE'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['SEX'] == 'female') & (df['AGE'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
num_cols
cat_but_car

# PassengerId'yi num_cols'tan çıkarıcaz
num_cols = [col for col in num_cols if "PASSENGERID" not in col]
num_cols
#############################################
# 2. Outliers (Aykırı Değerler)
#############################################

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################################
# 3. Missing Values (Eksik Değerler)
#############################################

missing_values_table(df)

df.drop("CABIN", inplace=True, axis=1)

remove_cols = ["TICKET", "NAME"]
df.drop(remove_cols, inplace=True, axis=1)


# na leri median ile değiştircez
df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))


df["NEW_AGE_PCLASS"] = df["AGE"] * df["PCLASS"]

df.loc[(df['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
df.loc[(df['AGE'] >= 18) & (df['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'


df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

#############################################
# 4. Label Encoding
#############################################

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#############################################
# 5. Rare Encoding
#############################################

rare_analyser(df, "SURVIVED", cat_cols)


df = rare_encoder(df, 0.01)

df["NEW_TITLE"].value_counts()

#############################################
# 6. One-Hot Encoding
#############################################

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", cat_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

# df.drop(useless_cols, axis=1, inplace=True)

#############################################
# 7. Standart Scaler
#############################################

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape


#############################################
# 8. Model
#############################################

y = df["SURVIVED"]
X = df.drop(["PASSENGERID", "SURVIVED"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#############################################
# Hiç bir işlem yapılmadan elde edilecek skor?
#############################################

dff = load()
dff.dropna(inplace=True)
dff = pd.get_dummies(dff, columns=["Sex", "Embarked"], drop_first=True)
y = dff["Survived"]
X = dff.drop(["PassengerId", "Survived", "Name", "Ticket", "Cabin"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

# Yeni ürettiğimiz değişkenler ne alemde?

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(rf_model, X_train)


