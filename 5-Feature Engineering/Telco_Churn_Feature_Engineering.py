
"""
Görev 1 : Keşifçi Veri Analizi
Adım 1: Genel resmi inceleyiniz.
Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
numerik değişkenlerin ortalaması)
Adım 5: Aykırı gözlem analizi yapınız.
Adım 6: Eksik gözlem analizi yapınız.
Adım 7: Korelasyon analizi yapınız.

Görev 2 : Feature Engineering
Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
Adım 2: Yeni değişkenler oluşturunuz.
Adım 3: Encoding işlemlerini gerçekleştiriniz.
Adım 4: Numerik değişkenler için standartlaştırma yapınız.
Adım 5: Model oluşturunuz

"""

# Gerekli Kütüphane ve Fonksiyonlar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)   # tüm kolonları göster
pd.set_option('display.width', None)    # satır genişliği
pd.set_option('display.max_rows', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("6.Hafta - Feature Engineering/datasets/Telco-Customer-Churn.csv")
df.head()
df.columns

##############################################
# Görev 1 : Keşifçi Veri Analizi
##############################################

##############################################
# Adım 1: Genel resmi inceleyiniz.
##############################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.select_dtypes(include=["int64","float64"]).quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)

check_df(df)

df.info()     # veri setinin yapısını hızlıca görmek için

#############################################################
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
#############################################################

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
        car_th: int, optional
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

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat     # gerçek cat_cols
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # categoric görünümlü cardinali de çıkartıyorum

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]   # numeric görünümlü categoricleri numeric içerisinden çıkartmamız gerek

    # ( observation , feature )
    print(f"Observations: {dataframe.shape[0]}")    # ( 0 , ..... )
    print(f"Variables: {dataframe.shape[1]}")       # ( ....., 1 )
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

# ilk 3 'ünün toplamı feature sayısı
# num_but_cat, cat_cols'un içerisinde
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.dtypes

# TotalCharges str gözüküyor.
# coerce = sayı varsa numeric yapar, sayı olmayan bir şey varsa boşluk gibi NaN yapar.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors = "coerce")

# kolonların içinden customerID kolonunu sil
df.drop("customerID",axis=1, inplace = True )
cat_cols, num_cols, cat_but_car = grab_col_names(df)

df.dtypes

num_cols
cat_cols
cat_but_car


##############################################################
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
##############################################################

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

def num_summary(dataframe, num_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    print(dataframe[num_cols].describe(quantiles).T)

    if plot:
        plt.figure(figsize=(6,4))
        dataframe[num_cols].hist(bins=20)
        plt.xlabel(num_cols)
        plt.title(num_cols)
        plt.show()

for col in num_cols:
    num_summary(df,col,plot=True)

# num_summary(df,"Insulin",plot=True)

##################################
# CATEGORIC DEĞİŞKENLERİN ANALİZİ
##################################

# plot=False grafik çizmiycek
# value_counts = bir değişkende her değerden kaç tane var ----> frequency table
def cat_summary(df,col_name,plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))
    print("##########################################")
    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show()


cat_summary(df, "Churn")

# for col in cat_cols:
# cat_summary(df, col)

##################################
# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef değişkene göre
# numerik değişkenlerin ortalaması)
##################################

##################################
# TARGETA GÖRE NUMERIC DEĞİŞKENLERİN MEANİ
##################################
# Amaç = Bu değişkenler bir kişinin diyabet olup olmadığını tahmin etmeye yardımcı mı ?
# Bunu öğrenmek amaç
def target_summary_with_num(df,target,num_col):
    print(df.groupby(target).agg({num_col: "mean"}),end = "\n\n\n")

for col in num_cols:
    target_summary_with_num(df,"Churn",col)

##########################################
# Adım 5: Eksik gözlem analizi yapınız.
##########################################

df.isnull().sum()
df.shape
df.isnull().sum()
# TotalCharges'da 11 eksik var, toplam 7000 gözlem var bu yüzden TotalCharges'daki NaN'ları siliyoruz.
df = df[df["TotalCharges"].notna()]
df.isnull().sum()

########################################
# Adım 6: Aykırı gözlem analizi yapınız.
########################################

# outlierlar için threshold belirliyoruz
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# outlier var mı ?
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# outlierları yakalayan fonksiyon
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

for col in num_cols:
    print(col,outlier_thresholds(df,col))

for col in num_cols:
    print(col,check_outlier(df,col))    # outlier yokmuş bu datasette


###########################################
# Adım 7: Korelasyon analizi yapınız.
###########################################

# Korelasyon = iki değişken arasındaki ilişki

plt.figure(figsize=(18,13))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="magma")

plt.title("Correlation Matrix", fontsize=15)
plt.tight_layout()
plt.show()

# avg_charge = TotalCharges / tenure
df["avg_per_month"] = (df["TotalCharges"] / df["tenure"] + 0.001)
df.head()
###############################
# Görev 2 : Feature Engineering
###############################
###################################################################
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
###################################################################

# eksik değer silindi az oldugundan
# aykırı gözlem yok

################################################
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
################################################
from sklearn.preprocessing import LabelEncoder

# Binary Encoding ( Label Encoding )
df.head()
binary_cols = [col for col in df.columns if df[col].nunique()==2]
binary_cols
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])
df[binary_cols].head(5)


# One-Hot-Encoding
ohe_cols = [col for col in cat_cols if 10 >= df[col].nunique() > 2 and col != "Churn"]
df = pd.get_dummies(df,columns=ohe_cols,drop_first=True)
ohe_cols
df.head()
df.dtypes
df.select_dtypes(include="object").columns

df.shape

############################################################
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
############################################################
# Standart Scaler = her numeric değişkeni
# avg = 0, std = 1 olacak şekilde dönüştürür.
# model için tüm değişkenler eşit önemli olsun diye yaptık.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape

###########################
# Adım 5: Model oluşturunuz
###########################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# customerID'yi zaten başlarda drop etmiştik
y = df["Churn"]
X = df.drop(["Churn"],axis = 1)

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=17)
# random forest modelini oluşturur ve train verisiyle eğitilir
# çok sayıda decision tree var ve avg'ı alınır
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)   #
accuracy_score(y_pred, y_test)

# X_train = modelin öğrendiği müşteri özellikleri (input)
# y_train = modelin öğrendiği output ----> Bu özelliklere sahip müşteri Churn etmiş mi ?

# X_test =  yeni müşteri (modelin hiç görmediği)
# y_test =  gerçek sonuç
# y_pred = modelin tahmini


"""
model öğrenir → X_train + y_train
tahmin yapar → X_test
çıktı üretir → y_pred
karşılaştırılır → y_test ile

ÖRNEK:
y_test = gerçek cevap anahtarı
y_pred = öğrencinin işaretlediği cevap

"""
################################################
# EK OLARAK
###############################################

##############################
# Feature Importance
##############################

feature_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance", ascending=False)

print(feature_imp.head(10))

plt.figure(figsize=(8,6))
sns.barplot(x="Importance", y="Feature", data=feature_imp.head(10))
plt.title("Feature Importance")
plt.show()


####################################
# Confusion Matrix
####################################
#               Actual P       Actual N
# Pred P          TP             FP
# Pred N          FN             TN

# TN = Churn etmeyecek müşteriyi doğru bulmuş
# TP = Churn edecek müşteriyi doğru bulmuş
# FP = model Churn olmayana Churn dedi, yanlış yakaladı
# FN = model Churn olana Churn değil dedi, Churn'ü yakalayamadı.

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show()


######################################
# Metrics
######################################

# precision = TP / (TP+FP)
# recall = TP / (TP+FN)
# f1 score = 2*(precision * recall) / (precision + recall)
from sklearn.metrics import classification_report

# y_pred (modelin tahmini) ile y_test(gerçek değerleri) karşılaştırır
# ve başarı raporu üretir
print(classification_report(y_test, y_pred))  

