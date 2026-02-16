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
Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama Glikoz, Insulin vb.
değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir. Örneğin; bir kişinin glikoz veya insulin değeri 0
olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
değerlere işlemleri uygulayabilirsiniz.
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

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("6.Hafta - Feature Engineering/datasets/diabetes.csv")
df.head()

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
    print(dataframe.quantile([0, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 1]).T)

check_df(df)

df.info()

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

df.head()

# num_but_cat, cat_cols'un içinde

df.dtypes

cat_cols
num_cols
cat_but_car

##############################################################
# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
##############################################################

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


cat_summary(df, "Outcome")

# for col in cat_cols:
# cat_summary(df, col)

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
    target_summary_with_num(df,"Outcome",col)

######################################
# TARGETA GÖRE CATEGORİC DEĞİŞKENLER
######################################
def cat_summary_with_target(df, target, col_name):
    print(f"######## {col_name} ########")
    print(pd.DataFrame({
        "Target_Mean": df.groupby(col_name)[target].mean(),
        "Count": df[col_name].value_counts(),
        "Ratio": 100 * df[col_name].value_counts() / len(df)
    }))
    print("\n")

for col in cat_cols:
    cat_summary_with_target(df,"Outcome",col)

# cat_summary_with_target(df, "Outcome", "Pregnancies")

##########################################
# Adım 5: Eksik gözlem analizi yapınız.
##########################################

# Glucose
# BloodPressure
# SkinThickness
# Insulin
# BMI
# Bu değişkenlerde 0 demek canlı değil demek
# Yani bu değişkenlerde 0 değeri fizyolojik olarak imkansız ----> yani eksik değeri temsil ediyor
# Bu yüzden 0 olan değerleri NaN ile değiştiriyoruz.

df.isnull().sum()

# tüm columlarda gez
# ve her kolonda kaç tane 0 var say
for col in df.columns:
    print(col, (df[col] == 0).sum())

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
    print(col, outlier_thresholds(df,col))

for col in num_cols:
    print(col, check_outlier(df,col))

for col in num_cols:
    print(col, grab_outliers(df,col))


###########################################
# Adım 7: Korelasyon analizi yapınız.
###########################################

# Korelasyon = iki değişken arasındaki ilişki
# Correlation Matrix
plt.figure(figsize=(18,13))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="magma")

plt.title("Correlation Matrix", fontsize=15)
plt.tight_layout()
plt.show()

###############################
# Görev 2 : Feature Engineering
###############################

###################################################################
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
###################################################################

# NaN = eksik değer
# eksik değer rastgele mi ?
# 1) eksik değer rastgele ise -----> silebiliriz, mean/median ile doldurabiliriz
# 2) eksik değer rastgele değil ise ----->

df.isnull().sum()

# Örneğin; bir kişinin glikoz veya insulin değeri 0
# olamayacaktır. Bu durumu dikkate alarak sıfır değerlerini ilgili değerlerde NaN olarak atama yapıp sonrasında eksik
# değerlere işlemleri uygulayabilirsiniz.

df.head()
cols = [ "Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
df[cols] = df[cols].replace(0,np.nan)
df.isnull().sum()  # eksik değerler var = NaN

# 0 olanları NaN yaptıktan sonra ne ile dolduracağıma karar vermek için
# mean ve median'ına bakıyorum her bir feature'un.
# Eğer median ve mean yakınsa birbirine, mean kullan
# median ve mean yakın değilse median kullan
df.mean()
df.describe().T[["mean","50%"]]  # mean ve medianı yanyana görüp yakınlar mı birbirlerine bakacağız

# her feature'u kendi median'ı ile dolduruyoruz
for col in cols:
    df[col] = df[col].fillna(df[col].median())

df.isnull().sum() # eksik değerleri (NaN) doldurduk

############################
# OUTLIERLARI BASKILADIK
############################
df[num_cols] = df[num_cols].astype(float)

check_outlier(df,col) # outlier var mı ?

# outlier thresholdlara göre alt limitten az ve üst limitten fazla olanları
# yani outlierları alt limit ve üst limitle güncelle
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

outlier_thresholds(df,col)

for col in num_cols:
    replace_with_thresholds(df,col)

check_outlier(df,col)
df.describe().T

##############################
# YENİ DEĞİŞKENLER OLUŞTURUNUZ
##############################

df["BMI_Category"] = pd.cut(df["BMI"],
                            bins = [0,18.5,25,30,100],
                            labels = ["Underweight","Normal","Overweight","Obese"])
df.head()

df["Age_Category"] = pd.cut(df["Age"],
                            bins=[18,30,45,60,100],
                            labels = ["Young","Adult","Middle_Age","Senior"])

df.head()

df["Glucose_Level"] = pd.cut(df["Glucose"],
                             bins = [0,100,125,300],
                             labels = ["Normal","Prediabetes","Diabetes"])

df.head()

# Glucose yüksek , Insulin düşük -----> risk
# Glucose yüksek , Insulin yüksek -----> insulin direnci
# İkisi de normal -----> sağlıklı

df["Insulin_Glucose_Ratio"] = df["Insulin"] / df["Glucose"]
df.head()

df["Insulin_Glucose_Ratio_Category"] = pd.qcut(
    df["Insulin_Glucose_Ratio"],
    q = 4,
    labels = ["Very_low","Low","High","Very_High"])

df.head()

################################################
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
################################################
# Encoding = categoric veriyi sayısala çevirme modele girebilsin diye

from sklearn.preprocessing import LabelEncoder

##### Binary Encoding ( Label Encoding )  ----> 2 sınıf varsa, derece yoksa
df.head()
# 2 sınıf olan sadece Outcome var o da zaten encode edilmiş ----> DOKUNMA !


##### One-Hot Encoding
cat_cols = ["BMI_Category",
            "Age_Category",
            "Glucose_Level",
            "Insulin_Glucose_Ratio_Category"]

ohe_cols = [col for col in cat_cols if 10 >= df[col].nunique() > 2 and col != "Outcome"]
df = pd.get_dummies(df,columns=ohe_cols,drop_first=True)
ohe_cols
df.head()
df.dtypes
df.shape

# model True = 1, False = 0 olarak algılar
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

###########################
# Adım 5: Model oluşturunuz
###########################

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# customerID'yi zaten başlarda drop etmiştik
y = df["Outcome"]
X = df.drop(["Outcome"],axis = 1)

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=17)
# random forest modelini oluşturur ve train verisiyle eğitilir
# çok sayıda decision tree var ve avg'ı alınır
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)   #
accuracy_score(y_pred, y_test)

# X_train = modelin öğrendiği insan özellikleri (input)
# y_train = modelin öğrendiği output ----> Bu özelliklere sahip biri Diabet mi ?

# X_test =  yeni insan (modelin hiç görmediği)
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

# TN = Diabet olmayan kişiyi doğru bulmuş
# TP = Diabet olan kişiyi doğru bulmuş
# FP = model Diabet olmayan sağlıklı bireye Diabet dedi, yanlış yakaladı
# FN = model Diabet olan kişiye Diabet değil dedi, Diabet'i yakalayamadı.

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

