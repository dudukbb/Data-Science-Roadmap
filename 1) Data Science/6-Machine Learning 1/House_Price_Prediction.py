"""
Görev 1: Keşifçi Veri Analizi
Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
Adım 6: Aykırı gözlem var mı inceleyiniz.
Adım 7: Eksik gözlem var mı inceleyiniz.

Görev 2: Feature Engineering
Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
Adım 2: Rare Encoder uygulayınız.
Adım 3: Yeni değişkenler oluşturunuz.
Adım 4: Encoding işlemlerini gerçekleştiriniz.

Görev 3: Model Kurma
Adım 1: Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
Adım 2: Train verisi ile model kurup, model başarısını değerlendiriniz.
Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz. Not: Log'un tersini (inverse)
almayı unutmayınız.
Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.
Adım 4: Değişken önem düzeyini inceleyeniz.
Bonus: Test verisinde boş olan salePrice değişkenlerini tahminleyiniz ve Kaggle sayfasına submit etmeye uygun halde bir
dataframe oluşturup sonucunuzu yükleyiniz.

"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Found Intel OpenMP.*")

# model & evaluation
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# encoding
from sklearn.preprocessing import LabelEncoder

# linear models
from sklearn.linear_model import Ridge, Lasso

# tree & ensemble
from sklearn.ensemble import RandomForestRegressor

# boosting
from xgboost import XGBRegressor

pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

train_df = pd.read_csv(r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning\ML_7.Hafta\ML_7.Hafta_Case_Study\Case_1\train.csv")
train_df.head()
train_df.shape

test_df = pd.read_csv(r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning\ML_7.Hafta\ML_7.Hafta_Case_Study\Case_1\test.csv")
test_df.head()
test_df.shape

# -------------------------------------------------------------------------------------------------------------------------
# Görev 1: Keşifçi Veri Analizi
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Adım 1: Train ve Test veri setlerini okutup birleştiriniz.
# Birleştirdiğiniz veri üzerinden ilerleyiniz.
# -------------------------------------------------------------------------------------------------------------------------

df = pd.concat([train_df, test_df], axis = 0, ignore_index=True)
print(df.shape)
print(train_df.shape)
print(test_df.shape)
# SalePrice -----> target, testte SalePrice bilinmiyor biz tahmin edeceğiz
# kaggle sorusu oldugu için testtde SalePrice yok

# ----------------------------------------------------------------------------------------------------------------------------
# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
# ----------------------------------------------------------------------------------------------------------------------------

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

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df,cat_th=10,car_th=20)
cat_cols
num_cols
cat_but_car
df.head()
df.info

# -------------------------------------------------------------------------------------------------------------------------
# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# -------------------------------------------------------------------------------------------------------------------------

for col in num_cols:
    df[col] = df[col].astype(float)

# SalePrice ---> target
df.drop("Id", axis=1, inplace=True)
cat_cols = [ col for col in df.columns if df[col].dtype == "O"]
num_cols = [ col for col in df.columns
             if df[col].dtype != "O"
             and col != "SalePrice"]
cat_but_car = [ col for col in cat_cols if df[col].nunique() > 20]
cat_cols = [ col for col in cat_cols if col not in cat_but_car]
num_cols
cat_cols
cat_but_car

# -------------------------------------------------------------------------------------------------------------------------
# Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Categoric değişkenlerin dağılımı
# -------------------------------------------------------------------------------------------------------------------------
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

# bu değişkenler %99 tek sınıf ---> neredeyse tek class, bilgi taşımıyor, overfit riski yüksek
drop_cols = ["Street","Utilities","PoolQC","MiscFeature"]
df.drop(drop_cols, axis=1, inplace=True)

# drop sonrası cat_cols'u güncellememiz gerekiyor
cat_cols = [col for col in df.columns if df[col].dtype == "O"]


# -------------------------------------------------------------------------------------------------------------------------
# Numeric değişkenlerin dağılımı
# -------------------------------------------------------------------------------------------------------------------------
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

# sayıdal değişkenlerin grafiğini oluşturmak istersek
for col in num_cols:
    num_summary(df, col)

# -------------------------------------------------------------------------------------------------------------------------
# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
# -------------------------------------------------------------------------------------------------------------------------
df.head()
def cat_summary_with_target(df, target, col_name):
    print(f"######## {col_name} ########")
    print(pd.DataFrame({
        "Target_Mean": df.groupby(col_name)[target].mean(),
        "Count": df[col_name].value_counts(),
        "Ratio": 100 * df[col_name].value_counts() / len(df)
    }))
    print("\n")

for col in cat_cols:
    cat_summary_with_target(df,"SalePrice",col)

# cat_summary_with_target(df, "SalePrice", "GarageType")

# -------------------------------------------------------------------------------------------------------------------------
# Adım 6: Eksik gözlem var mı inceleyiniz.
# -------------------------------------------------------------------------------------------------------------------------

df.isnull().sum()
df.shape

df["SalePrice"].isnull().sum()   # bu 1459 SalePrice’ı biz tahmin edeceğiz
# 1459 eksik = test seti train + test birleştirince test'in SalePrice targetı için eksik kısma nan koymuş

# -------------------------------------------------------------------------------------------------------------------------
# Adım 7: Aykırı gözlem var mı inceleyiniz.
# -------------------------------------------------------------------------------------------------------------------------

df.describe().T
# outlier alt-üst sınırlarını hesaplar
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# outlier var mı o featureda ?
def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
  low, up = outlier_thresholds(df,col)
  print(col, "low:",low, "up:", up)

for col in num_cols:
    result = check_outlier(df,col)
    print(f"{col:10} -----> {result}")

# -------------------------------------------------------------------------------------------------------------------------
# Görev 2: Feature Engineering
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# -------------------------------------------------------------------------------------------------------------------------
# ---------------
# missing value
# ---------------
#categoric değişkenleri mode ile dolduruyoruz
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df.isnull().sum()

# mean ve median değerlerine baktık her feature için
# -> mean mi median ile mi eksik değerleri dolduracağımıza karar vermek için
for col in num_cols:
    print(col)
    print("mean:", df[col].mean())
    print("median:", df[col].median())
    print("-----")

df.isnull().sum()

# numeric değişkenleri median ile doldurduk
for col in num_cols:
    if col != "SalePrice":
        df[col] = df[col].fillna(df[col].median())

df.isnull().sum()

# -------------------------
# outlier
# -------------------------
# outlierları thresholdlarla değiştirir.
# outlierları sınır içine çeker.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].astype(float)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    result = check_outlier(df,col)
    print(f"{col:10} -----> {result}")
# -------------------------------------------------------------------------------------------------------------------------
# Adım 2: Rare Encoder uygulayınız.
# -------------------------------------------------------------------------------------------------------------------------
"""
# Kategorik değişkenlerde bazı sınıflar çok az görülür (ör. %0.2, %0.5 gibi).
# One-hot encoding yaptığında bu sınıflar: çok az örnek içerdiği için
# modeli yanıltabilir (overfit), gereksiz çok kolon üretir (boyut şişer).

# RARE ENCODING şu işi yapar:
# Frekansı belirli bir eşikten (örn. %1) düşük olan kategorileri tek bir etikette toplar: "Rare".

# Örnek:
# RoofMatl içinde çok az görülen 3–4 kategori varsa → hepsi "Rare" olur.
"""

def rare_encoder(dataframe, rare_perc=0.01, exclude_cols=None):
    """
    rare_perc: örn 0.01 => %1'den az görülen sınıflar Rare olur
    exclude_cols: Rare'e sokmak istemediğin kolonlar (örn: ["SalePrice", "is_train"])
    """
    # kopya df üzerinden çalışıyoruz ---> orijinal df bozulmasın
    temp_df = dataframe.copy()
    if exclude_cols is None:
        exclude_cols = []

    # sadece categoric kolonlar
    obj_cols = [col for col in temp_df.columns
                if temp_df[col].dtype == "O" and col not in exclude_cols]

    rare_columns = []
    for col in obj_cols:    # 1.for ---> hangi kolonlara rare encoding yapılacak
        ratios = temp_df[col].value_counts(normalize=True, dropna=False)   # her categorinin oranını hesaplıyor
        if (ratios < rare_perc).any():    # rare_perc = 0.01 , bir kolonda %1 altı kategori varsa
            rare_columns.append(col)     # rare_columns'a append ediyor

    for col in rare_columns:  # rare columlardaki categorilerin oranını hesaplıyor, içinde dolanıyor ve %1 altındakileri Rare labelını atıyor
        ratios = temp_df[col].value_counts(normalize=True, dropna=False)
        # %1'den az görünen kategorileri bul, isimlerini listeye al, sonra bunları rare yap
        rare_labels = ratios[ratios < rare_perc].index
        temp_df[col] = np.where(temp_df[col].isin(rare_labels), "Rare", temp_df[col])  # np.where(df[col], rare_labels içinde mi yani rare mi ?, "Rare" , eski değer)
    return temp_df, rare_columns

# temp_df ---> rare_encoding yapılmış yeni df
# rare_columns ---> rare encoding yapılmış kolonların listesi
df_rare, rare_cols = rare_encoder(df, rare_perc=0.01, exclude_cols=["SalePrice", "is_train"])
print("Rare uygulanan kolonlar:", rare_cols)

for col in rare_cols:
    print(col)
    print(df_rare[col].value_counts(normalize=True).head(10)) #Bu kolondaki kategorilerin yüzde dağılımını göster
    print("-"*40) # ekrana 40 tane - çizgi basar

# normalize=False → sayı verir
# normalize=True → oran verir

# -------------------------------------------------------------------------------------------------------------------------
# Adım 3: Yeni değişkenler oluşturunuz.
# -------------------------------------------------------------------------------------------------------------------------

# evin toplam kullanılabilir m2'si = toplam ev alanı
df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

# toplam banyo sayısı
df["TotalBath"] = (
    df["FullBath"] +
    df["HalfBath"]*0.5 +
    df["BsmtFullBath"] +
    df["BsmtHalfBath"]*0.5
)

# evin yaşı
df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

# toplam balkon
df["TotalPorch"] = (
    df["OpenPorchSF"] +
    df["EnclosedPorch"] +
    df["3SsnPorch"] +
    df["ScreenPorch"]
)

# kalite * büyüklük
df["Qual_TotalSF"] = df["OverallQual"] * df["TotalSF"]

df.head()

# -------------------------------------------------------------------------------------------------------------------------
# Adım 4: Encoding işlemlerini gerçekleştiriniz.
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
#Label Encoding
# -------------------------------------------------------------------------------------------------------------------------
binary_cols = [col for col in df.columns
               if df[col].dtype == "O" and df[col].nunique() == 2]

binary_cols

le = LabelEncoder()

for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df[binary_cols]

# -------------------------------------------------------------------------------------------------------------------------
# One - Hot Encoding
# -------------------------------------------------------------------------------------------------------------------------
# cat_colsu güncelledik
cat_cols = [col for col in df.columns if df[col].dtype == "O"]
ohe_cols = [col for col in cat_cols if df[col].nunique()>2]
ohe_cols

# drop_first = True yapalım ki Dummy değişken tuzağından kurtulalım
# yani multicollinearity olmasın ---> bir değişken diğeri üzerinden üretilemesin / tahmin edilemesin
df = pd.get_dummies(df,columns=ohe_cols,drop_first=True)
df.head()
df.shape

# -------------------------------------------------------------------------------------------------------------------------
# Görev 3: Model Kurma
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Adım 1: Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
# -------------------------------------------------------------------------------------------------------------------------
# SalePrice dolu olanlar → TRAIN
# SalePrice boş olanlar → TEST (tahmin yapılacak)

# train ve test ayır
train_df = df[df["SalePrice"].notnull()].copy()
test_df = df[df["SalePrice"].isnull()].copy()

# target
y = train_df["SalePrice"]

# modelin öğreneceği özelllikler
X = train_df.drop("SalePrice",axis = 1)

# tahmin yapılacak test verisi
X_test = test_df.drop("SalePrice",axis=1)

# submission için test id'leri sakla
test_id = pd.read_csv(r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning\ML_7.Hafta\ML_7.Hafta_Case_Study\Case_1\test.csv")

# -------------------------------------------------------------------------------------------------------------------------
# Adım 2: Train verisi ile model kurup, model başarısını değerlendiriniz.
# ( RMSE , Cross Validation )
# -------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------
# Adım 2: Train verisi ile model kur ve CV ile RMSE değerlendir
# ------------------------------------------------------------

import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

def base_models_rmse(X, y, cv=3):
    print(f"Base Models (RMSE, CV={cv})")
    models = [
        ("Ridge", Ridge()),
        ("RF", RandomForestRegressor(n_estimators=400, random_state=17, n_jobs=1)),
        ("GBM", GradientBoostingRegressor(random_state=17)),
        ("XGB", XGBRegressor(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=17,
            objective="reg:squarederror",
            n_jobs=1
        ))
    ]

    for name, model in models:
        cv_res = cross_validate(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        rmse = -cv_res["test_score"].mean()
        print(f"RMSE: {rmse:.4f} ({name})")

print("### Log YOK ###")
base_models_rmse(X, y, cv=3)

# ------------------------------------------------------------------------------------------------------------------------------
# Bonus: Hedef değişkene log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz. Not: Log'un tersini (inverse)
# almayı unutmayınız.
# ------------------------------------------------------------------------------------------------------------------------------
# SalePrice log alınır çünkü dağılım çarpık ve outlier çoktur.
# Log dönüşümü modeli daha stabil ve doğru tahmin yapar hale getirir

# Target log dönüşümü
y_log = np.log1p(y)

print("\n### Log VAR (log1p) ###")
base_models_rmse(X, y_log, cv=3)

# -------------------------------------------------------------------------------------------------------------------------
# Adım 3: Hiperparemetre optimizasyonu gerçekleştiriniz.
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Adım 3: Hiperparametre Optimizasyonu
# Amaç: Seçilen modeller için en iyi parametreleri bulup CV-RMSE'yi düşürmek
# # Not: Bu versiyon PREPROCESS YOK varsayar (X tamamen sayısal, NaN yok) ---> preprocesisi biz yaptık  önceden
# -------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------
# Adım 3: Hiperparametre Optimizasyonu
# Amaç: Seçilen modeller için en iyi parametreleri bulup CV-RMSE'yi düşürmek
# Not: Bu versiyon PREPROCESS YOK varsayar (X tamamen sayısal, NaN yok)
# -------------------------------------------------------------------------------------------------------------------------

import numpy as np
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# Log hedef ile optimize edeceğiz (House Prices için önerilir)
y_log = np.log1p(y)

# ------------------------------
# model nesneleri
# ------------------------------
ridge = Ridge()
rf = RandomForestRegressor(random_state=17, n_jobs=1)
gbm = GradientBoostingRegressor(random_state=17)
xgb = XGBRegressor(random_state=17, objective="reg:squarederror", n_jobs=1)

# --------------------------------------------
# Ridge Params
# --------------------------------------------
ridge_params = {"alpha": [0.1, 1, 10, 50, 100]}
rf_params = {
    "n_estimators": [400, 700],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 10]
}

# --------------------------------------------
# GBM Params
# --------------------------------------------
gbm_params = {
    "n_estimators": [200, 500],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [2, 3, 4]
}

# --------------------------------------------
# XGBoost Params
# --------------------------------------------
xgb_params = {
    "n_estimators": [400, 800],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}

# ------------------------------------------
# Models
# ------------------------------------------
models = [
    ("Ridge", ridge, ridge_params),
    ("RF", rf, rf_params),
    ("GBM", gbm, gbm_params),
    ("XGB", xgb, xgb_params)
]

# ------------------------------------------------------------
# 2) Hiperparametre Optimizasyonu Fonksiyonu (Before/After RMSE)
# ------------------------------------------------------------

def hyperparameter_optimization(X, y, cv=3):
    print(f"Hyperparameter Optimization (CV={cv}) - RMSE on log1p(SalePrice)")
    best_models = {}

    for name, model, params in models:
        print(f"\n########## {name} ##########")

        # Optimizasyon öncesi RMSE
        before = cross_validate(model, X, y, cv=cv,
                                scoring="neg_root_mean_squared_error")["test_score"].mean()
        print(f"Before RMSE: {-before:.4f}")

        # GridSearch ile en iyi parametreleri bul
        gs = GridSearchCV(model, params, cv=cv, n_jobs=1,
                          scoring="neg_root_mean_squared_error")
        gs.fit(X, y)

        best_model = gs.best_estimator_

        # Optimizasyon sonrası RMSE
        after = cross_validate(best_model, X, y, cv=cv,
                               scoring="neg_root_mean_squared_error")["test_score"].mean()
        print(f"After RMSE : {-after:.4f}")
        print(f"Best Params: {gs.best_params_}")

        best_models[name] = best_model

    return best_models

best_models = hyperparameter_optimization(X, y_log, cv=3)

# -------------------------------------------------------------------------------------------------------------------------
# BONUS
# - Test verisindeki SalePrice'ları tahmin et
# - submission.csv üret
# -------------------------------------------------------------------------------------------------------------------------

final_model = best_models["XGB"]

y_log = np.log1p(y)
final_model.fit(X,y_log)

pred_log = final_model.predict(X_test)

pred_real = np.expm1(pred_log)

# submission için test id'leri sakla
test_id = pd.read_csv(r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning\ML_7.Hafta\ML_7.Hafta_Case_Study\Case_1\test.csv")["Id"]


submission = pd.DataFrame({
    "Id" : test_id,
    "SalePrice" : pred_real
})

save_path = r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning\ML_7.Hafta\ML_7.Hafta_Case_Study\Case_1\house_price_submission.csv"
submission.to_csv(save_path, index=False)
print("Kaydedildi:", save_path)