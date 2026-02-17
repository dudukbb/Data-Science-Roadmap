"""
Görev 1 : Keşifçi Veri Analizi
# ------------------------------------------------------------------------------------------------------------------------------
Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
Adım 5: Aykırı gözlem var mı inceleyiniz.
Adım 6: Eksik gözlem var mı inceleyiniz.


Görev 2 : Feature Engineering
# -----------------------------------------------------------------------------------------------------------------------------
Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
Adım 2: Yeni değişkenler oluşturunuz.
Adım 3: Encoding işlemlerini gerçekleştiriniz.
Adım 4: Numerik değişkenler için standartlaştırma yapınız.


Görev 3 : Modelleme
# -----------------------------------------------------------------------------------------------------------------------------
Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile
modeli tekrar kurunuz.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import RocCurveDisplay
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler as RBScaler
from sklearn.neighbors import KNeighborsClassifier


pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


# CSV'yi oku
df = pd.read_csv(r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning\ML_8.Hafta\ML_8.Hafta_Case_Study\Telco_Customer_Churn.csv")
df.head()

# --------------------------------------------------------------------------------------------------------------------------------
# Görev 1 : Keşifçi Veri Analizi
# ------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
# -----------------------------------------------------------------------------------------------------------------------------------------
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, num_but_car = grab_col_names(df, cat_th=10, car_th=20)
cat_cols
num_cols
num_but_car

# ------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# ------------------------------------------------------------------------------------------------------------------------------
# data hakkında genel bilgi edindim (tipleri vs)
df.info()

df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

df["SeniorCitizen"] = df["SeniorCitizen"].astype("category")

df["Churn"] = df["Churn"].map({"No":0, "Yes":1})
df["Churn"] = df["Churn"].astype(int)

df.info()

# ------------------------------------------------------------------------------------------------------------------------------
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
# -------------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------
# categoric değişkenlerin dağılımı
# ----------------------------------------------------------------------------------------
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df,col,plot=False)

# ---------------------------------------------------------------------------------------------
# numeric değişkenlerin dağılımı
# ---------------------------------------------------------------------------------------------------------
num_cols

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df,col,plot=False)

# -------------------------------------------------------------------------------------------------
# target'ın dağılımı
# -------------------------------------------------------------------------------------------------
def target_summary(df,target,plot=False):

    # numeric target
    if df[target].dtype != "O":
        quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

        print("Target Summary -- Numeric ")
        sum = df[target].describe(quantiles).to_frame().T
        print(sum)
        print("\n----Target Özet İstatistikler----")
        print(f"Max  : {df[target].max():.3f}")
        print(f"Min  : {df[target].min():.3f}")
        print(f"Mean  : {df[target].mean():.3f}")
        print(f"Median : {df[target].median():.3f}")

        if plot:
            plt.figure(figsize=(7,4))
            df[target].hist(bins=30)
            plt.xlabel(target)
            plt.title(f"{target} Distribution")
            plt.show()

    # categoric target
    else:
        print("Target Summary -- Categorical ")
        print(df[target].value_counts())
        print("Ratio: ")
        print(100* df[target].value_counts()/len(df))

        if plot:
            sns.countplot(x=df[target])
            plt.title(f"{target} Countplot")
            plt.show()

target_summary(df,"Churn",plot=False)

# ------------------------------------------------------------------------------------
#  numeric değişkenler ile hedef değişken incelemesi
# -----------------------------------------------------------------------------------------------------

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


# target ile numeric değişkenler arasındaki korelasyon analizi
df.corr(numeric_only=True)["Churn"].sort_values(ascending=False)

# ----------------------------------------------------------------------------------------------------------------------------------
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
# --------------------------------------------------------------------------------------------------------------------------------

def cat_summary_with_target(df, target, col_name, plot=False):
    print(f"######## {col_name} ########")

    print(pd.DataFrame({
        "Churn_Mean": df.groupby(col_name, observed=False)[target].mean(),
        "Count": df[col_name].value_counts(),
        "Ratio": 100 * df[col_name].value_counts() / len(df)
    }))

    print("\n")

for col in cat_cols:
    cat_summary_with_target(df,"Churn",col)

df.head()
# ----------------------------------------------------------------------------------------------------------------------------------
# Adım 5: Aykırı gözlem var mı inceleyiniz.
# ----------------------------------------------------------------------------------------------------------------------------------
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

    if ((dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)).any():
        return True
    else:
        return False


for col in num_cols:
  low, up = outlier_thresholds(df,col)
  print(col, "low:",low, "up:", up)


for col in num_cols:
    result = check_outlier(df,col)
    print(f"{col:10} -----> {result}")

# ---------------------------------------------------------------------------------------------------------------------------------
# Adım 6: Eksik gözlem var mı inceleyiniz.
# ---------------------------------------------------------------------------------------------------------------------------------
df.isnull().sum()

# -----------------------------------------------------------------------------------------------------------------------------
# Görev 2 : Feature Engineering
# -----------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# -------------------------------------------------------------------------------------------------------------------------------------------

# outlierları thresholdlarla değiştirir.
# outlierları sınır içine çeker.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    result = check_outlier(df,col)
    print(f"{col:10} -----> {result}")

# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Yeni değişkenler oluşturunuz.
# ------------------------------------------------------------------------------------------------------------------------------

# TotalServices: aldığı toplam "Yes" hizmet sayısı
    service_cols = ["PhoneService", "MultipleLines", "InternetService",
                    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]

    # müşterinin aldığı toplam hizmet sayısı
    df["TotalServices"] = df[service_cols].apply(lambda x: (x == "Yes").sum(), axis=1)

    # HasInternet: InternetService ---> Eğer x=No ise 0, değil ise 1
    df["HasInternet"] = df["InternetService"].apply(lambda x: 0 if x == "No" else 1)

    # Müşteri Fiber internet kullanıyorsa 1, kullanmıyorsa 0
    df["IsFiber"] = (df["InternetService"] == "Fiber optic").astype(int)

    # ContractRisk: sözleşme türüne göre churn riski skorlaması
    df["ContractRisk"] = df["Contract"].map({
        "Month-to-month": 3,
        "One year": 2,
        "Two year": 1
    })

    # PaymentRisk: ödeme yöntemine göre risk skoru
    df["PaymentRisk"] = df["PaymentMethod"].map({
        "Electronic check": 3,
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 1
    })

    # TenureGroup: müşteri yaşam süresi segmenti
    df["TenureGroup"] = pd.cut(df["tenure"],
                               bins=[-1, 6, 12, 24, 48, 72],
                               labels=["0-6 ay", "6-12 ay", "1-2 yıl", "2-4 yıl", "4+ yıl"])

    # ExpensivePlan: pahalı plan mı? (median üstü)
    df["ExpensivePlan"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # AvgMonthlySpend: toplam ücret / tenure (tenure 0 ise MonthlyCharges)
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, np.nan)
    df["AvgMonthlySpend"] = df["AvgMonthlySpend"].fillna(df["MonthlyCharges"])

    df.head()
    df.info()
# -------------------------------------------------------------------------------------------------------------------------------------
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# --------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

def label_encoder(dataframe, col):
    le = LabelEncoder()
    dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe

# binary_cols = [ "gender", "Partner", "Dependents","PhoneService","PaperlessBilling" ]
# Binary (2 sınıflı) kategorikler -> Label Encoding
binary_cols = [ col for col in df.columns
                if df[col].dtype == "O"
                and df[col].nunique() == 2
                and col != "Churn"]

for col in binary_cols:
    df = label_encoder(df, col)

# 2'den fazla sınıfı olan kategorikler -> One-Hot Encoding
# ohe_cols = ["MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
# "StreamingMovies","Contract","PaymentMethod","TenureGroup" ]
ohe_cols = [col for col in df.columns
            if col != "Churn"
            and (df[col].dtype == "O" or str(df[col].dtype) == "category")
            and df[col].nunique() > 2]

df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

df.info()
# ----------------------------------------------------------------------------------------------------------------------------------
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
# -----------------------------------------------------------------------------------------------------------------------------
# RobustScaler = (x-median)/IQR
from sklearn.preprocessing import RobustScaler

# scaling yapılacak numerikler (target hariç)
num_cols = [col for col in df.columns
            if col != "Churn" and df[col].dtype in [int, float] and df[col].nunique() > 10]

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()
# ------------------------------------------------------------------------------------------------------------------------------
# Görev 3 : Modelleme
# -----------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
# --------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Target ve feature ayır
y = df["Churn"]
X = df.drop("Churn", axis=1)

######################################################
# BASE MODELS
######################################################

def base_models(X, y):
    print("Base Models...\n")

    models = [
        ("LR", LogisticRegression(max_iter=1000)),
        ("KNN", KNeighborsClassifier()),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier()),
        ("GBM", GradientBoostingClassifier())
    ]

    results = []

    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=5,
                                    scoring=["accuracy", "f1", "roc_auc"])

        acc = cv_results["test_accuracy"].mean()
        f1 = cv_results["test_f1"].mean()
        auc = cv_results["test_roc_auc"].mean()

        print(f"######## {name} ########")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1     : {f1:.4f}")
        print(f"ROC_AUC: {auc:.4f}\n")

        results.append((name, acc))

    return results

results = base_models(X, y)

# En iyi 4 modeli seç
results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
best_models = [model[0] for model in results_sorted[:4]]

print("En iyi 4 model:", best_models)

# --------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile
# modeli tekrar kurunuz.
# -------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression
# ----------------------------------------------------------------------------------------------------------------------------------

# LR için denenecek parametre listesi
lr_params = {"C": [0.01, 0.1, 1, 10]}

# GridSearchCV ---> en iyi parametre kombinasyonunu bulur
# farklı parametreleri dener
# cv=5 ile test eder
# en iyi roc_auc veren parametreyi seçer
lr_best = GridSearchCV(LogisticRegression(max_iter=1000),
                       lr_params,
                       cv=5,
                       scoring="roc_auc").fit(X, y)

lr_final = LogisticRegression(**lr_best.best_params_).fit(X, y)

print("LR best params:", lr_best.best_params_)

# --------------------------------------------------------------------------------------------------------------------------------
# KNN
# --------------------------------------------------------------------------------------------------------------------------------
knn_params = {"n_neighbors": range(3,15)}

knn_best = GridSearchCV(KNeighborsClassifier(),
                        knn_params,
                        cv=5,
                        scoring="roc_auc").fit(X, y)

knn_final = KNeighborsClassifier(**knn_best.best_params_).fit(X, y)

print("KNN best params:", knn_best.best_params_)

# ----------------------------------------------------------------------------------------------------------------------------------
# Random Forest
# ----------------------------------------------------------------------------------------------------------------------------------
rf_params = {
    "n_estimators": [200, 500],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

rf_best = GridSearchCV(RandomForestClassifier(),
                       rf_params,
                       cv=5,
                       scoring="roc_auc").fit(X, y)

rf_final = RandomForestClassifier(**rf_best.best_params_).fit(X, y)

print("RF best params:", rf_best.best_params_)

# -------------------------------------------------------------------------------------------------------------------------------------
# Gradient Boosting
# --------------------------------------------------------------------------------------------------------------------------------------
gbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [200, 500],
    "max_depth": [3, 5]
}

gbm_best = GridSearchCV(GradientBoostingClassifier(),
                        gbm_params,
                        cv=5,
                        scoring="roc_auc").fit(X, y)

gbm_final = GradientBoostingClassifier(**gbm_best.best_params_).fit(X, y)

print("GBM best params:", gbm_best.best_params_)


# ------------------------------------------------------------------------------------------------------------------------------------
# Final Model Performansı Karşılaştırma
# ------------------------------------------------------------------------------------------------------------------------------------
models_final = [
    ("LR Final", lr_final),
    ("KNN Final", knn_final),
    ("RF Final", rf_final),
    ("GBM Final", gbm_final)
]

for name, model in models_final:
    cv_results = cross_validate(model, X, y, cv=5,
                                scoring=["accuracy", "f1", "roc_auc"])

    print(f"######## {name} ########")
    print(f"Accuracy: {cv_results['test_accuracy'].mean():.4f}")
    print(f"F1     : {cv_results['test_f1'].mean():.4f}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean():.4f}\n")


final_model = lr_final