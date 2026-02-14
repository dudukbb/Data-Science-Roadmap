
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("machine_learning/ML_7.Hafta_Case_Study/Bonus_Case_1/hitters.csv")
df.head()

# ---------------------------------------------------------------------------------------------------------
# Adım 1 : Keşifçi Veri Analizi
# -------------------------------------------------------------------------------------------

# numerik ve categorik değişkenleri yakalayınız

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

cat_cols   # League, Division, NewLeague
num_cols   # diğerleri num
cat_but_car  # 0

df.head()
# -------------------------------------------------------------------------------------
# gerekli preprocessing işlemlerini yapınız
# ---------------------------------------------------------------------------------------

# Salary ---> target, bu yüzden num_cols'dan çıkarmamız gerek
num_cols.remove("Salary")
num_cols

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
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
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
        quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                     0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

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

target_summary(df,"Salary",plot=False)

# -----------------------------------------------------------------------------------------------------
# target ile categoric değişkenlerin incelemesi
# -------------------------------------------------------------------------------------------------------

def cat_summary_with_target(df, target, col_name):
    print(f"######## {col_name} ########")
    print(pd.DataFrame({
        "Target_Mean": df.groupby(col_name)[target].mean(),
        "Count": df[col_name].value_counts(),
        "Ratio": 100 * df[col_name].value_counts() / len(df)
    }))
    print("\n")

for col in cat_cols:
    cat_summary_with_target(df,"Salary",col)

# ------------------------------------------------------------------------------------
# target ile numeric değişkenler arasındaki korelasyon analizi
# -----------------------------------------------------------------------------------------------------

df.corr(numeric_only=True)["Salary"].sort_values(ascending=False)

"""
# her maaş neredeyse unique oldugu için gruplamak istatiksel olarak anlamsız
# bu yüzden target ile numeric değişkenler arasındaki korelasyona bakacağız
# target = Salary

# Salary üzerinde en çok etki eden featurelar
# CRBI    0.567
# CRuns   0.563
# CHits   0.549
# CAtBat  0.526
# CHmRun  0.525
"""

# ------------------------------------------------------------------------------------
# eksik değer
# ----------------------------------------------------------------------------------------------------

df.isnull().sum()
df.shape

# Salary'de 59 eksik var
# targettaki eksik value'yu dolduramayız, model yanılır bu yüzden siliyoruz.
df.dropna(subset=["Salary"],inplace = True)
df.isnull().sum()

# ----------------------------------------------------------------------------------------
# outlier
# --------------------------------------------------------------------------------------------------

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

# outlierları thresholdlarla değiştirir.
# outlierları sınır içine çeker.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
  low, up = outlier_thresholds(df,col)
  print(col, "low:",low, "up:", up)

for col in num_cols:
    result = check_outlier(df,col)
    print(f"{col:10} -----> {result}")

for col in num_cols:
    replace_with_thresholds(df,col)

for col in num_cols:
    result = check_outlier(df,col)
    print(f"{col:10} -----> {result}")

# -------------------------------------------------------------------------------------------------
# yeni featurelar üret
# --------------------------------------------------------------------------------------------------------------

df["Avg_Hits"] = df["CHits"] / df["CAtBat"]
df["Hits_per_Year"] = df["CHits"] / df["Years"]
df["Career_Power"] = df["CHmRun"] + df["CRBI"]
df["Total_Performance"] = df["CRuns"] + df["CRBI"] + df["CHits"]
df["Hit_Ratio"] = df["Hits"] / df["CHits"]
df.head()

new_cols = ["Avg_Hits","Hits_per_Year","Career_Power","Total_Performance","Hit_Ratio"]
# -----------------------------------------------------------------------------------------------
# yeni ürettiğimiz featurelarda eksik değer var mı? YOKMUŞ
# ------------------------------------------------------------------------------------------------------------
df[new_cols].isnull().sum()

# -----------------------------------------------------------------------------------------
# yeni ürettiğimiz featurelarda outlier var mı ?
# ------------------------------------------------------------------------------------------------------

for col in new_cols:
  low, up = outlier_thresholds(df,col)
  print(col, "low:",low, "up:", up)

for col in new_cols:
    result = check_outlier(df,col)
    print(f"{col:10} -----> {result}")

for col in new_cols:
    replace_with_thresholds(df,col)

for col in new_cols:
    result = check_outlier(df, col)
    print(f"{col:10} -----> {result}")

# ---------------------------------------------------------------------------------------------------------------
# binary encoding
# ------------------------------------------------------------------------------------------------------
# League --> A/N
# Division --> E/W
# NewLeague --> A/N
# categoric olan sadece 3 değişken var ve onlara da binary encoding uygulayacağız.

from sklearn.preprocessing import LabelEncoder

binary_cols = ["League","Division","NewLeague"]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

df[binary_cols]
df.head()

# --------------------------------------------------------------------------------------------------------
# Model kurma
# --------------------------------------------------------------------------------------------------
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# right skewed target o nedenle log transform yapıcaz
df["Salary"] = np.log1p(df["Salary"])

y = df["Salary"]
X = df.drop("Salary",axis=1)

# --------------------------------------------------------------
# Train / Test split
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# --------------------------------------------------------------
# RMSE fonksiyonu (sadece traind ekullnacagız)
# ---------------------------------------------------------------
"""
# cross_validate(model, x, y, cv=5)
# - veriyi 5 parçaya böl
# - 5 kez model eğit
# - her seferinde test et
# - ortalama performansı hesapla   ---> bu işlemlerin tamamına cross validation denir.
"""
def rmse_cv(model, X, y):
    rmse = np.mean( np.sqrt(-cross_validate(model, X, y,cv=5,
            scoring="neg_mean_squared_error")["test_score"])
    )
    return rmse

# --------------------------------------------------------------------------------------------
# Base models
# --------------------------------------------------------------------------------------------
from sklearn.pipeline import Pipeline

def base_models(X_train, y_train):
    models = [
        ("LR", LinearRegression()),         # linear regression
        ("Ridge", Ridge()),                 # L2 Regularization
        ("Lasso", Lasso(max_iter=10000)),   # L1 Regularization
        ("RF", RandomForestRegressor(random_state=42)),    # ensemble tree
        ("GBM", GradientBoostingRegressor(random_state=42))    # boosting model
    ]


    print("----Base Models RMSE Results----\n")
    for name, model in models:
        pipe = Pipeline([
            ("scaler", StandardScaler()),     # satndart scaler
            ("model", model)
        ])

        # model performansını RMSE ile ölçüyoruz
        rmse = rmse_cv(pipe, X_train, y_train)
        print(f"{name} RMSE: {rmse:.4f}")

 # bu çıktı bize hangi model daha iyi gösterir
 # en düşük RMSE = en iyi model
 base_models(X_train,y_train)


# --------------------------------------------------------------
# Final model (GBM seçildi) -> SADECE train ile fit
# --------------------------------------------------------------
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42))
])

final_model.fit(X_train, y_train)

# --------------------------------------------------------------
# Test performansı (gerçek, hiç görülmemiş veri)
# --------------------------------------------------------------
from sklearn.metrics import mean_squared_error

# model prediction yapıyor eğittimiz X_test ile
y_pred = final_model.predict(X_test)

# gerçek test RMSE (y_test) ile modelin prediction ettiği RMSE değeri (y_pred) karşılaştırıyoruz
# gerçek - tahmin = hata -----> hataların ortalaması RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\nFinal Model: GBM")

# Test RMSE: modelin hiç görmediği veri üzerindeki gerçek performansı
# Değer ne kadar küçükse model o kadar başarılıdır
print(f"TEST RMSE: {test_rmse:.4f}")


#-------------------------------------------------------------
# Pipeline neden kullandık ?
# Data leakage önlemek ve scaling işlemini cross validation
# içinde doğru şekilde uygulamak için.