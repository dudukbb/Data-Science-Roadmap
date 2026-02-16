"""
Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
olacak şekilde manipülasyon yapınız.
- Step 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
“attribute_value” olacak şekilde pivot table’ı oluşturunuz.
- Step 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz

Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz

"""
# -----------------------------------------------------------------------------------------------------------------------------------
# Adım 1: scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.
# -----------------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("machine_learning/ML_8.Hafta_Case_Study/scoutium_attributes.csv", sep=";")
df.head()
df_label = pd.read_csv("machine_learning/ML_8.Hafta_Case_Study/scoutium_potential_labels.csv", sep = ";")
df_label.head()

# -----------------------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
# ------------------------------------------------------------------------------------------------------------------------------------------------
common_columns = ["task_response_id", "match_id", "evaluator_id", "player_id"]
merged_df = pd.merge(df,df_label,on = common_columns,how="inner")
merged_df.head()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 3: position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# merged_df içinde position_id'si 1 olanların indexlerini al ve bunları merged_df'ten drop et
merged_df = merged_df.drop(merged_df[merged_df["position_id"] == 1].index)
merged_df["position_id"].unique() # 1 yok, nice

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
# ------------------------------------------------------------------------------------------------------------------------------------------------------------
merged_df = merged_df.drop(merged_df[merged_df["potential_label"] == "below_average"].index)
merged_df["potential_label"].unique()

# -----------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
# olacak şekilde manipülasyon yapınız.
# -----------------------------------------------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# - Step 1: İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
# “attribute_value” olacak şekilde pivot table’ı oluşturunuz.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
pivot_df = merged_df.pivot_table(
        index=["player_id", "position_id", "potential_label"],
        columns="attribute_id",
        values="attribute_value"
).reset_index()

pivot_df.head()

pivot_df.columns = [str(col) for col in pivot_df.columns]
pivot_df.head()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# - Step 2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz
# --------------------------------------------------------------------------------------------------------------------------------------------------------------

pivot_df = pivot_df.reset_index()
pivot_df.columns = pivot_df.columns.astype(str)
pivot_df.head()

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
# --------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder

binary_cols = ["potential_label"]

le = LabelEncoder()

for col in binary_cols:
    pivot_df[col] = le.fit_transform(pivot_df[col])

pivot_df.head()
pivot_df["potential_label"]

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 7: Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
pivot_df.dtypes

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

cat_cols, num_cols, cat_but_car = grab_col_names(pivot_df,cat_th=10,car_th=20)

cat_cols = [ "position_id","potential_label"]
cat_cols
num_cols = [col for col in pivot_df.columns if col not in ["player_id","index","position_id","potential_label"]]
num_cols

cat_but_car
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# StandardScaler -----> avg = 0, std = 1
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
pivot_df[num_cols] = sc.fit_transform(pivot_df[num_cols])
pivot_df.head()

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

y = pivot_df["potential_label"]
X = pivot_df.drop("potential_label",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state = 42)

# evalutaion metric function
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# --------------------------------------------------------------------------------------------------------------------------
# evaluation metrics
# --------------------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, \
    precision_score, recall_score, f1_score, roc_auc_score

def print_metrics(y_test, y_pred, y_proba):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)  # y_proba --> modelin class 1 olma olasılık tahmini

    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC_AUC  : {roc:.3f}")

"""
ROC-AUC = Modelin ayırt etme gücü
- ROC eğrisi altındaki alandır
- TPR ile FPR ilişkisini ölçer
- modelin ayırma gücünü gösterir
- eğer model iyi ayrım yapıyorsa ROC-AUC skor, 1'e yaklaşır
- highlighted oyuncular → yüksek skor
- normal oyuncular → düşük skor
"""
# ---------------------------------------------------------------------------------------------------
# base models function
# ------------------------------------------------------------------------------------------------
def base_models(X_train, X_test, y_train, y_test):
    print("-----------------Base Models--------------")
    models = [
        ("LR", LogisticRegression(max_iter=5000,class_weight="balanced")),
        ("KNN", KNeighborsClassifier()),
        ("SVC", SVC(class_weight="balanced", probability = True)),
        ("CART", DecisionTreeClassifier(class_weight="balanced")),
        ("RF", RandomForestClassifier(class_weight="balanced",random_state=42)),
        ("GBM", GradientBoostingClassifier()),
        ("NaiveBayes", GaussianNB())
    ]

    for name, model in models:
        print(f"\n--------{name}-------")
        model.fit(X_train, y_train)    # modeli trainlerle eğitiyoruz
        y_pred = model.predict(X_test)  # Model daha önce görmediği test verisine bakar (X_test), Her futbolcu için tahmin üretir
        y_probability = model.predict_proba(X_test)[:,1]  # her futbolcunun Class 1 (highlighted) olma olasılığı
        print_metrics(y_test, y_pred, y_probability)

base_models(X_train, X_test, y_train, y_test)
"""
# bu problem için en uygun modeli seçerken Accuracy dışında 
bunlara da bakmalıyız sırayla
# 1) ROC-AUC 
# 2) F1-score (class 1 için)
# 3) Recall (class 1 için)
"""

final_model = RandomForestClassifier(class_weight="balanced",random_state = 42)
final_model.fit(X_train, y_train)

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def plot_feature_importance(model, X, num = 10):
    feature_imp =  pd.DataFrame({
         "Feature": X.columns,
         "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)

    print(feature_imp.head(num))

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_imp.head(10))
    plt.title("Feature Importance")
    plt.show()

plot_feature_importance(final_model,X)

