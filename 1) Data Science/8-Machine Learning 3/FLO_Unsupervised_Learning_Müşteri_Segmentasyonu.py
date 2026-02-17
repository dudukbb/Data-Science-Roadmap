
"""
Görev 1: Veriyi Hazırlama
---------------------------------------------------------------------------------------------------
Adım 1: flo_data_20K.csv verisini okutunuz.
Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.


Görev 2: K-Means ile Müşteri Segmentasyonu
-------------------------------------------------------------------------------------
Adım 1: Değişkenleri standartlaştırınız.
Adım 2: Optimum küme sayısını belirleyiniz.
Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.


Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
-----------------------------------------------------------
Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz
"""


# -----------------------------------------------------------------------------------------------------------------
# Adım 1: flo_data_20K.csv verisini okutunuz.
# -----------------------------------------------------------------------------------------------------------------
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv("machine_learning/ML_9.Hafta_Case_Study/flo_data_20k.csv")
df.head()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.
# -------------------------------------------------------------------------------------------------------------------------------------------------------------
import datetime as dt

df.head()

# today_date tarihi belirledik
today_date = dt.datetime(2021, 6, 1)

# dataset içinde tarihler string olarak geliyor ----> "2020-05-12"
# biz ise stringden tarih çıkarma işlemi yaptık
df["first_order_date"] = pd.to_datetime(df["first_order_date"])
df["last_order_date"] = pd.to_datetime(df["last_order_date"])
df["last_order_date_offline"] = pd.to_datetime(df["last_order_date_offline"])
df["last_order_date_online"] = pd.to_datetime(df["last_order_date_online"])
df.dtypes

# tenure ---> müşteri yaşı (hafta cinsinden)
# tenure = müşteri bizimle ne kadar süredir ilişkide ?
df["tenure"] = ((today_date - df["first_order_date"]).dt.days) / 7
df.head()

# recency = müşterinin son alışverişinden bugüne kadar geçen süre
# son görülme
df["recency"] = (today_date - df["last_order_date"]).dt.days / 7

# toplam order sayısı
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# toplam harcama
df["total_value"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# average order value
df["avg_order_value"] = df["total_value"] / df["total_order"]

df.head()

# -----------------------------------------------------------------------------------------------------------
# Görev 2: K-Means ile Müşteri Segmentasyonu
# ------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------
# Adım 1: Değişkenleri standartlaştırınız.
# ---------------------------------------------------------------------------------------------------------------------------------------
# Standardization = Z-score Normalization
# z = (x - mean)/std
# AMAÇ: mean --> 0 , std --> 1

from sklearn.preprocessing import StandardScaler

kmeans_cols = ["tenure","recency","total_order","total_value","avg_order_value"]

# fit_transform = model veriyi öğrenir (fit), veriyi dönüştürür (transform)
sc = StandardScaler()
kmeans_df = pd.DataFrame(
    sc.fit_transform(df[kmeans_cols]),
    columns = kmeans_cols,
    index = df.index
)

kmeans_df.head()

# ---------------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Optimum küme sayısını belirleyiniz.
# ------------------------------------------------------------------------------------------------------------------------------------------
# KMeans ---> veri setini k tane kümeye ayıran unsupervised model
# matplotlib ---> grafik çizmek için

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# path hatası olmasın diye r koyduk başa
save_path = r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning/ML_9.Hafta_Case_Study/elbow.png"

# WCSS (Within Cluster Sum of Squarred)
# WCSS = her noktanın kendi cluster merkezine uzaklığının karesi ----> (x-centroid)^2
# her k değeri için modelin hata değerini saklayacağımız liste ---> k tane WCSS
inertia = []       # KMeans modelinin hata değeri = WCSS = inertia

K = range(1, 11)                  # k = 1,2,3,4,5,6,7,8,9,10
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(kmeans_df)     # modeli eğitiyoruz ---> model datasetindeki noktaları kümelere ayırır ve cluster merkezlerini öğrenir
    inertia.append(kmeans.inertia_)   # WCSS'yi inertia listine ekler

plt.figure(figsize=(8,5))
plt.plot(K, inertia, marker="o")
plt.xlabel("Cluster Sayısı (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method For Optimal k")

plt.tight_layout()
plt.savefig(save_path, dpi=200)
plt.close()
print(f"Grafik buraya kaydedildi: {save_path}")

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.
# ------------------------------------------------------------------------------------------------------------------------------------------------
# optimal k değeri = 6, buna göre modeli fit ediyoruz
kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
kmeans.fit(kmeans_df)

# Her gözleme bir cluster numarası verir.
# 0.satır            cluster 0
# 1.satır            cluster 2
# 2.satır            cluster 1
df["cluster"] = kmeans.labels_
df.head()

# ---------------------------------------------------------------------
# PCA: Çok boyutlu veriyi 2 boyuta indirip görselleştirmek için kullanılır.
# ---------------------------------------------------------------------
# KMeans 5-6 feature ile çalışır, PCA ile 2 boyuta indirgeriz ----> böylece grafik çizebiliriz.
from sklearn.decomposition import PCA

# veriyi 2 ana bileşene indir ---> PC1, PC2 ---> sırasıyla x ve y
pca = PCA(n_components=2)

# veriyi öğrenir(fit) ve 2 boyuta indirir(transform)
# pca_comp = pca sonrası oluşan numpy array ---> içinde PC1 ve PC2 var
# bütün dataset ---> 2 feature'a indirildi
pca_comp = pca.fit_transform(kmeans_df)

save_path2 = r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning/ML_9.Hafta_Case_Study/cluster_pca.png"

plt.figure(figsize=(8,6))

for i in range(6):
    plt.scatter(
        pca_comp[df["cluster"] == i, 0],    # cluster i olan satırların PC1 değerleri  / 0 --> PC1 sütunu
        pca_comp[df["cluster"] == i, 1],    # cluster i olan satırların PC2 değerleri  / 1 --> PC2 sütunu
        label=f"Cluster {i}",
        alpha=0.6
    )


plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Customer Segments PCA Visualization")
plt.legend()

plt.tight_layout()
plt.savefig(save_path2, dpi=200)
plt.close()

print("Kaydedildi:", save_path2)

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Adım 4: Herbir segmenti istatistiksel olarak inceleyeniz.
# -----------------------------------------------------------------------------------------------------------------------------------------------------
# agg ---> istediğimiz sütunlara, istediğimiz işlemi uygular
def segment_summary(df, cluster_col="cluster"):
    summary = df.groupby(cluster_col).agg({
        "tenure": ["mean", "min", "max"],
        "recency": ["mean", "min", "max"],
        "total_order": ["mean", "min", "max"],
        "total_value": ["mean", "min", "max"],
        "avg_order_value": ["mean", "min", "max"],
        cluster_col: "count"
    })
    # cluster sütununun adını customer_count yap
    summary = summary.rename(columns={cluster_col: "customer_count"}).round(2)

    print("----- SEGMENT STATISTICAL ANALYSIS -----")
    return summary.reset_index()

    # groupby yaptığında cluster index olur, normal sütun olmaz
    # cluster normal sütun olsun isteriz
    # reset_index() ---> index, normal sütuna çevrilir
# ------------------------------------------------------------------------------------
# recency ---> düşük , yakın zamanda alışveriş yapmış iyi
# total_order ---> yüksek , sık alışveriş iyi
# total_value ---> yüksek , çok harcama iyi
# avg_order_value ---> yüksek , sepette pahalı ürün --> değerli


# ----------------------------------------------------------------------------------------------------------
# Evaluation Metrices
# -------------------------------------------------------------------------------------------------------------
# Silhouette Score (-1 ile 1 arası değerler ----> 1'e yakın, iyi clusterlama)
# Davies-Bouldin Score (cluster içi benzerlik, cluster arası fark ölçer ---> düşük, iyi clusterlama)
# Calinski-Harabasz Score (clusterlar arası mesafeyi ölçer ---> yüksek, iyi clusterlama)

from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def metrics(data, k_min=2, k_max=10):
    print("----- KMEANS CLUSTER METRICS -----\n")
    results = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)

        sil = silhouette_score(data, labels)
        db = davies_bouldin_score(data, labels)
        ch = calinski_harabasz_score(data, labels)

        results.append([k, sil, db, ch])

        print(f"k = {k}")
        print(f" Silhouette Score     : {sil:.4f}")
        print(f" Davies-Bouldin Score : {db:.4f}")
        print(f" Calinski-Harabasz    : {ch:.2f}")
        print("----------------------------------")

    # en iyi k bul
    best_k_sil = max(results, key=lambda x: x[1])[0]
    best_k_db = min(results, key=lambda x: x[2])[0]
    best_k_ch = max(results, key=lambda x: x[3])[0]

    print("\n----- BEST K -----")
    print(f"Best k (Silhouette)        : {best_k_sil}")
    print(f"Best k (Davies-Bouldin)    : {best_k_db}")
    print(f"Best k (Calinski-Harabasz) : {best_k_ch}")

metrics(kmeans_df, 2, 6)

# -----------------------------------------------------------------------------------------------------------------------------
# Görev 3: Hierarchical Clustering ile Müşteri Segmentasyonu
# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.
# ------------------------------------------------------------------------------------------------------------------------------

from scipy.cluster.hierarchy import linkage, dendrogram

sample_df = kmeans_df.sample(2000, random_state=42)
hc_average = linkage(sample_df, method="average")

save_path3 = r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning\Ml_9.Hafta_Case_Study\dendrogram_full.png"
save_path4 = r"C:\Users\ACER\Desktop\MIUUL-Data Scientist Bootcamp\ML_7_8_9_Hafta\machine_learning\Ml_9.Hafta_Case_Study\dendrogram_truncated.png"

# 1) full dendogram
plt.figure(figsize=(10,5))
plt.title("Hierarchical Clustering Dendrogram (sample=2000)")
plt.xlabel("Gözlemler")
plt.ylabel("Distance")
dendrogram(hc_average, leaf_font_size=6)
plt.tight_layout()
plt.savefig(save_path3, dpi=300)
plt.close()

# ----------------------------------------------------------------------------------------------------------------------

# 2) truncated dendogram
plt.figure(figsize=(7,5))
plt.title("Dendrogram (Truncated, sample=2000)")
plt.xlabel("Clusters")
plt.ylabel("Distance")
dendrogram(hc_average, truncate_mode="lastp", p=10, show_contracted=True, leaf_font_size=10)
plt.axhline(y=0.5, color="r", linestyle="--")
plt.axhline(y=0.6, color="b", linestyle="--")
plt.tight_layout()
plt.savefig(save_path4, dpi=300)
plt.close()

print("Kaydedildi:", save_path3)
print("Kaydedildi:", save_path4)

# -------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
# -------------------------------------------------------------------------------------------------------------------------------
# cluster = 6
from sklearn.cluster import AgglomerativeClustering

# hierarchical clusterin ---> 6 cluster, ward yöntemi
hc = AgglomerativeClustering(n_clusters=6, linkage="ward")
# modeli veriye uyguladı
# veriyi inceledi, benzer müşterileri buldu, 6 segmente ayırdı
df["cluster"] = hc.fit_predict(kmeans_df)  # cluster etiketlerini dfe ekledi
df.head()

# -------------------------------------------------------------------------------------------------------------------------------
# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz
# ----------------------------------------------------------------------------------------------------------------------------------

segment_summary(df,cluster_col="cluster")

