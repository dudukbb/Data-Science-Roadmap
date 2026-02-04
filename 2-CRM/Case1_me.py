import pandas as pd
import datetime as dt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)  # yanyana aşağıya şnmeden 500 sütunu göster
pd.set_option('display.max_rows',None)
pd.set_option('display.float_format',lambda x: '%.3f' % x)

### GÖREV 1: Veriyi anlama ve hazırlama
# ADIM 1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz
df_ = pd.read_csv("CRM_3.Hafta_Case_Study/Case1/FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy()
df.head()
df

# ADIM 2: Veri setinde
# a. İlk 10 gözlem
df.head(10)

# b. Değişken isimleri
df.columns
df.columns.tolist()

# c. Betimsel istatistik
df.describe().T

# d. Boş değer
df.isnull().sum()

# e. Değişken tipleri incelemesi yapınız
df.dtypes


# ADIM 3:Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["total_order_sum"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()
# ADIM 4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes
date_cols = ["first_order_date",
             "last_order_date",
             "last_order_date_online",
             "last_order_date_offline"]
df[date_cols] = df[date_cols].apply(pd.to_datetime)
df.dtypes

# ADIM 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.groupby('order_channel').agg({
    'master_id':"nunique",
    'total_order_sum':"sum",
    'total_customer_value':"sum"
})
# ADIM 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.sort_values("total_customer_value",ascending=False).head(10)

# ADIM 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values("total_order_sum",ascending=False).head(10)

# ADIM 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_prep(df):
    df["total_order_sum"] = df["order_num_total_ever_offline"] + df["order_num_total_ever_online"]
    df["total_customer_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    date_cols = df.columns[df.columns.str.contains("date")]
    df[date_cols] = df[date_cols].apply(pd.to_datetime)
    return df

df = data_prep(df)
df.head()

###GÖREV 2: RFM Metriklerinin Hesaplanması
#Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.

# Recency = Müşterinin en son alım yaptığı dönemden bu yana geçen süreye denir.
# Frequency = Müşterinin yaptığı toplam alışveriş sayısına denir
# Monetary = Müşterilerin bize bıraktığı parasal değere denir.

#Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.

today_date = dt.datetime(2026,1,1)
rfm = df.groupby("master_id").agg({
    "last_order_date": lambda x: (today_date-x.max()).days,
    "total_order_sum" : "sum",
    "total_customer_value":"sum",
})
rfm.columns = [ "recency", "frequency","monetary"]
rfm.head()
#Adım 3: Hesapladığınız metrikleri rfm isimli bir değişkene atayınız.

today_date = dt.datetime(2026,1,1)
rfm = df.groupby("master_id").agg({
    "last_order_date": lambda x: (today_date-x.max()).days,
    "total_order_sum" : "sum",
    "total_customer_value":"sum",
})


#Adım 4: Oluşturduğunuz metriklerin isimlerini recency, frequency ve monetary olarak değiştiriniz.
rfm.columns = [ "recency", "frequency","monetary"]
rfm.head()

### GÖREV 3: RF Skorunun Hesaplanması
#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
#Adım 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.
rfm["recency_score"] = pd.qcut(rfm['recency'],5,labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"),5,labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm["monetary"],5,labels=[1,2,3,4,5])
rfm.head()

#Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str)
rfm.head()

#Görev 4: RF Skorunun Segment Olarak Tanımlanması
#Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
#Adım 2: Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.
# RFM_SCORE değerlerini, seg_map dictionary'e göre regex kullanarak
# segment isimlerine dönüştürüyor.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm["segment"] = rfm["RF_SCORE"].replace(seg_map,regex=True)
rfm.head()

### GÖREV 5: Aksiyon Zamanı !
# ADIM 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm.groupby("segment").agg({
    "recency":"mean",
    "frequency":"mean",
    "monetary":"mean"
})

# Adım 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri idlerini csv olarak kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

loyal_cust_ids = rfm[rfm["segment"].isin(["champions", "loyal_customers"])].index
women_loyal_cust_ids = df[
    (df["master_id"].isin(loyal_cust_ids)) &
    (df["interested_in_categories_12"].str.contains("KADIN",na=False))
    ]["master_id"]

women_loyal_cust_ids.to_csv("loyal_champions_ID.csv", index=False)
# loyal_champions_ID.csv

# b. Erkek ve Çocuk ürünlerinde %40a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin idlerini csv dosyasına kaydediniz.

customer_ids = rfm[rfm["segment"].isin(["hibernating", "cant_loose", "new_customers"])].index
target_customer_ids = df[(df["master_id"].isin(customer_ids)) & df["interested_in_categories_12"].str.contains("ERKEK|COCUK")]["master_id"]

target_customer_ids.to_csv("indirim_hedef_müşteri_ids.csv", index=False)
