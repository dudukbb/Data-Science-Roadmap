
"""
----------------------------- PYTHON ALIŞTIRMALAR -----------------------------------

Python görevlerini tamamlayınız.

Görev 1: Verilen değerlerin veri yapılarını inceleyiniz. Type() metodunu kullanınız.

Görev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz,
kelime kelime ayırınız.

Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.
Adım 1: Verilen listenin eleman sayısına bakınız.
Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
Adım 4: Sekizinci indeksteki elemanı siliniz.
Adım 5: Yeni bir eleman ekleyiniz.
Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.

Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
Adım 1: Key değerlerine erişiniz.
Adım 2: Value'lara erişiniz.
Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
Adım 5: Antonio'yu dictionary'den siliniz.

Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu listeleri
return eden fonksiyon yazınız.
Liste elemanlarına tek tek erişmeniz gerekmektedir.
Her bir elemanın çift veya tek olma durumunu kontrol etmekiçin % yapısını kullanabilirsiniz.

Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri
bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de
tıp fakültesi öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.

Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer
almaktadır. Zip kullanarak ders bilgilerini bastırınız.

Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
Kapsayıp kapsamadığını kontrol etmek için issuperset() metodunu,
farklı ve ortak elemanlar için ise intersection ve difference metodlarını kullanınız.

"""

# -------------------------------------------------------------------------------------------------------------------------------
# Görev 1: Verilen değerlerin veri yapılarını inceleyiniz. Type() metodunu kullanınız.
# -------------------------------------------------------------------------------------------------------------------------------
x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
c = 23<22
l = [1,2,3,4]
d = {"Name":"Jake",
     "Age":27,
     "Adress":"Downtown"}
t = ("Machine Learning","Data Science")
s = {"Python","Machine Learning","Data Science"}
# -------------------------------------------------------------------------------------------------------------------------------

type(x)
type(y)
type(z)
type(a)
type(b)
type(c)
type(l)
type(d)
type(t)
type(s)

# -------------------------------------------------------------------------------------------------------------------------------
# Görev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz.
# Virgül ve nokta yerine space koyunuz, kelime kelime ayırınız.
# -------------------------------------------------------------------------------------------------------------------------------

text = "The goal is to turn data into information, and information into insight."

text = text.upper()   # büyük harf
text = text.replace(","," ")
text = text.replace("."," ")
text
texts = text.split()
texts

# -------------------------------------------------------------------------------------------------------------------------------
# Görev 3: Verilen listeye aşağıdaki adımları uygulayınız.
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 1: Verilen listenin eleman sayısına bakınız.
# -------------------------------------------------------------------------------------------------------------------------------
lst = ["D","A","T","A","S","C","I","E","N","C","E"]
len(lst)
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
# -------------------------------------------------------------------------------------------------------------------------------
lst[0]
lst[10]
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 3: Verilen liste üzerinden ["D", "A", "T", "A"] listesi oluşturunuz.
# -------------------------------------------------------------------------------------------------------------------------------
new_lst = lst[0:4]
new_lst

# -------------------------------------------------------------------------------------------------------------------------------
# Adım 4: Sekizinci indeksteki elemanı siliniz.
# -------------------------------------------------------------------------------------------------------------------------------
lst.pop(7)
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 5: Yeni bir eleman ekleyiniz.
# -------------------------------------------------------------------------------------------------------------------------------
lst.append("p")
lst
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 6: Sekizinci indekse "N" elemanını tekrar ekleyiniz.
# -------------------------------------------------------------------------------------------------------------------------------
lst.insert(7,"N")
lst

# -------------------------------------------------------------------------------------------------------------------------------
# Görev 4: Verilen sözlük yapısına aşağıdaki adımları uygulayınız.
# -------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 1: Key değerlerine erişiniz.
# -------------------------------------------------------------------------------------------------------------------------------
dict = { 'Christian': ["America", 18],
         'Daisy' : ["England", 12],
         'Antonio' : ["Spain", 22],
         'Dante' : ["Italy", 25]}
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 2: Value'lara erişiniz.
# -------------------------------------------------------------------------------------------------------------------------------
dict.keys()
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
# -------------------------------------------------------------------------------------------------------------------------------
dict["Daisy"][1] = 13
dict
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
# -------------------------------------------------------------------------------------------------------------------------------
dict.update({"Ahmet" : ["Turkey", 24]})
dict
# -------------------------------------------------------------------------------------------------------------------------------
# Adım 5: Antonio'yu dictionary'den siliniz.
# -------------------------------------------------------------------------------------------------------------------------------
dict.pop("Antonio")
dict

# -------------------------------------------------------------------------------------------------------------------------------
# Görev 5: Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları
# ayrı listelere atayan ve bu listeleri return eden fonksiyon yazınız.
# Liste elemanlarına tek tek erişmeniz gerekmektedir.
# Her bir elemanın çift veya tek olma durumunu kontrol
# etmek için % yapısını kullanabilirsiniz.
# -------------------------------------------------------------------------------------------------------------------------------

l = [2,13,18,93,22]
even_list = []
odd_list = []

def func():
     for i in l:
         if i%2 == 0:
              even_list.append(i)
         else:
              odd_list.append(i)

     return even_list, odd_list

even_list, odd_list = func()

print("Çift :",even_list)
print("Tek :",odd_list)

# -------------------------------------------------------------------------------------------------------------------------------
# Görev 6: Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri
# bulunmaktadır. Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de
# tıp fakültesi öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.
# -------------------------------------------------------------------------------------------------------------------------------

ogrenciler = ["Ali","Veli","Ayşe","Talat","Zeynep","Ece"]

"""
Beklenen Çıktı
---------------------------------------------------
Mühendislik Fakültesi 1 . öğrenci: Ali
Mühendislik Fakültesi 2 . öğrenci: Veli
Mühendislik Fakültesi 3 . öğrenci: Ayşe
Tıp Fakültesi 1 . öğrenci: Talat
Tıp Fakültesi 2 . öğrenci: Zeynep
Tıp Fakültesi 3 . öğrenci: Ece
----------------------------------------------------
"""

ogrenciler_müh = ogrenciler[0:3]
ogrenciler_müh

for i, j in enumerate(ogrenciler_müh, start=1):
     print(f"Mühendislik Fakültesi {i}. öğrenci: {j} ")

ogrenciler_tıp = ogrenciler[3:6]

for i, j in enumerate(ogrenciler_tıp,start = 1):
     print(f"Tıp Fakültesi {i}. öğrenci: {j} ")

# -------------------------------------------------------------------------------------------------------------------------------
# Görev 7: Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri yer
# almaktadır. Zip kullanarak ders bilgilerini bastırınız.
# -------------------------------------------------------------------------------------------------------------------------------
ders_kodu = ["CMP1005","PSY1001","HUK1005","SEN2204"]
kredi = [3,4,2,4]
kontenjan = [30,75,150,25]

"""
Beklenen Çıktı
----------------------------------------------------------
Kredisi 3 olan CMP1005 kodlu dersin kontenjanı 30 kişidir.
Kredisi 4 olan PSY1001 kodlu dersin kontenjanı 75 kişidir.
Kredisi 2 olan HUK1005 kodlu dersin kontenjanı 150 kişidir.
Kredisi 4 olan SEN2204 kodlu dersin kontenjanı 25 kişidir.

"""

for i , j , k in zip(ders_kodu,kredi,kontenjan):
     print(f"Kredisi {j} olan {i} kodlu dersin kontenjanı {k} kişidir.")


# -------------------------------------------------------------------------------------------------------------------------------
# Görev 8: Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1. küme 2. kümeyi kapsiyor ise ortak elemanlarını
# eğer kapsamıyor ise 2. kümenin 1. kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.
# Kapsayıp kapsamadığını kontrol etmek için issuperset() metodunu,
# farklı ve ortak elemanlar için ise intersection ve difference metodlarını kullanınız.
# -------------------------------------------------------------------------------------------------------------------------------

kume1 = set(["data","python"])
kume2 = set(["data","function","qcut","lambda","python","miuul"])

def func(a,b):
     if kume1.issuperset(kume2):
          print(kume1.intersection(kume2))
     else:
          print(kume2.difference(kume1))

func(kume1,kume2)

"""
Beklenen Çıktı
----------------------------------------------
{"function" , "qcut" , "miuul" , "lambda"}
"""