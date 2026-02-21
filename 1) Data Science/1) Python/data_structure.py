from statsmodels.sandbox.regression.example_kernridge import upper

# veri yapılarına giriş
# numbers: int, float, complex
# karakter dizileri (Strings) : str
# boolean ( T-F ) : bool
# liste ( list )
# sözlük ( dictionary )
# demet ( tuple )
# set



# 1) VERİ YAPILARINA GİRİŞ

#int
x = 46
type(x)

#float
y = 9.4
type(y)

# complex
x = 2j + 1
type(x)

# string
x = "hello ai era"
type(x)

#boolean
x = True
type(x)
False
5 == 4
1 == 1
type(2 == 4)

############################################################

# 2) LIST
# mutable
# oredered
# farklı türde öge içerebilir
# duplicates allowed

x = [1, 2, 3]
type(x)

# 3) DICTIONARY
# key-value pairs
# mutable
# hılı erişim
# 3.7 + sürümlerinde sıralı


x = {"name":"Miuul", "age":4}
type(x)

# 4) TUPLE
# immutable
# ordered
# hızlıdır
# listenin aksi huysuz kardeşidir.
# farklı türde ögeler içerebilir.
# duplicates allowed
# genelde python fonksiyonları birden fazla değeri
# bir arada döndürmek için tuple kullanır.

x = (1, 2, 3)
y = ("python", "ml", "ds")
type(x)
type(y)

# 5) SET
# not duplicates values
# unordered
# mutable

x = {"N", "M", "A"}
type(x)


# List, tuple, set ve dictionary veri yapıları
# aynı zamanda Python Collections( Arrays ) olarak geçmektedir.
########################################################################3
# 6) karaktger dizileri ( strings )

print("dudu")
name = "dudu"
name

# 7) çok satırlı karakter dizileri

""" dudu """
# 8 ) karakter dizilerinin elemanlarına erişmek

name[0]
name[3]
name[0:3]

# karakter dizilerinde eleman sorgulama

"veri" in name

# string ( karakter dizisi ) metodları

# metod : classlar içinde taımlanan fonksiyonlardır
dir(int)
dir(str)


# eğer bir fonksiyon class yapısı içrisinde tanımlandıysa methoddur, değilse fonksiyondur
# fonksiyonlar bağımsız, methodlar ise class yapısına bağlıdır
# len = stringlerin uzunluğu
name = "lk"
type(name)
type(len)
len(name)
len("vahit")

# upper(), lower()

"dudu".upper()
"KSLDM".lower()

type(upper)

# replace

hi = "dudu"
hi.replace("d","k")

# split

"dudu kkkk".split()

# stirp = kırma methodu

" dudu kkkk ".strip("u")

# capitalize

"dudu".capitalize()

# dir = hangi methodları kullanacagımıza bakabiliriz

# starts with

"dudu".startswith("d")

# LIST
# mutable ----> değiştirilebilir
# ordinered ----> index işlemleri yapılabilir.
# kapsayıcıdır ---> içerisinde birden fazla veri yapısını tutabilir.

notes = [90,40,55,60]
type(notes)

names = ["a","b","c","d"]
type(names)

names = ["a","b","c","d","e","f", True , [1,2,3]]
type(names)

names[0]
names[7][1]
names[0] = 99
names
names[0:4]

# liste metodları

dir(names)
names.append(22)
len(names)

# pop = indexe göre eleman siler
names.pop(0)

names

# insert = indexe göre ekler

names.insert(1,32)
names

# dictionary ( sözlük )
# mutable
# not ordered ( +3.7 sonrası sıralı )
# kapsayıcı ------ > birden fazla veri yapısı içerebilir.
# key-value

name ={"eng":"dudu",
       "dent":"hasan",
       "teach":"elif"}


name
name["eng"]

name = { "child":["elif","hasan","dudu"],
         "major":["teacher","dentist","engineer"],
         "age":[25,21,23]}

name["child"]
name["major"]
name["age"]
name["age"][1]

# key sorgulama
# in ile sadece keylerin dictionaryde olup olmadıgına bakabiliriz
"child" in name
"elif" in name

# key'e göre valu'a erişmek
name["child"]
name.get("child")
name["major"] = ["a","b","c"]
name["major"]

name

# tüm keylere erişmek

name.keys()
name.values()

# tüm çiftleri tuple halinde listeye çevirme

name.items()

# key-value değerlerini update etmek
name
name.update({"age":[12,23,33]})
name.get("age")

name.update({"gender":["female","male"]})
name


# Demet(Tuple)
# tuple listelerin değişime kapalı kardeşidir
# immutable ------> tuple içindeki elemanlar değiştirilemez.
# ----------> ama list() ile listeye çevirip değiştirip elemanı
# sonra tekrar tuple'a dönüşütürüp yapabiliriz.
# ordered -----> sıralı indexle erişebiliyoruz
# kapsayıcıdır -----> birden farklı veri yapısını tutabiliyor, saklayabiliyor


t = ("a","mom",1,3)
type(t)
t[0:3]

t = list(t)
t[0] = 99
t = tuple(t)
t


# SET
# mutable
# unordered + unique
# kapsayıcıdır
# hız gerektiren işlemlerde ve küme işlemlerinde kullanılır.
# list üzerinden oluşturuyoruz seti
# köşeli parantez gördüğümüzde list oldugunu bilcez

#difference() : iki kümenin farkı
set1 = set([1,2,3,4])
set2 = set([3,4,5,6])
set1-set2
set2-set1
type(set1)

#set1'de olup set2'de olmayanlar
set1.difference(set2)

#set2de olup set1de olmayanlar
set2.difference(set1)

# tümü - kesişim
# symmetric_difference() : iki kümede de birbirlerine göre olmayanları verir

set1.symmetric_difference(set2)
set2.symmetric_difference(set1)

# intersection

set1.intersection(set2)
set2.intersection(set1)

set1 & set2

# union

set1.union(set2)
set2.union(set1)

# isdisjoint() : iki kümenin kesişimi boş mu, değil mi ?
# is ile başlayan methodlar True ya da False dönecektir

set1.isdisjoint(set2)
set2.isdisjoint(set1)

# issubset() : o küme diğerinin alt kümsei mi ?

set1.issubset(set2)
set2.issubset(set1)

# issuperset() : bir küme diğer kümeyi kapsıyor mu ?

set2.issuperset(set1)
set1.issuperset(set2)






