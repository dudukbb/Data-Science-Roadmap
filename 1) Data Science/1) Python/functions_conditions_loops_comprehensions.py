from win32com.servers import dictionary

# function
# conditions
# loops
# comprehensions

# 1) functions
# belirli görevleri yerine getirmeye çalışan kod parçacıklarıdır

# foksiyon okuryazarlığı
# paremetre : fonsksiyon tanımlanması esansında ifade edilen değişkenlerdir
# argüman : bu fonksiyonlar çağırılıdığında bu parametre değerlerine karşılık
# girilen değerlerdir
# yaygın kullanım hepsine argüman denmesi

print("a","b",sep="__")

# function definiton

def calculate(x):
    print(x*2)

calculate(2)

# iki argümanlı bir function define et

def addition(num1,num2):
    print(num1+num2)

addition(num2=2,num1=3)

def addition(num1,num2):
    """
    Sum of two numbers

    Args
    ---------------

    arg1: int,float

    arg2: int,float

    Returns
    --------------
    """
    print(num1+num2)

# functionların body/ statement bölümü

def say_hi(string):
    print(string)
    print("hi")
    print("hello")
    print("merhaba")

say_hi("miuul")

def multiplication(a,b):
    c = a*b
    print(c)

multiplication(2,5)


# girilen değerleri bir liste içinde saklayacak fonksiyon

list_store = []

def add(a,b):
    c = a*b
    list_store.append(c)
    print(list_store)

add(10,2.4)
add(8,7)


# ön tanımlı argümanlar / parametreler ( default parameters / arguments )
def division(a,b):
    print(a/b)

division(10,2)

def division(a,b=3):
    print(a/b)

say_hi("miuul")

# dont repeat yourself
# ne zaman fonksiyon yazma ihtiyacımız olur ?

def calculate(x,y,z):
    print((x+y)/z)

calculate(2,3,4)


# return
# fonksiyon çıktılarını girdi olarak kullanmak için kullanılan yapıdır

def calculate(x,y,z):
    x = x*2
    y = y*3
    z = z*4
    output = (x+y)/z
    return x,y,z,output

calculate(2,3,4)
type(calculate(2,3,4))

# fonksiyon içerisinden fonksiyon çağırmak

def calculate(x,y,z):
    return int((x+y)/z)

calculate(2,3,4)*10

def div(x,y,z,l):
    k = calculate(x,y,z)
    n = division(k,l)
    print(n*10)


div(2,3,4,5)



# lokal ve globaql değişkenler
# list oldugunu köşeli parantezlerden anlıyoruz

list_store = [1,2]
type(list_store)

def add(a,b):
    c = a*b
    list_store.append(c)
    print(list_store)

add(10,45)
add(32,34)


# KOŞULLAR ( CONDITIONS )

# bir program yazımında akış kontrolü sağlayan ,
# programların nasıl hareket etmesi gerektipğini vs sağlayan yapılardır

# if
if 2 == 1:
    print("something")

num = 10

if num==10:
    print("svav")

def num_check(num):
    if num>10:
        print("greater than 10")
    elif num<10:
        print("less than 10")
    else:
        print("equal to 10")

num_check(10)

# loops

# for loop

students = ["john","mark","john"]
students[0]

for student in students:
    print(student.upper())

salaries = [70,80,90]
for salary in salaries:
    salary = (int(salary*(1/5) + salary))
    print(salary)

def salaryt(salary,rate):
    return int(salary*rate/100 + salary)


salaryt(1500,10)

for salary in salaries:
    print(salaryt(salary,10))


for salary in salaries:
    if salary>3000:
        print()

range(len("dudu"))

for i in range(0,5):
    print(i)
# girilen stringin çift indexlerini büyük harf yap
# mülakat sorusu
def alternating(str):
    new_string = ""
    # girilen stringin indexlerinde gez
    for i in range(len(str)):
        # i çift ise büyük harfe çevir
        if i%2==0:
           new_string += str[i].upper()
           # i tek ise küçük harfe çevir
        else:
           new_string += str[i].lower()
    print(new_string)


alternating("dudu kabakçı lokma ")


# brak , continue , while
# break : aranan koşul sağlandıgında döngünün durmasını sağlar
salaries = [70,80,90]

for salary in salaries:
    if salary == 81:
        break
        print(salary

# contiune : aranan koşul sağlandıgında devam et atla , o kısmı çalıştırma altındaki


for salary in salaries:
    if salary == 80:
        continue
    print(salary)




# while : -dığı sürece yap
num = 6
while num<5:
    print(num)
    num += 1

# enumerate : otomatik counter indexer with for loop
# hem ilgili objeyi hem de indexini temsil eder

students = ["john","mark","tohn"]
for student in students:
    print(student)

even_list = []
odd_list = []

for index,student in enumerate(students):
    if index %2 ==0:
        even_list.append(student)
    else:
        odd_list.append(student)


print(even_list)
print(odd_list)

# enumerate : mülakat sorusu

# divide_students foknsiyonu yazınız
# çift indexte yer alan öğrencileri bir listeye alınız.
# tek indexte yer alan öğrencileri başka bir listeye yazınız.
# fakat bu iki liste tek bir liste olarak return olsun.

S = ["hasan","dudu","elif","elveda","sefer"]
def divide_students(S):
    groups = [[],[]]
    for index, student in enumerate(S):
        if index % 2 == 0:
            groups[0].append(student)
        else:
            groups[1].append(student)
    print(groups)
    return groups

divide_students(S)

def alternating()


#zip : birbirinden farklı olan listeleri
# bir arada değerlendirme imkanı sağlar
# tuple formunda

students = ["john","mark","tohn"]
departments = ["programming","science"]
ages = [23,36,26,22]

list(zip(students,departments,ages))

# lambda , map , filter , reduce

# lambda : kullan at fonksiyondur

def add(a,b):
    return a+b


new_sum = lambda a, b: a + b
new_sum(4,5)


# map : bizi döngü yazmaktan kurtarır
# içerisinde iteratif gezebileceğim bir nesne ver ve uygulanacvk fonksiyonu ver
# ben otomatik yaparım der

salaries = [1000,2000,3000,4000,5000]
def new_salary(x):
    return x*20/100 + x

for salary in salaries:
    print(new_salary(salary))

list(map(new_salary, salaries))

list(map(lambda x: x**2 + x, salaries))
del new_sum

# filter : filtreleme işlemleri için kullanılır

list = [1,2,3,4,5]
list(filter(lambda x:x%2 == 0, list))

# Reduce : indirgemek

from functools import reduce
list_store = [1,2,3,4,5]
reduce(lambda a, b : a+b , list_store)


# comprehensions
# birden fazla satır ve kod ile yapılabilecek işlemleri
# kolayca istediğimiz çıktı veri yapısına göre
# tek bir satırda gerçekleştirmemizi sağlayan yapılardır

# list comprehension ( önemli )
# köşeli parantez içinde yani sonuç liste halinde çıkacak
# köşeli parantez içinde function loop vs uygularız tek satırda
salaries = [1000,2000,3000,4000,5000]

def new_salary(x):
    return x*20/100 + x

new_salary = []
for salary in salaries:
    if salary >3000:
        new_salary.append((new_salary(salary))
    else:
        new_salary.append(new_salary(salary*2))

for s in new_salary:
    print(s)

# list comprehensionda sadece if kullanacaksak önce for sonra if,
# eğer if else kullanacaksak önce if else sonra for
        # if koşula bak , uyuyorsa koşula solundakini yap
        # else koşuluna uyuyor , o zaman else'in sağındakini yap

# maaşları 2 ile çarp
[salary * 2 for salary in salaries ]

# maaşı 3000'den küçük olanları 2 ile çarp
[ salary * 2 for salary in salaries if salary < 3000]
[salary * 2 if salary < 3000 else salary*0 for salary in salaries]

[new_salary(salary*2) if salary < 3000 else new_salary(salary * 0.2) for salary in salaries]

students1 = ["john","mark","tohn"]

students2 = ["dudu","hasan"]

[student.upper() if student in students1 else student.upper() for student in students1 ]

# dict comprehension

dictionary = { "a": 1
        , "b": 2
        , "c": 3
        , "d": 4}

dictionary.keys()
dictionary.values()
dictionary.items()

{ k : v ** 2 for (k, v) in dictionary.items() }
{ k .upper():v for (k,v) in dictionary.items()}

{ k.upper() : v**2 for (k,v) in dictionary.items() }

# mülakat sorusu ( dict comprehensions )

# amaç: elimizdeki listede çift sayıların karesi alınarak
# bir sözlüğe eklenmek istenmektedir.
# keyler orijinal değerler, valuelar ise değiştirilmiş değerler olacak
# {} sözlük , [] list
numbers = range(10)

new_dict = {}
for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n**2

{ n : n**2 for n in numbers if n %2 == 0}

# list & dict comprehensions uygulamalar
# bir veri setindeki değişken isimlerini değiştirmek

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

A = []

for col in df.columns:
    A.append(col.upper())

df.columns = A

df = sns.load_dataset("car_crashes")
# df.columnsda gez ve değişkenlerin isimlerini büyüt
df.columns = [ col.upper() for col in df.columns ]

# isminde "INS" olan değişkenlerin başına FLAG diğerlerine NO_FLAG ekle

[ col for col in df.columns if "INS" in col ]
df.columns = [ "FLAG_"+ col if "INS" in col else "NO_FLAG" + col for col in df.columns]

# AMAÇ: key'i string ,value'su aşağıdaki gibi
# bir liste olan sözlük oluşturmak
# sadeec sayısal ddeğişkenler için bu illkemi yapmak istiyorux

# Output:
# { 'total': [ 'mean','min','max','var']
   #  'speeding': [ 'mean','min','max','var']}

# dict comprehension example

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

num_cols = [ col for col in df.columns  if df[col].dtype != "object"]
dic = {}
agg_list = ["mean", "median", "mode","min", "max","sum"]

 # uzun yol
for col in num_cols:
    dic[col] = agg_list
# kısa yol
new_dict = { col: agg_list for col in num_cols}

df[num_cols].head()
df[num_cols].agg(new_dict)






