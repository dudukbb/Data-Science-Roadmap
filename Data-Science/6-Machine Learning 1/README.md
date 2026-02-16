# 🧠 Machine Learning Model Türleri – Kısa Özet

Makine öğrenmesi problemleri genel olarak **supervised** , **unsupervised** ve **reinforcement learning** olmak üzere üç ana gruba ayrılır.

---

## 🔹 1. Supervised Learning (Denetimli Öğrenme)

Bu öğrenme türünde veri setinde bir **target (y)** değişkeni bulunur.  
Amaç: Bağımsız değişkenleri (X) kullanarak target değişkeni tahmin etmektir.

Supervised learning ikiye ayrılır:

### 📊 Regression
Target değişken **sayısal (numeric)** ise kullanılır.

**Örnek problemler:**
- Maaş tahmini (Salary)
- Ev fiyatı tahmini
- Satış tahmini

**Regression modelleri:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- SVR (Support Vector Regression)
- KNN Regressor

Bu modeller sayısal değer tahmini yapar.

**Regression evaluation metrikleri:**
- RMSE (en yaygın)
- MSE
- MAE
- R2

---

### 📊 Classification
Target değişken **kategorik** ise kullanılır.

**Örnek problemler:**
- Müşteri churn tahmini (0/1)
- Spam mail sınıflandırma
- Hastalık var/yok tahmini

**Classification modelleri:**
- Logistic Regression
- KNN Classifier
- SVC (Support Vector Machine)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- XGBoost Classifier
- LightGBM Classifier
- Naive Bayes

Bu modeller sınıf tahmini yapar.

**Classification evaluation metrikleri:**
- Accuracy
- Precision
- Recall
- F1-score 
- ROC-AUC

Dengesiz veri varsa F1-score ve Recall tercih edilir.

---

## 🔹 2. Unsupervised Learning (Denetimsiz Öğrenme)

Bu öğrenme türünde **target değişken yoktur**.  
Amaç: Veri içerisindeki gizli yapıları ve örüntüleri keşfetmektir.

**Unsupervised modeller:**
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- PCA (Principal Component Analysis)
- Association Rule Learning

**Unsupervised evaluation metrikleri**
- Silhouette Score (Clustering)
- Inertia (K-Means)
- Explained Variance (PCA)


---

## 🔹 Hızlı Ezber Formülü

- Target numeric → **Regression**
- Target kategorik → **Classification**
- Target yok → **Unsupervised**

---


