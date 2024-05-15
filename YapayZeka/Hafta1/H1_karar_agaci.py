from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.target)
print(iris.data)

cikis = iris.target
giris = iris.data

from sklearn.model_selection import train_test_split

giris_egitim, giris_test, cikis_egitim,cikis_test = train_test_split(giris,cikis,test_size=0.33,random_state =109)

print("Egitim veri seti boyutu =", len(giris_egitim))
print("Test veri seti boyutu =", len(cikis_test))

from sklearn.tree import DecisionTreeClassifier
#from sklearn.neighbors import KNeighborsClassifier

model = DecisionTreeClassifier()
#model = KNeighborsClassifier()

model.fit(giris_egitim,cikis_egitim)

cikis_tahmin = model.predict(giris_test)

from sklearn.metrics import confusion_matrix

hata_matrisi = confusion_matrix(cikis_test,cikis_tahmin)

print(hata_matrisi)

import matplotlib.pyplot as plt 
import pandas as pd

index = ['setosa','versicolor','virginica']
columns = ['setosa','versicolor','virginica']

hata_goster = pd.DataFrame(hata_matrisi,index)
plt.figure(figsize=(10,6))
plt.imshow(hata_goster, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Hata Matrisi')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gercek Etiket')
plt.show()

import seaborn as sns
sns.heatmap(hata_goster,annot=True)

