import numpy as np 
import pandas as pd 
import os 
import PIL.Image as img 

bakteri_yaprak_yanik = "rice_leaf_diseases/Bacterial leaf blight"
kahve_nokta = "rice_leaf_diseases/Brown spot"
yaprak_isi = "rice_leaf_diseases/Leaf smut"

def dosya(yol):
    return [os.path.join(yol,f) for f in os.listdir(yol)]

def veri_donusturme(klasor_adi,sinif_adi):
    goruntuler = dosya(klasor_adi)
    
    goruntu_sinif = []
    
    for goruntu in goruntuler:
        goruntu_oku= img.open(goruntu).convert('L')
        goruntu_boyutlandirma = goruntu_oku.resize((28,28))
        goruntu_donusturme = np.array(goruntu_boyutlandirma).flatten()
        
        if sinif_adi == "bakteri_yaprak":
            veriler = np.append(goruntu_donusturme, [0])
        
        elif sinif_adi == "kahve_nokta":
            veriler = np.append(goruntu_donusturme, [1])

        elif sinif_adi == "yaprak_isi":
            veriler = np.append(goruntu_donusturme, [2])
            
        else: 
            continue
        goruntu_sinif.append(veriler)
    return goruntu_sinif
    
yanik_veri = veri_donusturme(bakteri_yaprak_yanik,"bakteri_yaprak")   
yanik_veri_df = pd.DataFrame(yanik_veri)
 
kahve_nokta_veri = veri_donusturme(kahve_nokta,"kahve_nokta")   
kahve_nokta_veri_df = pd.DataFrame(kahve_nokta_veri)

yaprak_isi_veri = veri_donusturme(yaprak_isi,"yaprak_isi")   
yaprak_isi_df = pd.DataFrame(yaprak_isi_veri)

tum_veri = pd.concat([yanik_veri_df, kahve_nokta_veri_df, yaprak_isi_df ])

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV



Giris = np.array(tum_veri)[:,:784]
Cikis = np.array(tum_veri)[:,784]

Giris_train, Giris_test, Cikis_train, Cikis_test = train_test_split(Giris,Cikis,test_size=0.2,random_state=109)


agac_parametreleri = {'criterion':['gini','entropy'],'max_depth':[2,5,10,20,30,100,150] }




model = DecisionTreeClassifier()





tahminmodel = model.fit(Giris_train,Cikis_train)

Cikis_pred = tahminmodel.predict(Giris_test)

print("Dogruluk : " , metrics.accuracy_score(Cikis_test,Cikis_pred))


import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle
Cikis_test = label_binarize(Cikis_test, classes=[0, 1, 2])
Cikis_pred = label_binarize(Cikis_pred, classes=[0, 1, 2])

plt.figure(figsize=(60, 40),dpi=150)
n_classes=3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Cikis_test[:, i], Cikis_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, 
             label=' {0} Sınıfına ait ROC eğrisi (AUC = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()









    


