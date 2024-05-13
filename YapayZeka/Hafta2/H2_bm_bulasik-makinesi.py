import numpy as np
import skfuzzy as bulanik 
from skfuzzy import control as kontrol

bulasik_miktari = kontrol.Antecedent(np.arange(0,100,1), 'bulaşık miktari' )
kirlilik = kontrol.Antecedent(np.arange(0,100,1), 'kirlilik' )

yikama = kontrol.Consequent(np.arange(0,180,1), 'yikama')


bulasik_miktari['az'] = bulanik.trimf(bulasik_miktari.universe, [0,0,30])
bulasik_miktari['normal'] = bulanik.trimf(bulasik_miktari.universe, [10,30,60])
bulasik_miktari['cok'] = bulanik.trimf(bulasik_miktari.universe, [50,60,100])
kirlilik['az'] = bulanik.trimf(kirlilik.universe, [0,0,30])
kirlilik['normal'] = bulanik.trimf(kirlilik.universe, [10,30,60])
kirlilik['cok'] = bulanik.trimf(kirlilik.universe, [50,60,100])

yikama['kisa'] = bulanik.trimf(yikama.universe, [0,0,50])
yikama['normal'] = bulanik.trimf(yikama.universe, [40,50,100])
yikama['uzun'] = bulanik.trimf(yikama.universe, [80,120,180])

kural1 = kontrol.Rule(bulasik_miktari['az'] & kirlilik['az'] , yikama['kisa'] )
kural2 = kontrol.Rule(bulasik_miktari['normal'] & kirlilik['az'] , yikama['normal'] )
kural3 = kontrol.Rule(bulasik_miktari['cok'] & kirlilik['az'] , yikama['normal'] )
kural4 = kontrol.Rule(bulasik_miktari['az'] & kirlilik['normal'] , yikama['normal'] )
kural5 = kontrol.Rule(bulasik_miktari['normal'] & kirlilik['normal'] , yikama['uzun'] )
kural6 = kontrol.Rule(bulasik_miktari['cok'] & kirlilik['normal'] , yikama['uzun'] )
kural7 = kontrol.Rule(bulasik_miktari['az'] & kirlilik['cok'] , yikama['normal'] )
kural8 = kontrol.Rule(bulasik_miktari['normal'] & kirlilik['cok'] , yikama['uzun'] )
kural9 = kontrol.Rule(bulasik_miktari['cok'] & kirlilik['cok'] , yikama['uzun'] )


sonuc = kontrol.ControlSystem([kural1, kural2,kural3,kural4,kural5,kural6,kural7,kural8,kural9]  )
                         
model_sonuc = kontrol.ControlSystemSimulation(sonuc)

model_sonuc.input['bulaşık miktari'] = int(input('Bulasik miktarini giriniz : '))
model_sonuc.input['kirlilik'] = int( input('Kirlilik miktarini giriniz : '))





model_sonuc.compute()

print(model_sonuc.output['yikama'] )

