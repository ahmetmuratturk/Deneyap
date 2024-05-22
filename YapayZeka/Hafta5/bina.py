# Gerekli kütüphaneleri import et
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Veriyi yükle
dataset = pd.read_excel('ENB2012_data.xlsx')
dataset = dataset.values
X = dataset[:, 0:8]
y = dataset[:, 8:10]

# Eğitim ve test verilerini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Verileri ölçekle
sc = StandardScaler()
data_x_train_scaled = sc.fit_transform(X_train)
data_x_test_scaled = sc.transform(X_test)

data_x_train_scaled, data_x_test_scaled, data_y_train, data_y_test = \
    np.array(data_x_train_scaled), np.array(data_x_test_scaled), np.array(y_train), \
    np.array(y_test)

data_y_train = (data_y_train[:, 0], data_y_train[:, 1])

# Modelin giriş katmanı
input_layer = Input(shape=(data_x_train_scaled.shape[1]), name='Input_Layer')

# Ortak katmanlar
common_path = Dense(units='128', activation='relu', name='First_Dense')(input_layer)
common_path = Dropout(0.3)(common_path)
common_path = Dense(units='128', activation='relu', name='Second_Dense')(common_path)
common_path = Dropout(0.3)(common_path)

# İlk çıkış yolu
first_output = Dense(units='1', name='First_Output__Last_Layer')(common_path)

# İkinci çıkış yolu
second_output_path = Dense(units='64', activation='relu', name='Second_Output__First_Dense')(common_path)
second_output_path = Dropout(0.3)(second_output_path)
second_output = Dense(units='1', name='Second_Output__Last_Layer')(second_output_path)

# Modeli oluştur
model = Model(inputs=input_layer, outputs=[first_output, second_output])
print(model.summary())

# Modeli derle
optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)
model.compile(optimizer=optimizer,
              loss={'First_Output__Last_Layer': 'mse', 'Second_Output__Last_Layer': 'mse'},
              metrics={'First_Output__Last_Layer': tf.keras.metrics.RootMeanSquaredError(),
                       'Second_Output__Last_Layer': tf.keras.metrics.RootMeanSquaredError()})

# EarlyStopping callback tanımla
earlyStopping_callback = EarlyStopping(monitor='val_loss',
                                       min_delta=0,
                                       patience=10,
                                       verbose=1)

# Modeli eğit
history = model.fit(x=data_x_train_scaled, y=data_y_train, verbose=0, epochs=500, batch_size=10,
                    validation_split=0.3, callbacks=earlyStopping_callback)

# Test verileri üzerinde tahmin yap
y_pred = np.array(model.predict(data_x_test_scaled))

# R2 skorlarını hesapla ve yazdır
from sklearn.metrics import r2_score
print("İlk çıkışın R2 değeri:", r2_score(data_y_test[:,0], y_pred[0,:].flatten()))
print("İkinci çıkışın R2 değeri:", r2_score(data_y_test[:,1], y_pred[1,:].flatten()))

# İlk çıkış için RMSE grafiğini çiz
plt.plot(history.history['First_Output__Last_Layer_root_mean_squared_error'])
plt.plot(history.history['val_First_Output__Last_Layer_root_mean_squared_error'])
plt.title('Model\'s İlk çıkış için RMSE kayıp değerleri')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
plt.figure()

# İkinci çıkış için RMSE grafiğini çiz
plt.plot(history.history['Second_Output__Last_Layer_root_mean_squared_error'])
plt.plot(history.history['val_Second_Output__Last_Layer_root_mean_squared_error'])
plt.title('Model\'s İkinci çıkış için RMSE kayıp değerleri')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()
