import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras


df = pd.read_csv("dane_hurtowni.csv")

df_przetworzone = pd.get_dummies(df, columns=["Kategoria"], dtype=int)


X = df_przetworzone.drop(columns=["Produkt_Nazwa", "Realna_Sprzedaz"])
y = df_przetworzone["Realna_Sprzedaz"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),   
    keras.layers.Dense(64, activation='relu'),              
    keras.layers.Dense(32, activation='relu'),              
    keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mse', metrics=['mae'])

 
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test_scaled,y_test),
    verbose=1    
)

loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)     

X_all_scaled = scaler.transform(X) 
prognozy_raw = model.predict(X_all_scaled, verbose=0)

df["Prognoza_AI"] = np.round(prognozy_raw.flatten())
df["Prognoza_AI"] = df["Prognoza_AI"].clip(lower=0)

df.to_csv("wyniki_prognozy.csv", index=False)
