

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Activation, Reshape, GlobalAveragePooling1D, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json


# Model Making

model_m = Sequential()
model_m.add(Reshape((101, 67), input_shape=(6767,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(101, 67)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(200, 10, activation='relu'))
model_m.add(Conv1D(200, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Dropout(0.2))
#model_m.add(Flatten())
model_m.add(Dense(75, activation='relu'))
model_m.add(Flatten())
model_m.add(Dropout(0.25))
model_m.add(Dense(25, input_dim=75, activation='relu'))
model_m.add(Dropout(0.25))
model_m.add(Dense(10, input_dim=25, activation='relu'))
model_m.add(Dropout(0.25))
model_m.add(Dense(1, input_dim=10, activation='tanh'))
print(model_m.summary())

model_m.compile(optimizer='Adam',
              loss='mse',
              metrics=['mae','accuracy'])

model_m.fit(df_final.iloc[:,0:-1],df_final.iloc[:,-1:],batch_size=500,epochs=30,verbose=1,callbacks=None,validation_split=0.2)

!pip install h5py

model_c1d_8April_2259_t_json = model_m.to_json()

with open("model_c1d_8April_2259_t.json", "w") as json_file:
    json_file.write(model_c1d_8April_2259_t_json)

model_m.save_weights("model_c1d_8April_2259_t.h5")
print("Saved model to disk")

