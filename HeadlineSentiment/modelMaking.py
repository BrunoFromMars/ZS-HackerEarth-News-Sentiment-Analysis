

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Activation, Reshape, GlobalAveragePooling1D, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json





"""# Model Making Conv1D with Dense"""



model_m = Sequential()
model_m.add(Reshape((863, 9), input_shape=(7767,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(123, 55)))
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


X = df_final.iloc[:,0:-2]

X = X.as_matrix()

Y = df_final.iloc[:,-1:].as_matrix()

#Y = min_max_scaler.fit_transform(Y)

#training_samples = 40000
#validation_samples = 55932-40000

#x_train = X[:training_samples]
#y_train = Y[:training_samples]
#x_val = X[training_samples:training_samples + validation_samples]
#y_val = Y[training_samples:training_samples + validation_samples]

#x_train = []
#x_val = []

model_m.compile(optimizer='Adam',
              loss='mse',
              metrics=['mae','accuracy'])

#!pip install h5py

model_m.fit(X,Y,batch_size=500,epochs=30,verbose=1,callbacks=None,validation_split=0.2)

#saving model
model_c1d_8April_1820_json = model_m.to_json()

with open("model_c1d_8April_1820.json", "w") as json_file:
    json_file.write(model_c1d_8April_1820_json)

model_m.save_weights("model_c1d_8April_1820.h5")
print("Saved model to disk")
""" Deep Neural Network Simple"""
"""
model_d = Sequential()
model_d.add(Dense(1000, input_dim=6765, activation='relu'))
model_d.add(Dropout(0.5))
model_d.add(Dense(700, input_dim=1000, activation='relu'))
model_d.add(Dropout(0.5))
model_d.add(Dense(500, input_dim=700, activation='relu'))
model_d.add(Dropout(0.2))
model_d.add(Dense(100, input_dim=500, activation='relu'))
model_d.add(Dropout(0.2))
model_d.add(Dense(50, input_dim=100, activation='relu'))
model_d.add(Dropout(0.2))
model_d.add(Dense(10, input_dim=50, activation='relu'))
model_d.add(Dropout(0.2))
model_d.add(Dense(1, input_dim=10, activation='tanh'))
print(model_d.summary())

model_d.compile(optimizer='Adam',
              loss='mse',
              metrics=['mae','accuracy'])

model_d.fit(x_train,y_train,batch_size=500,epochs=30,verbose=1,callbacks=None,validation_data=(x_val,y_val))

model_d_json = model_d.to_json()
with open("model_d.json", "w") as json_file:
    json_file.write(model_d_json)

model_d.save_weights("model_d.h5")
print("Saved model to disk")

"""
"""Deep Neural Network Simple"""

"""

model_d2 = Sequential()
model_d2.add(Dense(3000, input_dim=6765, activation='relu'))
model_d2.add(Dropout(0.2))
model_d2.add(Dense(1000, input_dim=3000, activation='relu'))
model_d2.add(Dropout(0.2))
model_d2.add(Dense(500, input_dim=1000, activation='relu'))
model_d2.add(Dropout(0.2))
model_d2.add(Dense(100, input_dim=500, activation='relu'))
model_d2.add(Dropout(0.2))
model_d2.add(Dense(50, input_dim=100, activation='relu'))
model_d2.add(Dropout(0.2))
model_d2.add(Dense(10, input_dim=50, activation='relu'))
model_d2.add(Dropout(0.2))
model_d2.add(Dense(1, input_dim=10, activation='tanh'))
print(model_d2.summary())

model_d2.compile(optimizer='Adam',
              loss='mse',
              metrics=['mae','accuracy'])

model_d2.fit(x_train,y_train,batch_size=500,epochs=60,verbose=1,callbacks=None,validation_data=(x_val,y_val))

model_d2_json = model_d2.to_json()
with open("model_d2.json", "w") as json_file:
    json_file.write(model_d2_json)

model_d2.save_weights("model_d2.h5")
print("Saved model to disk")
"""

""" Conv2D with Dense """

"""
model_c2 = Sequential()

model_c2.add(Reshape((123, 55,1), input_shape=(6765,)))
model_c2.add(Conv2D(100, kernel_size=(3, 3), activation='relu', input_shape=(123, 55, 1)))
model_c2.add(Conv2D(200,kernel_size=(3, 3), activation='relu'))
model_c2.add(MaxPooling2D(pool_size=(2, 2)))
model_c2.add(Dropout(0.25))
model_c2.add(Flatten())
model_c2.add(Dense(20, activation='relu'))
model_c2.add(Dropout(0.25))
model_c2.add(Dense(15, input_dim=20, activation='relu'))
model_c2.add(Dropout(0.25))
model_c2.add(Dense(10, input_dim=15 , activation='relu'))
model_c2.add(Dropout(0.25))
model_c2.add(Dense(1, input_dim=10 ,activation='tanh'))
print(model_c2.summary())

model_c2.compile(optimizer='Adam',
              loss='mse',
              metrics=['mae','accuracy'])

filepath="weights-improvement-{epoch:02d}-{val_mae:.2f}.hdf5"


checkpoint = ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]

model_c2.fit(x_train,y_train,batch_size=200,epochs=30,callbacks=callbacks_list, verbose=1,validation_data=(x_val,y_val))

model_c2_json = model_c2.to_json()
with open("model_c2.json", "w") as json_file:
    json_file.write(model_c2_json)

model_c2.save_weights("model_c2.h5")
print("Saved model to disk")



