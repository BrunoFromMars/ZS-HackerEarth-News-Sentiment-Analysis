
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Activation, Reshape, GlobalAveragePooling1D, Dropout, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json


import pandas as pd
import numpy as np

json_file = open('model_c1d_8April_2259_t.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_c1d_8April_2259_t.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer='Adam',
              loss='mse',
              metrics=['mae','accuracy'])

predict_t = loaded_model.predict(df_final_test,verbose=1)

predict_t

pd.DataFrame(predict_t).to_csv('predicted_t.csv')

p_t = pd.DataFrame(predict_t)

p_t.describe()


