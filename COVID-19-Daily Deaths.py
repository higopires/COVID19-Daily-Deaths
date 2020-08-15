#Inspired by Mr. Venelin Valkov's work in "Towards Data Science":
# URL: https://towardsdatascience.com/demand-prediction-with-lstms-using-tensorflow-2-and-keras-in-python-1d1076fc89a0

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import similaritymeasures as sm

plt.style.use(['science','ieee']) #requirement: pip install SciencePlots
# go to Matplotlib style directory:
    #import matplotlib
    #print(matplotlib.get_configdir())
# then go to stylelib folder
# on ieee.mplstyle file, comment the following line:
    #figure.figsize : 3.3, 2.5
#save file

df = pd.read_csv(
  "data_MA.csv", 
  parse_dates=['date'], 
  index_col="date"
)

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)] #create train and test sets


predictors = ['new_confirmed','V_coronavirus_trend','A_quarentena_trend'] #predictor attributes

#Robust scaling predictors and response attributes
pred_transformer = RobustScaler()
cntd_transformer = RobustScaler()
pred_transformer = pred_transformer.fit(train[predictors].to_numpy())
cntd_transformer = cntd_transformer.fit(train[['new_deaths']])

#copy the scaled values to the train set
train.loc[:, predictors] = pred_transformer.transform(train[predictors].to_numpy())
train['new_deaths'] = cntd_transformer.transform(train[['new_deaths']])

#copy the scaled values to the test set
test.loc[:, predictors] = pred_transformer.transform(test[predictors].to_numpy())
test['new_deaths'] = cntd_transformer.transform(test[['new_deaths']])

#function to split the sequence into multiple samples
def split_data(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

X_train, y_train = split_data(train[predictors], train.new_deaths) 
X_test, y_test = split_data(test[predictors], test.new_deaths)

##############################################################################

#Vanilla LSTM

vanilla_lstm = tf.keras.Sequential()
vanilla_lstm.add(tf.keras.layers.LSTM(128, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
vanilla_lstm.add(tf.keras.layers.Dropout(0.5))
vanilla_lstm.add(tf.keras.layers.Dense(1))
vanilla_lstm.compile(optimizer='adam', loss='mse')

vanilla_lstm_my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2, 
                                         patience=5, 
                                         min_lr=0.001),
    tf.keras.callbacks.ModelCheckpoint(filepath='models/Vanilla_LSTM_New_Deaths.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='min',
                                       save_weights_only=False)
    ]

vanilla_lstm_history = vanilla_lstm.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=1, 
    validation_split=0.05,
    callbacks = vanilla_lstm_my_callbacks,
    shuffle=False
)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(3.5, 3.5 / 1.618))

# Calculate the simple average of the data
vanilla_mean = np.mean(vanilla_lstm_history.history['val_loss'])

plt.plot(vanilla_lstm_history.history['loss'], label='Training Loss')
plt.plot(vanilla_lstm_history.history['val_loss'], label='Validation Loss')
plt.axhline(y=vanilla_mean, color='b', linestyle='--', label='Average Validation Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Mean Squared Error - Vanilla LSTM')
plt.legend();
fig.savefig('mse/MSE-VanillaLSTM.pdf')
plt.close(fig)

vanilla_y_pred = vanilla_lstm.predict(X_test)

y_train_inv = cntd_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cntd_transformer.inverse_transform(y_test.reshape(1, -1))
vanilla_y_pred_inv = cntd_transformer.inverse_transform(vanilla_y_pred)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), label="Actual Daily Deaths")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), label="Ground Truth")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), vanilla_y_pred_inv.flatten(), label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Days after First COVID-19 Case')
plt.title('COVID Daily Deaths Prediction - Vanilla LSTM')
plt.legend()
fig.savefig('actualdaily_actualdata_prediction/VanillaLSTM.pdf')
plt.close(fig)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

date_list = pd.date_range(start='06/18/2020', end='07/09/2020').to_series()
date_list = date_list.dt.strftime('%d')

plt.plot(np.array(date_list), y_test_inv.flatten(), marker='.', label="Ground Truth")
plt.plot(np.array(date_list), vanilla_y_pred_inv.flatten(), marker='.', label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Date (June-July 2020)')
plt.title('COVID Daily Deaths Prediction - Vanilla LSTM')
plt.legend()
fig.savefig('actualdata_prediction/VanillaLSTM.pdf')
plt.close(fig)

##############################################################################

#Stacked LSTM

stacked_lstm = tf.keras.Sequential()
stacked_lstm.add(tf.keras.layers.LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
stacked_lstm.add(tf.keras.layers.Dropout(0.5))
stacked_lstm.add(tf.keras.layers.LSTM(64, activation='relu'))
stacked_lstm.add(tf.keras.layers.Dropout(0.5))
stacked_lstm.add(tf.keras.layers.Dense(1))
stacked_lstm.compile(optimizer='adam', loss='mse')

stacked_lstm_my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2, 
                                         patience=5, 
                                         min_lr=0.001),
    tf.keras.callbacks.ModelCheckpoint(filepath='models/Stacked_LSTM_New_Deaths.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='min',
                                       save_weights_only=False)
    ]

stacked_lstm_history = stacked_lstm.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=1, 
    validation_split=0.05,
    callbacks = stacked_lstm_my_callbacks,
    shuffle=False
)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(3.5, 3.5 / 1.618))

stacked_mean = np.mean(stacked_lstm_history.history['val_loss'])

plt.plot(stacked_lstm_history.history['loss'], label='Training Loss')
plt.plot(stacked_lstm_history.history['val_loss'], label='Validation Loss')
plt.axhline(y=stacked_mean, color='b', linestyle='--', label='Average Validation Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Mean Squared Error - Stacked LSTM')
plt.legend();
fig.savefig('mse/MSE-StackedLSTM.pdf')
plt.close(fig)

stacked_y_pred = stacked_lstm.predict(X_test)

y_train_inv = cntd_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cntd_transformer.inverse_transform(y_test.reshape(1, -1))
stacked_y_pred_inv = cntd_transformer.inverse_transform(stacked_y_pred)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), label="Actual Daily Deaths")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), label="Ground Truth")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), stacked_y_pred_inv.flatten(), label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Days after First COVID-19 Case')
plt.title('COVID Daily Deaths Prediction - Stacked LSTM')
plt.legend()
fig.savefig('actualdaily_actualdata_prediction/StackedLSTM.pdf')
plt.close(fig)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.array(date_list), y_test_inv.flatten(), marker='.', label="Ground Truth")
plt.plot(np.array(date_list), stacked_y_pred_inv.flatten(), marker='.', label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Date (June-July 2020)')
plt.title('COVID Daily Deaths Prediction - Stacked LSTM')
plt.legend()
fig.savefig('actualdata_prediction/StackedLSTM.pdf')
plt.close(fig)

##############################################################################

#Bidirectional LSTM

bidirect_lstm = keras.Sequential()
bidirect_lstm.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128, 
      input_shape=(X_train.shape[1], X_train.shape[2]))))
bidirect_lstm.add(keras.layers.Dropout(rate=0.5))
bidirect_lstm.add(keras.layers.Dense(units=1))
bidirect_lstm.compile(loss='mean_squared_error', optimizer='adam')

bidirect_lstm_my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2, 
                                         patience=5, 
                                         min_lr=0.001),
    tf.keras.callbacks.ModelCheckpoint(filepath='models/Bidirectional_LSTM_New_Deaths.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='min',
                                       save_weights_only=False)
    ]

#treinamento: por ser uma série temporal, os dados não devem ser embaralhados
bidirect_lstm_history = bidirect_lstm.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=1, 
    validation_split=0.05,
    callbacks = bidirect_lstm_my_callbacks,
    shuffle=False
)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(3.5, 3.5 / 1.618))

bidirect_mean = np.mean(bidirect_lstm_history.history['val_loss'])

plt.plot(bidirect_lstm_history.history['loss'], label='Training Loss')
plt.plot(bidirect_lstm_history.history['val_loss'], label='Validation Loss')
plt.axhline(y=bidirect_mean, color='b', linestyle='--', label='Average Validation Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Mean Squared Error - Bidirectional LSTM')
plt.legend();
fig.savefig('mse/MSE-BidirectionalLSTM.pdf')
plt.close(fig)

bidirect_y_pred = bidirect_lstm.predict(X_test)

y_train_inv = cntd_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cntd_transformer.inverse_transform(y_test.reshape(1, -1))
bidirect_y_pred_inv = cntd_transformer.inverse_transform(bidirect_y_pred)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), label="Actual Daily Deaths")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), label="Ground Truth")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), bidirect_y_pred_inv.flatten(), label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Days after First COVID-19 Case')
plt.title('COVID Daily Deaths Prediction - BidirectionalLSTM')
plt.legend()
fig.savefig('actualdaily_actualdata_prediction/BidirectionalLSTM.pdf')
plt.close(fig)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.array(date_list), y_test_inv.flatten(), marker='.', label="Ground Truth")
plt.plot(np.array(date_list), bidirect_y_pred_inv.flatten(), marker='.', label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Date (June-July 2020)')
plt.title('COVID Daily Deaths Prediction - Bidirectional LSTM')
plt.legend()
fig.savefig('actualdata_prediction/BidirectionalLSTM.pdf')
plt.close(fig)

##############################################################################

#Gated Recurrent Unit (GRU)

model_gru = tf.keras.Sequential()
model_gru.add(tf.keras.layers.GRU(128,input_shape=(X_train.shape[1], X_train.shape[2])))
model_gru.add(tf.keras.layers.Dropout(0.5))
model_gru.add(tf.keras.layers.Dense(units=1))
model_gru.compile(loss='mse', optimizer='adam')

model_gru_my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2, 
                                         patience=5, 
                                         min_lr=0.001),
    tf.keras.callbacks.ModelCheckpoint(filepath='models/GRU_New_Deaths.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='min',
                                       save_weights_only=False)
    ]

model_gru_history = model_gru.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=1, 
    validation_split=0.05,
    callbacks = model_gru_my_callbacks,
    shuffle=False
)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(3.5, 3.5 / 1.618))

model_gru_mean = np.mean(model_gru_history.history['val_loss'])

plt.plot(model_gru_history.history['loss'], label='Training Loss')
plt.plot(model_gru_history.history['val_loss'], label='Validation Loss')
plt.axhline(y=model_gru_mean, color='b', linestyle='--', label='Average Validation Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Mean Squared Error - GRU')
plt.legend();
fig.savefig('mse/MSE-GRU.pdf')
plt.close(fig)

model_gru_y_pred = model_gru.predict(X_test)

y_train_inv = cntd_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cntd_transformer.inverse_transform(y_test.reshape(1, -1))
model_gru_y_pred_inv = cntd_transformer.inverse_transform(model_gru_y_pred)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), label="Actual Daily Deaths")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), label="Ground Truth")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), model_gru_y_pred_inv.flatten(), label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Days after First COVID-19 Case')
plt.title('COVID Daily Deaths Prediction - GRU')
plt.legend()
fig.savefig('actualdaily_actualdata_prediction/GRU.pdf')
plt.close(fig)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.array(date_list), y_test_inv.flatten(), marker='.', label="Ground Truth")
plt.plot(np.array(date_list), model_gru_y_pred_inv.flatten(), marker='.', label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Date (June-July 2020)')
plt.title('COVID Daily Deaths Prediction - GRU')
plt.legend()
fig.savefig('actualdata_prediction/GRU.pdf')
plt.close(fig)

##############################################################################

#CNN LSTM

n_features = 3 #number of predictive attributes
n_seq = 1 #number of values processed in the prediction
n_steps = 1 #number of result values in the prediction

cnn_lstm = tf.keras.Sequential()
cnn_lstm.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Conv1D(filters=64,
                                                                    kernel_size=1,
                                                                    activation='relu'),
                                             input_shape=(n_seq, n_steps, n_features)))
cnn_lstm.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling1D(pool_size=1)))
cnn_lstm.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))
cnn_lstm.add(tf.keras.layers.LSTM(64, activation='relu'))
cnn_lstm.add(tf.keras.layers.Dense(1))
cnn_lstm.compile(optimizer='adam', loss='mse')

X_cnnlstm = X_train.reshape((X_train.shape[0], n_seq, n_steps, n_features))

cnn_lstm_my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2, 
                                         patience=5, 
                                         min_lr=0.001),
    tf.keras.callbacks.ModelCheckpoint(filepath='models/CNN_LSTM_New_Deaths.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='min',
                                       save_weights_only=False)
    ]

cnn_lstm_history = cnn_lstm.fit(
    X_cnnlstm, y_train, 
    epochs=100, 
    batch_size=1, 
    validation_split=0.05,
    callbacks = cnn_lstm_my_callbacks,
    shuffle=False
)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(3.5, 3.5 / 1.618))

cnn_lstm_mean = np.mean(cnn_lstm_history.history['val_loss'])

plt.plot(cnn_lstm_history.history['loss'], label='Training Loss')
plt.plot(cnn_lstm_history.history['val_loss'], label='Validation Loss')
plt.axhline(y=cnn_lstm_mean, color='b', linestyle='--', label='Average Validation Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Mean Squared Error - CNN LSTM')
plt.legend();
fig.savefig('mse/MSE-CNNLSTM.pdf')
plt.close(fig)

X_test_cnn_lstm = X_test.reshape((X_test.shape[0], n_seq, n_steps, n_features))
cnn_lstm_y_pred = cnn_lstm.predict(X_test_cnn_lstm)

y_train_inv = cntd_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cntd_transformer.inverse_transform(y_test.reshape(1, -1))
cnn_lstm_y_pred_inv = cntd_transformer.inverse_transform(cnn_lstm_y_pred)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), label="Actual Daily Deaths")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), label="Ground Truth")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), cnn_lstm_y_pred_inv.flatten(), label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Days after First COVID-19 Case')
plt.title('COVID Daily Deaths Prediction - CNN LSTM')
plt.legend()
fig.savefig('actualdaily_actualdata_prediction/CNNLSTM.pdf')
plt.close(fig)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.array(date_list), y_test_inv.flatten(), marker='.', label="Ground Truth")
plt.plot(np.array(date_list), cnn_lstm_y_pred_inv.flatten(), marker='.', label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Date (June-July 2020)')
plt.title('COVID Daily Deaths Prediction - CNN LSTM')
plt.legend()
fig.savefig('actualdata_prediction/CNNLSTM.pdf')
plt.close(fig)

##############################################################################

#ConvLSTM

#pre-processing: reshape the data to fit in the model
X_convlstm = X_train.reshape((X_train.shape[0], n_seq, 1, n_steps, n_features))

convlstm = tf.keras.Sequential()
convlstm.add(tf.keras.layers.ConvLSTM2D(filters=128, kernel_size=(1,1), activation='relu',
                     input_shape=(n_seq, 1, n_steps, n_features)))
convlstm.add(tf.keras.layers.Flatten())
convlstm.add(tf.keras.layers.Dense(1))
convlstm.compile(optimizer='adam', loss='mse')

convlstm_my_callbacks = [
    #tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                         factor=0.2, 
                                         patience=5, 
                                         min_lr=0.001),
    keras.callbacks.ModelCheckpoint(filepath='models/ConvLSTM_New_Deaths.h5',
                                       monitor='val_loss',
                                       save_best_only=True,
                                       mode='min',
                                       save_weights_only=False)
    ]

convlstm_history = convlstm.fit(
    X_convlstm, y_train, 
    epochs=100, 
    batch_size=1, 
    validation_split=0.05,
    callbacks = convlstm_my_callbacks,
    shuffle=False
)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(3.5, 3.5 / 1.618))

convlstm_mean = np.mean(convlstm_history.history['val_loss'])

plt.plot(convlstm_history.history['loss'], label='Training Loss')
plt.plot(convlstm_history.history['val_loss'], label='Validation Loss')
plt.axhline(y=convlstm_mean, color='b', linestyle='--', label='Average Validation Loss')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Mean Squared Error - ConvLSTM')
plt.legend();
fig.savefig('mse/MSE-ConvLSTM.pdf')
plt.close(fig)

X_test_convlstm = X_test.reshape((X_test.shape[0], n_seq, 1, n_steps, n_features))
convlstm_y_pred = convlstm.predict(X_test_convlstm)

y_train_inv = cntd_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = cntd_transformer.inverse_transform(y_test.reshape(1, -1))
convlstm_y_pred_inv = cntd_transformer.inverse_transform(convlstm_y_pred)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), label="Actual Daily Deaths")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), label="Ground Truth")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), convlstm_y_pred_inv.flatten(), label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Days after First COVID-19 Case')
plt.title('COVID Daily Deaths Prediction - ConvLSTM')
plt.legend()
fig.savefig('actualdaily_actualdata_prediction/ConvLSTM.pdf')
plt.close(fig)

fig, ax = plt.subplots()
fig = plt.figure(figsize=(7.16, 7.16 / 1.618))

plt.plot(np.array(date_list), y_test_inv.flatten(), marker='.', label="Ground Truth")
plt.plot(np.array(date_list), convlstm_y_pred_inv.flatten(), marker='.', label="Prediction")
plt.ylabel('Daily COVID-19 Deaths')
plt.xlabel('Date (June-July 2020)')
plt.title('COVID Daily Deaths Prediction - ConvLSTM')
plt.legend()
fig.savefig('actualdata_prediction/ConvLSTM.pdf')
plt.close(fig)

##############################################################################

#Calculating the Dynamic Time Warping values

# Ground truth
truth = y_test_inv
truth_data = np.zeros((22, 2))
truth_data[:, 0] = np.arange(22)
truth_data[:, 1] = truth

# Vanilla
vanilla = vanilla_y_pred_inv.flatten()
vanilla_data = np.zeros((22, 2))
vanilla_data[:, 0] = np.arange(22)
vanilla_data[:, 1] = vanilla

# Stacked
stacked = stacked_y_pred_inv.flatten()
stacked_data = np.zeros((22, 2))
stacked_data[:, 0] = np.arange(22)
stacked_data[:, 1] = stacked

# Bidirectional
bidirect = bidirect_y_pred_inv.flatten()
bidirect_data = np.zeros((22, 2))
bidirect_data[:, 0] = np.arange(22)
bidirect_data[:, 1] = bidirect

# GRU
gru = model_gru_y_pred_inv.flatten()
gru_data = np.zeros((22, 2))
gru_data[:, 0] = np.arange(22)
gru_data[:, 1] = gru

# CNN LSTM
cnn = cnn_lstm_y_pred_inv.flatten()
cnn_data = np.zeros((22, 2))
cnn_data[:, 0] = np.arange(22)
cnn_data[:, 1] = cnn

# ConvLSTM
convolstm = convlstm_y_pred_inv.flatten()
convolstm_data = np.zeros((22, 2))
convolstm_data[:, 0] = np.arange(22)
convolstm_data[:, 1] = convolstm

#Similarity Measures
dtw_vanilla, d_vanilla = sm.dtw(truth_data, vanilla_data)
dtw_stacked, d_stacked = sm.dtw(truth_data, stacked_data)
dtw_bidirect, d_bidirect = sm.dtw(truth_data, bidirect_data)
dtw_gru, d_gru = sm.dtw(truth_data, gru_data)
dtw_cnn, d_cnn = sm.dtw(truth_data, cnn_data)
dtw_convolstm, d_convolstm = sm.dtw(truth_data, convolstm_data)

#Write DTW results in file
file = open("DTW.txt","w")
file.write("DTW Vanilla: {}\n".format(round(dtw_vanilla,3)))
file.write("DTW Stacked: {}\n".format(round(dtw_stacked,3)))
file.write("DTW Bidirectional: {}\n".format(round(dtw_bidirect,3)))
file.write("DTW GRU: {}\n".format(round(dtw_gru,3)))
file.write("DTW CNN LSTM: {}\n".format(round(dtw_cnn,3)))
file.write("DTW ConvLSTM: {}\n".format(round(dtw_convolstm,3)))
file.write("------------------------\n")
file.write("Mean Loss Vanilla: {}\n".format(round(vanilla_mean,3)))
file.write("Mean Loss Stacked: {}\n".format(round(stacked_mean,3)))
file.write("Mean Loss Bidirectional: {}\n".format(round(bidirect_mean,3)))
file.write("Mean Loss GRU: {}\n".format(round(model_gru_mean,3)))
file.write("Mean Loss CNN LSTM: {}\n".format(round(cnn_lstm_mean,3)))
file.write("Mean Loss ConvLSTM: {}\n".format(round(convlstm_mean,3)))
file.close()