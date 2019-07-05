import random
from collections import deque
import numpy as np
import pandas as pd
from sklearn import preprocessing
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, BatchNormalization, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

# start=dt.datetime(2004,8,19)
# end=dt.datetime.now()
#
# df=web.DataReader('GOOGL','yahoo',start,end)
#
# df.to_csv('GOOGL_data.csv')

TS = 10  # timestep


# if future value is > current value buy stock
# else sell stock
def classify(now, future):
    if future > now:
        return 1
    else:
        return 0


def processing(df):
    # dont need future column, we have target column
    df = df.drop('future', 1)

    for col in df.columns:  # high low open close volume adj columns and target
        if col != 'target':  # target column already in scale
            df[col] = df[col].pct_change()  # normalize the data

            df.dropna(inplace=True)  # first column is going top be nan
            df[col] = preprocessing.scale(df[col].values)  # sclale all column

    df.dropna(inplace=True)  # just in case

    previus_day = deque(maxlen=TS)  # deque: if reach maxlen first value pop
    feed_data = []#values and target
    for i in df.values:#i:first row
        previus_day.append([n for n in i[:-1]])#Take all data in row except target
        if len(previus_day) == TS:
            feed_data.append([np.array(previus_day), i[-1]])#add data and target
            print(feed_data)
    random.shuffle(feed_data)#just in case suffle

    buys = []
    sell = []
    #to balance the data
    for seq, target in feed_data:
        if target == 1:
            buys.append([seq, target])
        elif target == 0:
            sell.append([seq, target])
    #which is lower
    lower = min(len(buys), len(sell))
    #balancing
    buys = buys[:lower]
    sell = sell[:lower]

    feed_data = buys + sell

    random.shuffle(feed_data)

    X = []
    y = []

    for seq, target in feed_data:
        X.append(seq)
        y.append(target)
    return np.array(X), y

#read in csv file
df = pd.read_csv('GOOGL_data.csv')

#Shift up
df['future'] = df['Close'].shift(-TS)
df.dropna(inplace=True)

main_df = df[['Date', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'future', ]]

main_df.set_index('Date', inplace=True)

#mapping
main_df['target'] = list(map(classify, main_df['Close'], main_df['future']))

times = main_df.index.values

last5_p = times[-int(len(times) * 0.05)]
#take %95 for training , %5 testing
train = main_df[(main_df.index < last5_p)]
validation = main_df[(main_df.index >= last5_p)]

# print(train)
# print(validation)

x_train, y_train = processing(train)
x_valid, y_valid = processing(validation)

print(f"Train data : {len(x_train)}  validation data : {len(x_valid)}")
print(f"Dont buys : {y_train.count(0)} buys : {y_train.count(1)}")
print(f"validation dont buys {y_valid.count(0)},validation buys  {y_valid.count(1)}")


model = Sequential()
#if run on cpu change CuDNNLSTM to LSTM and add activation
model.add(CuDNNLSTM(128, return_sequences=True, input_shape=x_train.shape[1:]))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(CuDNNLSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_valid, y_valid), verbose=2)

# model.save("Google-Stock-Prediction.h5")

# Epoch 100/100
# 3000/3000 - 1s - loss: 0.6119 - acc: 0.6473 - val_loss: 0.7631 - val_acc: 0.5987
