from data_loader import DataLoader
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Reshape, Dropout
import pickle

#parameters
validation_size = 0.2
correction=0.2
batch_size=32
dropout = 0.2 #probability of droping out
epochs = 20

data_loader = DataLoader()
X_train, y_train = data_loader.load()

#X_train, X_valid, y_train, y_valid = data_loader.load_samples(validation_size, correction)
#training_generator = data_loader.generator(X_train, y_train, batch_size=batch_size)
#validation_generator = data_loader.generator(X_valid, y_valid, batch_size=batch_size)

print("Trainig samples:", len(X_train)) 
#print("Validation samples:", len(X_valid))
model = Sequential()
model.add(Lambda(lambda x: x/127.5 -1, input_shape=(65,320,3)))
model.add(Conv2D(24,(5,5), activation="relu", strides=(2,2)))
model.add(Dropout(dropout))
model.add(Conv2D(36,(5,5), activation="relu", strides=(2,2)))
model.add(Dropout(dropout))
model.add(Conv2D(48,(5,5), activation="relu", strides=(2,2)))
model.add(Dropout(dropout))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Dropout(dropout))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(dropout))
model.add(Dense(50))
model.add(Dropout(dropout))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#history_object = model.fit_generator(training_generator, steps_per_epoch= \
#                    len(X_train), validation_data=validation_generator, \
#                    validation_steps=len(X_valid), epochs=epochs, verbose=1)
history_object = model.fit(X_train, y_train, validation_split=0.2, epochs=5, shuffle=True)

model.save('model.h5')

pickle.dump(history_object, open('loss.p', 'wb'))
