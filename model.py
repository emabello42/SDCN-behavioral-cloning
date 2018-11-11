from data_loader import DataLoader
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
import pickle

data_loader = DataLoader()
X_train, X_valid, y_train, y_valid = data_loader.load_samples()
training_generator = data_loader.generator(X_train, y_train, batch_size=32)
validation_generator = data_loader.generator(X_valid, y_valid, batch_size=32)
#X_train, y_train = data_loader.load()

print("Trainig samples:", len(X_train)*2) #original images + flipped images
print("Validation samples:", len(X_valid)*2)#original images + flipped images
model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Conv2D(24,(5,5), activation="relu", strides=(2,2)))
model.add(Conv2D(36,(5,5), activation="relu", strides=(2,2)))
model.add(Conv2D(48,(5,5), activation="relu", strides=(2,2)))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, epochs=5, shuffle=True)

history_object = model.fit_generator(training_generator, steps_per_epoch= \
                    len(X_train), validation_data=validation_generator, \
                    validation_steps=len(X_valid), epochs=5, verbose=1)


model.save('model.h5')

with open('loss.pkl', 'wb') as f:
    pickle.dump(history_object, f, pickle.HIGHEST_PROTOCOL)
