from data_loader import DataLoader
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Reshape, Dropout
from keras.utils import plot_model

# NOTE: if less than 16GB of RAM memory is available (64179 images of 65*320*3*4 bytes = 14.9 GB),
# python generators should be used to load the images from the disk in batches of
# 32 images at time, which could considerably decrease the training speed
# (around 50 times slower) compared to loading all the images directly in
# memory
use_generators=False

# Hyperparameters
validation_size = 0.2 # ratio of images used as validation set
correction=0.3 # correction applied to the left and right camera images
batch_size=32
dropout = 0.2 #probability of droping out
epochs = 5

#data laoder, used to load all the sample images and labels (steering angles)
data_loader = DataLoader()

if use_generators:
    X_train, X_valid, y_train, y_valid = data_loader.load_samples(validation_size, correction)
    training_generator = data_loader.generator(X_train, y_train, batch_size=batch_size)
    validation_generator = data_loader.generator(X_valid, y_valid, batch_size=batch_size)
    print("Trainig samples:", len(X_train)) 
    print("Validation samples:", len(X_valid))
else:
    X_train, y_train = data_loader.load(correction)

#Creates the model
model = Sequential()


model.add(Lambda(lambda x: x/127.5 -1, input_shape=(65,320,3))) # Add normalization layer
model.add(Conv2D(24,(5,5), activation="relu", strides=(2,2))) # Conv layer with kernel size 5x5 + RELU activation layer. Outputs 24 feature maps of 31x158
model.add(Dropout(dropout))
model.add(Conv2D(36,(5,5), activation="relu", strides=(2,2))) # Conv layer with kernel size 5x5 + RELU activation layer. Outputs 36 feature maps of 14x77
model.add(Dropout(dropout))
model.add(Conv2D(48,(5,5), activation="relu", strides=(2,2))) # Conv layer with kernel size 5x5 + RELU activation layer. Outputs 48 feature maps of 5x37
model.add(Dropout(dropout))
model.add(Conv2D(64,(3,3), activation="relu")) # Conv layer with kernel size 3x3 + RELU activation layer. Outputs 48 feature maps of 3x35
model.add(Dropout(dropout))
model.add(Conv2D(64,(3,3), activation="relu")) # Conv layer with kernel size 3x3 + RELU activation layer. Outputs 48 feature maps of 1x33
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dense(100)) # Fully connected layer
model.add(Dropout(dropout))
model.add(Dense(50)) # Fully connected layer
model.add(Dropout(dropout))
model.add(Dense(10)) # Fully connected layer
model.add(Dense(1)) # output: steering angle

model.compile(loss='mse', optimizer='adam')
model.summary() #print summary of the model
plot_model(model, to_file='model.png', show_shapes=True)

if use_generators:
    model.fit_generator(training_generator, steps_per_epoch= \
                        len(X_train), validation_data=validation_generator, \
                        validation_steps=len(X_valid), epochs=epochs, verbose=1)
else:
    model.fit(X_train, y_train, validation_split=validation_size, epochs=epochs, shuffle=True)

model.save('model.h5')
