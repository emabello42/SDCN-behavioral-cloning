import matplotlib.pyplot as plt
#history = {'loss': [0.0562 ,0.0482 ,0.0453 ,0.0436,0.0422 ,0.0414 ,0.0407 ,0.0401 ,0.0398,0.0394,0.0391,0.0388,0.0385 ,0.0383 ,0.0382,0.0380,0.0378,0.0377 ,0.0376,0.0374],
#        'val_loss': [0.0517,0.0494,0.0477,0.0475,0.0469,0.0467,0.0464,0.0465,0.0470,0.0459,0.0464,0.0509,0.0472,0.0485,0.0465,0.0510,0.0471,0.0466,0.0476,0.0500]}
history = {'loss': [0.0905, 0.0802, 0.0777, 0.0762, 0.0737], 'val_loss': [0.1248, 0.1411, 0.1297, 0.1224, 0.1458]}
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
