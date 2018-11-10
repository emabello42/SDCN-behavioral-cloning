import csv
import numpy as np
from scipy import ndimage

class DataLoader(object):
    
    def load(self):
        lines = []
        with open('../data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        car_images = []
        steering_angles = []
        for line in lines:
            path = '../data/IMG/'
            correction = 0.2
            filename_img_center = path + line[0].split('/')[-1]
            filename_img_left = path + line[1].split('/')[-1]
            filename_img_right = path + line[2].split('/')[-1]
            img_center = ndimage.imread(filename_img_center)
            img_left = ndimage.imread(filename_img_left)
            img_right = ndimage.imread(filename_img_right)
            steering_center = float(line[3])
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            car_images.append(img_center)
            car_images.append(img_left)
            car_images.append(img_right)
            steering_angles.append(steering_center)
            steering_angles.append(steering_left)
            steering_angles.append(steering_right)

        augmented_images, augmented_measurements = [], []
        for image, steering_angle in zip(car_images, steering_angles):
            augmented_images.append(image)
            augmented_measurements.append(steering_angle)
            augmented_images.append(np.fliplr(image))
            augmented_measurements.append(-steering_angle)
        
        X_train = np.array(augmented_images)
        y_train = np.array(augmented_measurements)
        return X_train, y_train
