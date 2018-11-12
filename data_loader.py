import csv
import numpy as np
from scipy import ndimage
import sklearn
from sklearn.model_selection import train_test_split

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

    def load_samples(self,test_size= 0.2, correction=0.2):
        lines = []
        with open('../data/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        X_samples = []
        y_measurements = []
        for line in lines:
            filename_img_center = line[0].split('/')[-1]
            filename_img_left = line[1].split('/')[-1]
            filename_img_right = line[2].split('/')[-1]
            steering_center = float(line[3])
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            X_samples.append(filename_img_center)
            X_samples.append(filename_img_left)
            X_samples.append(filename_img_right)
            y_measurements.append(steering_center)
            y_measurements.append(steering_left)
            y_measurements.append(steering_right)

        X_samples = np.array(X_samples)
        y_measurements = np.array(y_measurements)
        X_train, X_valid, y_train, y_valid = train_test_split(X_samples, y_measurements, test_size=test_size, random_state=42)
        return X_train, X_valid, y_train, y_valid
    
    def generator(self,X_samples, y_measurements, batch_size=32, path='../data/IMG/'):
        num_samples = len(X_samples)
        while True: # Loop forever so the generator never terminates
            for offset in range(0, num_samples, batch_size):
                batch_samples = X_samples[offset:offset+batch_size]
                batch_measurements = y_measurements[offset:offset+batch_size]
                images = []
                angles = []
                for i in range(0, len(batch_samples)):
                    image = ndimage.imread(path + batch_samples[i])
                    angle = batch_measurements[i]
                    images.append(image)
                    angles.append(angle)
                    
                    #augment the dataset with the corresponding flipped images and angles
                    images.append(np.fliplr(image))
                    angles.append(-angle)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)
