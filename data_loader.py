import csv
import numpy as np
from scipy import ndimage
import sklearn
from sklearn.model_selection import train_test_split

class DataLoader(object):

    def __init__(self, path = '../data/IMG/', csv_file='../data/driving_log.csv'):
        self.path = path
        self.csv_file = csv_file

    def load_samples_v2(self, correction = 0.2):
        ''' loads the image data directly in memory'''
        lines = []
        with open(self.csv_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        car_images = []
        steering_angles = []
        for line in lines:
            filename_img_center = self.path + line[0].split('/')[-1]
            filename_img_left = self.path + line[1].split('/')[-1]
            filename_img_right = self.path + line[2].split('/')[-1]
            img_center = ndimage.imread(filename_img_center)
            img_left = ndimage.imread(filename_img_left)
            img_right = ndimage.imread(filename_img_right)
            steering_center = float(line[3])
            # corrects the steering angles for the images taken from the left
            # and right cameras
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            # the images are cropped to focus the image on the road
            car_images.append(img_center[70:135,:])
            car_images.append(img_left[70:135,:])
            car_images.append(img_right[70:135,:])
            steering_angles.append(steering_center)
            steering_angles.append(steering_left)
            steering_angles.append(steering_right)

        X_train = np.array(car_images)
        y_train = np.array(steering_angles)
        return X_train, y_train

    def load_samples_v1(self,test_size= 0.2, correction=0.2):
        ''' loads the filenames of the images in the dataset and the steering angles.
            Generates the training and validation sets too.
        '''
        lines = []
        with open(self.csv_file) as csvfile:
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
            
            center = self.path + filename_img_center
            left = self.path + filename_img_left
            right = self.path + filename_img_right
            X_samples.append(center)
            X_samples.append(left)
            X_samples.append(right)
            y_measurements.append(steering_center)
            y_measurements.append(steering_left)
            y_measurements.append(steering_right)

        X_samples = np.array(X_samples)
        y_measurements = np.array(y_measurements)
        X_train, X_valid, y_train, y_valid = train_test_split(X_samples, y_measurements, test_size=test_size, random_state=42)
        return X_train, X_valid, y_train, y_valid
    
    def generator(self,X_samples, y_measurements, batch_size=32):
        ''' Generator used to load the images in batches from disk'''
        num_samples = len(X_samples)
        while True: # Loop forever so the generator never terminates
            for offset in range(0, num_samples, batch_size):
                batch_samples = X_samples[offset:offset+batch_size]
                batch_measurements = y_measurements[offset:offset+batch_size]
                images = []
                angles = []
                for i in range(0, len(batch_samples)):
                    image = ndimage.imread(batch_samples[i])
                    image = image[70:135,:]
                    angle = batch_measurements[i]
                    images.append(image)
                    angles.append(angle)
                    
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)
