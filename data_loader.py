import csv
import numpy as np
from scipy import ndimage
import sklearn
from sklearn.model_selection import train_test_split

class DataLoader(object):

    def __init__(self, path = '../data/IMG/', csv_file='../data/driving_log.csv'):
        self.path = path
        self.csv_file = csv_file

    def load(self, correction = 0.2):
        ''' loads the image data directly into memory and augments it'''
        lines = []
        with open(self.csv_file) as csvfile:
            reader = csv.reader(self.csvfile)
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
            car_images.append(img_center[70:135,:])
            car_images.append(img_left[70:135,:])
            car_images.append(img_right[70:135,:])
            steering_angles.append(steering_center)
            steering_angles.append(steering_left)
            steering_angles.append(steering_right)

        # the original dataset is augmented flipping the images and computing
        # their corresponding steering angles as the negative of the original
        # one.
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
        ''' loads the filenames of the images in the dataset and the steering angles
            Generates the training and validation sets
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
            
            ## NOTE: instead of loading the filenames, and if enough RAM is available
            ## (at least 16GB), it is possible to load all the images directly into memory,
            ## so that the generator function takes only these images from the RAM memory,
            ## which is generally faster than reading from disk every time the generator
            ## function is called (specially if you are using a HDD instead of a SSD).
            ## center = ndimage.imread(path + filename_img_center)
            ## left = ndimage.imread(path + filename_img_left)
            ## right = ndimage.imread(path + filename_img_right)
            ## X_samples.append(center[70:135,:])
            ## X_samples.append(left[70:135,:])
            ## X_samples.append(right[70:135,:])
            
            center = path + filename_img_center
            left = path + filename_img_left
            right = path + filename_img_right
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
                    image = image[70:135,:]
                    # NOTE: see note above in load_samples() function 
                    #image = batch_samples[i]
                    
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
