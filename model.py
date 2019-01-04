import csv 
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn

from keras.models import Model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.utils import plot_model

lines = [] # Stores lines read in csv file
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile) 
    for line in reader:
        lines.append(line)

# Separates path to images in training and validation sets
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                source_path = batch_sample[0]
                # Use windows 10 separator
                filename =source_path.split('\\')[-1]
                # Redefine path of each image
                current_path = '../data/IMG/' + filename
                
                # Read the image in current path    
                image = mpimg.imread(current_path)
                # Append image to the list of images
                images.append(image)
                measurement = float(batch_sample[3])
                # Append measurement to the list of measurements
                measurements.append(measurement)

                # Flip the image, store the flipped image and the modified measurement
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                measurement_flipped = -measurement
                measurements.append(measurement_flipped)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Create a Sequential object to define the neural network
model = Sequential()

# Lambda layer to normalize the input data
model.add(Lambda(lambda x: (x/255.0)-0.5 , input_shape=(160,320,3)))

# Cropping image in the y axis to avoid feeding undesired features to the network
model.add(Cropping2D(cropping=((70,25), (0,0))))

# Set of 5 convolutional layers. First 3 have a sumbsampling of 2x2, the others are the typical 1x1
model.add(Conv2D(24,(5,5), activation='relu', strides= (2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(36,(5,5),activation='relu', strides = (2,2)))
model.add(Conv2D(48,(5,5), activation='relu', strides=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(Dropout(0.5))

# Flatter the data to enter new fully connected layers
model.add(Flatten())

# Set of 3 fully connected layers 
model.add(Dense(100))
#model.add(Dropout(0.3))
model.add(Dense(50))
#model.add(Dropout(0.3))
model.add(Dense(10))

# Output layer
model.add(Dense(1))

# Compile deep neural network with loss function Mean Square Error (mse) and adam optimizer 
model.compile(loss='mse', optimizer='adam')

plot_model(model, to_file='model.png', show_shapes=True)
'''
#history_object = model.fit(X_train, y_train, verbose=1, validation_split=0.2, shuffle= True, nb_epoch = 20)

# Use fit_generator to train the model
history_object = model.fit_generator(train_generator, steps_per_epoch= \
            int(np.ceil(len(train_samples)/32)), validation_data=validation_generator, \
            validation_steps=int(np.ceil(len(validation_samples)/32)), verbose= 1, nb_epoch=20)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

# Save trained model in file
model.save('model2.h5')
'''