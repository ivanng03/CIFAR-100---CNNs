#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Compare the accuracy between CNN model and KNN model


# In[ ]:


# CNN model


# In[1]:


# Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization


# In[2]:


# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0 # reshape and normalize pixel values
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0
y_train = to_categorical(y_train, 100) # convert labels to one-hot encoding
y_test = to_categorical(y_test, 100)
x_val = x_train[-10000:] # split 10,000 images for validation
y_val = y_train[-10000:]
x_train = x_train[:-10000] # use the remaining 40,000 images for training
y_train = y_train[:-10000]


# In[3]:


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3))) # Convolutional layer with 32 filters, 3x3 kernel, 'same' padding, ReLU activation, input shape (32, 32, 3)
model.add(MaxPooling2D(pool_size=(2, 2))) # Max pooling layer with 2x2 pool size
model.add(Dropout(0.25)) # Dropout layer with 25% dropout rate

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten()) # Flatten layer to transition from convolutional layers to dense layers
model.add(Dense(128, activation='relu')) # Dense layer with 128 units and ReLU activation
model.add(Dropout(0.5)) # Dropout layer with 50% dropout rate
model.add(Dense(100, activation='softmax')) # Dense output layer with 100 units based on the number of classes, softmax activation


# In[4]:


model.summary()


# In[5]:


# Compile the model
# Use the Adam optimizer for training
# Categorical crossentropy loss for multiclass classification
# Monitor categorical accuracy during training

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[6]:


# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint

# ModelCheckpoint callback to save the best model based on validation categorical accuracy
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', save_best_only=True) 
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=15,
                    validation_data=(x_val, y_val), # Validation data for monitoring performance
                    callbacks=[checkpoint]) # List of callbacks


# In[7]:


# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss') 
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss') #label title
plt.xlabel('Epochs') #label x-axis 
plt.ylabel('Loss') #label y-axis
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')  
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy') 
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[8]:


# Evaluate the model
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[ ]:


# KNN model


# In[4]:


# Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# In[5]:


# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train_flat = x_train.reshape(x_train.shape[0], -1).astype('float32') / 255.0  # Flatten images and normalize pixel values
x_test_flat = x_test.reshape(x_test.shape[0], -1).astype('float32') / 255.0


# In[7]:


# Flatten the target variable y_train
y_train_flat = y_train.ravel()

# Create an instance of the KNeighborsClassifier class
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors (k) as needed

# Fit the KNN model on the training data
knn_model.fit(x_train_flat, y_train_flat)  # Use the flattened y_train


# In[12]:


# Predictions on the test set
y_pred = knn_model.predict(x_test_flat)


# In[13]:


# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')


# In[ ]:


#Compare the accuracy of difference Convolutional and Dense layers


# In[14]:


# Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization


# In[15]:


# Load and preprocess data
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0 # reshape and normalize pixel values
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0
y_train = to_categorical(y_train, 100) # convert labels to one-hot encoding
y_test = to_categorical(y_test, 100)
x_val = x_train[-10000:] # split 10,000 images for validation
y_val = y_train[-10000:]
x_train = x_train[:-10000] # use the remaining 40,000 images for training
y_train = y_train[:-10000]


# In[16]:


# Increase the Convolutional and Dense layers
# CNNs model 1
# Convolutional Block 1
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional Block 2
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Convolutional Block 3
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flatten the output for Dense layers
model.add(Flatten())

# Dense layer
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))


# In[17]:


model.summary()


# In[18]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[19]:


# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', save_best_only=True)
history = model.fit(x_train, y_train, 
                    batch_size=64, 
                    epochs=15, 
                    validation_data=(x_val, y_val), 
                    callbacks=[checkpoint])


# In[20]:


# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')  
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')  
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[21]:


# Evaluate the model
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[ ]:





# In[22]:


# Increase the Convolutional and Dense layers
# CNNs model 2
# Convolutional Block 1
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation, input shape (32,32,3)
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.3))  # Dropout with 30% dropout rate

# Convolutional Block 2
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Convolutional Block 3
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Flatten the output for Dense layers
model.add(Flatten())

# Dense Block
model.add(Dense(128, activation='relu'))  # Dense layer with 128 units and ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Output layer
model.add(Dense(100, activation='softmax'))  # Dense output layer with 100 units, softmax activation


# In[23]:


# Checking the model summary
model.summary()


# In[24]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[25]:


# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=15, validation_data=(x_val, y_val), callbacks=[checkpoint])


# In[26]:


# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')  
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')  
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[27]:


# Evaluate the model
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[ ]:


# Adding data augmentation


# In[9]:


# Convolutional Block 1
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation, input shape (32,32,3)
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.3))  # Dropout with 30% dropout rate

# Convolutional Block 2
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Convolutional Block 3
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Flatten the output for Dense layers
model.add(Flatten())

# Dense Block
model.add(Dense(128, activation='relu'))  # Dense layer with 128 units and ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Output layer
model.add(Dense(100, activation='softmax'))  # Dense output layer with 100 units, softmax activation


# In[10]:


# With Data Augmentation
# Create an instance of the ImageDataGenerator class
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False  # randomly flip images
)

# Fit the ImageDataGenerator instance on the training data
data_generator.fit(x_train)

# Create an iterator that yields batches of augmented data and labels
train_iterator = data_generator.flow(x_train, y_train, batch_size=64)


# In[11]:


model.summary()


# In[30]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[31]:


# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=1z5, validation_data=(x_val, y_val), callbacks=[checkpoint])


# In[32]:


# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')  
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[33]:


# Evaluate the model
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[ ]:


# Compare the accuracy between the different epochs.


# In[ ]:


# Epoch = 30


# In[34]:


# Convolutional Block 1
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation, input shape (32,32,3)
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.3))  # Dropout with 30% dropout rate

# Convolutional Block 2
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Convolutional Block 3
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Flatten the output for Dense layers
model.add(Flatten())

# Dense Block
model.add(Dense(128, activation='relu'))  # Dense layer with 128 units and ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Output layer
model.add(Dense(100, activation='softmax'))  # Dense output layer with 100 units (adjust based on the number of classes), softmax activation


# In[35]:


# Create an instance of the ImageDataGenerator class
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
    horizontal_flip=True, # randomly flip images
    vertical_flip=False,  # randomly flip images
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zoom_range=0.2, # randomly zoom image 
    fill_mode='nearest'
)

# Fit the ImageDataGenerator instance on the training data
data_generator.fit(x_train)

# Create an iterator that yields batches of augmented data and labels
train_iterator = data_generator.flow(x_train, y_train, batch_size=64)


# In[36]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[37]:


# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=30, validation_data=(x_val, y_val), callbacks=[checkpoint])


# In[38]:


# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')  
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[39]:


# Evaluate the model
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[ ]:


# Epoch = 50


# In[40]:


# Convolutional Block 1
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation, input shape (32,32,3)
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.3))  # Dropout with 30% dropout rate

# Convolutional Block 2
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Convolutional Block 3
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Flatten the output for Dense layers
model.add(Flatten())

# Dense Block
model.add(Dense(128, activation='relu'))  # Dense layer with 128 units and ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Output layer
model.add(Dense(100, activation='softmax'))  # Dense output layer with 100 units (adjust based on the number of classes), softmax activation


# In[41]:


# Create an instance of the ImageDataGenerator class
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
    horizontal_flip=True, # randomly flip images
    vertical_flip=False,  # randomly flip images
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zoom_range=0.2, # randomly zoom image 
    fill_mode='nearest'
)

# Fit the ImageDataGenerator instance on the training data
data_generator.fit(x_train)

# Create an iterator that yields batches of augmented data and labels
train_iterator = data_generator.flow(x_train, y_train, batch_size=64)


# In[42]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[43]:


# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val), callbacks=[checkpoint])


# In[46]:


# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')  
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[47]:


# Evaluate the model
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[54]:


# Import modules
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns

# Make predictions on the test data
y_pred = model.predict(x_test)

# Convert one-hot encoded labels back to class labels
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)


# In[55]:


# Define the class names
class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
               'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
               'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
               'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
               'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
               'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
               'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
               'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
               'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
               'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
               'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
               'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
               'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
               'worm']


# In[56]:


# Compute and print the classification report
print('Classification Report:\n', classification_report(y_test_labels, y_pred_labels))

# Compute and print the F1 score
f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
print('F1 score:', f1)


# In[57]:


# Plot the confusion matrix using seaborn
plt.figure(figsize=(24, 24))
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[60]:


# Select 50 random images from the test set
idx = np.random.choice(len(x_test), 50, replace=False)
x_sample = x_test[idx]
y_sample = y_test[idx]
# Predict the class probabilities for the images
y_pred = model.predict(x_sample)

# Initialize counts
correct_count = 0
incorrect_count = 0

# Plot the images and the predicted labels
plt.figure(figsize=(15, 15))
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(x_sample[i])
    plt.xticks([])
    plt.yticks([])
    pred_label = class_names[np.argmax(y_pred[i])]
    true_label = class_names[np.argmax(y_sample[i])]
    color = 'green' if pred_label == true_label else 'red'
    plt.xlabel(f'{pred_label} ({true_label})', color=color)
    if pred_label == true_label:
        correct_count += 1
    else:
        incorrect_count += 1
    
plt.tight_layout()
plt.show()

# Print total number of true and false predictions
print('Total Correct Predictions:', correct_count)
print('Total Incorrect Predictions:', incorrect_count)


# In[ ]:


# Image Classification using CNN on cifar-10 dataset


# In[13]:


# Import modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10 #change dataset to CIFAR-10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization


# In[14]:


# Load and preprocess CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data() #change dataset to CIFAR-10
x_train = x_train.reshape(-1, 32, 32, 3).astype('float32') / 255.0  # Reshape and normalize pixel values
x_test = x_test.reshape(-1, 32, 32, 3).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)  # Convert labels to one-hot encoding for 10 classes
y_test = to_categorical(y_test, 10)
# Split 10,000 images for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]  # Use the remaining images for training
y_train = y_train[:-10000]


# In[15]:


# Convolutional Block 1
model = Sequential()
model.add(Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32,32,3)))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation, input shape (32,32,3)
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(64, (3,3), padding='same', activation='relu'))  # 64 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.3))  # Dropout with 30% dropout rate

# Convolutional Block 2
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(128, (3,3), padding='same', activation='relu'))  # 128 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Convolutional Block 3
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Conv2D(256, (3,3), padding='same', activation='relu'))  # 256 filters, 3x3 kernel, 'same' padding, ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(MaxPooling2D(pool_size=(2,2)))  # Max pooling with 2x2 pool size
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Flatten the output for Dense layers
model.add(Flatten())

# Dense Block
model.add(Dense(128, activation='relu'))  # Dense layer with 128 units and ReLU activation
model.add(BatchNormalization())  # Batch normalization
model.add(Dropout(0.5))  # Dropout with 50% dropout rate

# Output layer
model.add(Dense(10, activation='softmax'))  # Dense output layer with 100 units (adjust based on the number of classes), softmax activation


# In[16]:


# Create an instance of the ImageDataGenerator class
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=0, # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
    horizontal_flip=True, # randomly flip images
    vertical_flip=False,  # randomly flip images
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zoom_range=0.2, # randomly zoom image 
    fill_mode='nearest'
)

# Fit the ImageDataGenerator instance on the training data
data_generator.fit(x_train)

# Create an iterator that yields batches of augmented data and labels
train_iterator = data_generator.flow(x_train, y_train, batch_size=64)


# In[17]:


# model summary
model.summary()


# In[18]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


# In[19]:


# Train the model
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_categorical_accuracy', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_val, y_val), callbacks=[checkpoint])


# In[20]:


# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['categorical_accuracy'], label='Training Accuracy')  
plt.plot(history.history['val_categorical_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[21]:


# Evaluate the model
from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)


# In[22]:


from sklearn.metrics import confusion_matrix, f1_score, classification_report
import seaborn as sns

# Make predictions on the test data
y_pred = model.predict(x_test)

# Convert one-hot encoded labels back to class labels
y_test_labels = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test_labels, y_pred_labels)


# In[23]:


# Define the class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[24]:


# Compute and print the classification report
print('Classification Report:\n', classification_report(y_test_labels, y_pred_labels))

# Compute and print the F1 score
f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
print('F1 score:', f1)


# In[25]:


# Plot the confusion matrix using seaborn
plt.figure(figsize=(24, 24))
ax = plt.subplot()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[26]:


# Select 50 random images from the test set
idx = np.random.choice(len(x_test), 50, replace=False)
x_sample = x_test[idx]
y_sample = y_test[idx]
# Predict the class probabilities for the images
y_pred = model.predict(x_sample)

# Initialize counts
correct_count = 0
incorrect_count = 0

# Plot the images and the predicted labels
plt.figure(figsize=(15, 15))
for i in range(50):
    plt.subplot(5, 10, i+1)
    plt.imshow(x_sample[i])
    plt.xticks([])
    plt.yticks([])
    pred_label = class_names[np.argmax(y_pred[i])]
    true_label = class_names[np.argmax(y_sample[i])]
    color = 'green' if pred_label == true_label else 'red'
    plt.xlabel(f'{pred_label} ({true_label})', color=color)
    if pred_label == true_label:
        correct_count += 1
    else:
        incorrect_count += 1
    
plt.tight_layout()
plt.show()

# Print total number of true and false predictions
print('Total Correct Predictions:', correct_count)
print('Total Incorrect Predictions:', incorrect_count)


# In[ ]:




