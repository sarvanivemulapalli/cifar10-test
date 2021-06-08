import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import argparse


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

parser = argparse.ArgumentParser(description='Running CNN to classify CIFAR10')
parser.add_argument('--epochs', help='number of epochs to run', default='1')
parser.add_argument('--batch_size', help='iteration batch size', default='128')
args = parser.parse_args()

num_classes = 10
batch_size = int(args.batch_size)
epochs = int(args.epochs)

print('batch_size:', batch_size)
print('num_classes:', num_classes)
print('epochs:', epochs)



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, 
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(test_images, test_labels))

score = model.evaluate(test_images, test_labels, verbose=0)
print('test_loss:', score[0])
print('test_accuracy:', score[1])
model.save('model.h5')
