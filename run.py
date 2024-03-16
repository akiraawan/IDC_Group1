from network import model_builder
from data import *
from sklearn.model_selection import KFold, train_test_split
import asyncio

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

SHAPE = CROP_SIZE + (1,)

model = model_builder(shape=SHAPE)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=0.0001), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

rows = asyncio.run(tensor_train_data())
result = np.asarray(rows)
result = result.astype(np.float32)
x = result[:,0,:,:,:]
y = result[:,0,:,:,:]


x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

history = model.fit(x_train, y_train, epochs=100, batch_size=10, verbose=2, validation_data=(x_val,y_val))

model_name = "unet-1"
model.save(model_name)

acc = history.history['sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
val_acc = history.history['val_sparse_categorical_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.savefig(model_name+'_acc.png')
plt.clf()

plt.plot(epochs, loss, 'r', label="Training loss")
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title("Training and validation loss")
plt.legend(loc=0)
plt.savefig(model_name+'_loss.png')
plt.clf()

with open('unet_val.txt', 'a') as file:
    file.write(str(fold_acc)+'\n')
    file.write(str(fold_loss)+'\n')
