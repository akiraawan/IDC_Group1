from network import model_builder
from data import *
from sklearn.model_selection import KFold, train_test_split
import asyncio

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

SHAPE = CROP_SIZE + (1,)

model = model_builder(shape=SHAPE)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

rows = asyncio.run(tensor_train_data())
result = np.asarray(rows)
x = result[:,0,:,:,:]
y = result[:,0,:,:,:]

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

inputs = np.concatenate((x_train, x_val), axis=0)
targets = np.concatenate((y_train, y_val), axis=0)

kf = KFold(n_splits=5, shuffle=True)

fold_scores = []
for train_index, val_index in kf.split(inputs, targets):
    # Train the model on the training data for this fold
    history = model.fit(inputs[train_index], targets[train_index], epochs=50, batch_size=10, verbose=2)
    
    # Evaluate the model on the validation data for this fold
    val = model.evaluate(inputs[val_index], targets[val_index], verbose=2)
    fold_scores.append(val)
    model_name = "unet-"+str(len(fold_scores))
    model.save(model_name)

average_val = np.mean(fold_scores)
print(average_val)
