from network import model_builder
from data import *
from sklearn.model_selection import KFold

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

SHAPE = CROP_SIZE + (1,)

x, y = tensor_train_data()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
val_scores = []
x = np.array(x)
y = np.array(y)
for fold_index, (train_index, val_index) in enumerate(kf.split(x)):
    print(f"Fold: {fold_index+1}")
    x_train = x[train_index]
    x_val = x[val_index]
    y_train = y[train_index]
    y_val = y[val_index]

    model = model_builder(shape=SHAPE)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=150, batch_size=20, verbose=2)
    val = model.evaluate(x_val, y_val, verbose=1)

    val_scores.append(val)
    model_name = "unet-"+str(fold_index)
    model.save(model_name)

    acc = history.history['categorical_accuracy']
    loss = history.history['loss']
    val_loss = val['val_loss']
    val_acc = val['val_categorical_accuracy']
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
    file.write(str(val_scores)+'\n')
