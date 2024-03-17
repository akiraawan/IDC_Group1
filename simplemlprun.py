from densenetwork import model_builder
from data import *
from sklearn.model_selection import train_test_split

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def make_or_restore_model(shape):
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print("Restoring from", latest_checkpoint)
        return tf.keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    model = model_builder(shape)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005, weight_decay=0.03), loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])
    return model

def plot_stats(history, model_name):
    acc = history.history['accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
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

def run_training(epochs=300, batch_size=30):
    x, y = simple_MLP_train_data(Structure.LVM)
    x = np.array(x)
    y = np.array(y)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
    with strategy.scope():
        model = make_or_restore_model(x.shape[1:])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/ckpt-{epoch}",
            save_freq="epoch"
        )
    ]

    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, callbacks=callbacks, batch_size=batch_size, verbose=2)

    model.save("simpleMLP.keras")
    plot_stats(history, "simpleMLP")

run_training()