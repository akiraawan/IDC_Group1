from network import model_builder
from data import *
from sklearn.model_selection import KFold

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

SHAPE = CROP_SIZE + (1,)

checkpoint_dir = "./ckpt"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def make_or_restore_model(fold_index:int):
    checkpoints = [checkpoint_dir + "/" + name for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        checkpoint_name = os.path.basename(latest_checkpoint)
        if checkpoint_name == f"fold_{fold_index+1}":
            print("Restoring from", latest_checkpoint)
            return tf.keras.models.load_model(latest_checkpoint)
    print("Creating a new model")
    model = model_builder(shape=SHAPE)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, weight_decay=0.00001), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['categorical_accuracy'])
    return model

def plot_stats(history, val, model_name):
    acc = history.history['categorical_accuracy']
    loss = history.history['loss']
    val_loss = val[0]
    val_acc = val[1]
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

def run_training(epochs=170, batch_size=10):
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

        with strategy.scope():
            model = make_or_restore_model(fold_index)
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_dir + f"/fold_{fold_index+1}",
                save_weights_only=True,
                save_freq=10 * len(x_train) // strategy.num_replicas_in_sync
            )
        ]

        history = model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks, batch_size=batch_size, verbose=2)
        val = model.evaluate(x_val, y_val, verbose=1)

        model_name = "unet-"+str(fold_index)
        model.save(model_name+".keras")

        val_scores.append(val)
        plot_stats(history, val, model_name)
    with open('unet_val.txt', 'a') as file:
        file.write(str(val_scores)+'\n')

run_training()