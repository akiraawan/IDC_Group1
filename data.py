from utils import *
import tensorflow as tf
from matplotlib.colors import ListedColormap

CROP_SIZE = (224,224,10)
MODEL_PATH = "../unet-0.keras"

def tensor_train_data(random:bool=True):
    x_train = []
    y_train = []
    for i in range(1, PT_NUM+1):
        dia = img_standard(i, Frame.END_DIASTOLIC, CROP_SIZE, random)
        sys = img_standard(i, Frame.END_SYSTOLIC, CROP_SIZE, random)
        x_train += [dia[0], sys[0]]
        y_train += [dia[1], sys[1]]
    return x_train, y_train

def tensor_test_data(random:bool=True):
    x_test = []
    y_test = []
    for i in range(1, TEST_PT_NUM+1):
        dia = img_standard(i, Frame.END_DIASTOLIC, CROP_SIZE, random)
        sys = img_standard(i, Frame.END_SYSTOLIC, CROP_SIZE, random)
        x_test += [dia[0], sys[0]]
        y_test += [dia[1], sys[1]]
    return x_test, y_test

def feature_train_data():
    model = tf.keras.models.load_model(MODEL_PATH)
    for i in range(1, PT_NUM+1):
        img = img4d_extraction()
        img_mask = model.predict(img)

# img, img_mask = img_standard(1, Frame.END_DIASTOLIC, CROP_SIZE, True)
# img2, img_mask2 = img_standard(30, Frame.END_DIASTOLIC, CROP_SIZE, True)
# plt.figure(figsize = (10,10))
# plt.subplot(2,2,1)
# plt.imshow(img[:,:,5], cmap='gray')
# plt.title("Image Data1")
# class_indices = np.argmax(img_mask[:,:,5], axis=-1)
# colors = ['white', 'green', 'blue', 'red']
# cmap = ListedColormap(colors)
# plt.subplot(2,2,2)
# plt.imshow(class_indices, cmap=cmap, interpolation='nearest', aspect='equal')
# plt.title("Image Data2")

# plt.subplot(2,2,3)
# plt.imshow(img2[:,:,5], cmap='gray')
# plt.title("Image Data3")

# class_indices = np.argmax(img_mask2[:,:,5], axis=-1)
# plt.subplot(2,2,4)
# plt.imshow(class_indices, cmap=cmap, interpolation='nearest', aspect='equal')
# plt.title("Image Data4")
# plt.show()