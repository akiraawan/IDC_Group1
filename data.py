from utils import *
import tensorflow as tf
from matplotlib.colors import ListedColormap

CROP_SIZE = (224,224,10)
MODEL_PATH = "./unet_models/unet-0.keras"

MASKS_DIR = "./masks"
if not os.path.exists(MASKS_DIR):
    os.makedirs(MASKS_DIR)

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

def masks_creation():
    model = tf.keras.models.load_model(MODEL_PATH)
    for i in range(1, PT_NUM+1):
        img = img4d_extraction(i, CROP_SIZE, True)
        total_frames = data.loc[i-1,'NbFrame']
        for j in range(total_frames):
            img_mask = model.predict(img[:,:,:,j])
            nifti_mask = nib.Nifti1Image(img_mask, affine=np.eye(4))
            nifti_mask.header['pixdim'] = (1.25, 1.25, 10.0, 1.0)
            ptnum = str(i).zfill(3)
            framenum = str(j).zfill(2)
            save_dir = os.path.join(MASKS_DIR, f"/patient{ptnum}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"/patient{ptnum}_frame{framenum}_gt.nii")
            nib.save(nifti_mask, save_path)
    for i in range(PT_NUM+1, PT_NUM+TEST_PT_NUM+1):
        img = img4d_extraction(i, CROP_SIZE, True)
        get_pd_data(True)
        total_frames = data.loc[i-1,'NbFrame']
        for j in range(total_frames):
            img_mask = model.predict(img[:,:,:,j])
            nifti_mask = nib.Nifti1Image(img_mask, affine=np.eye(4))
            nifti_mask.header['pixdim'] = (1.25, 1.25, 10.0, 1.0)
            ptnum = str(i).zfill(3)
            framenum = str(j).zfill(2)
            save_dir = os.path.join(MASKS_DIR, f"/patient{ptnum}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_path = os.path.join(save_dir, f"/patient{ptnum}_frame{framenum}_gt.nii")
            nib.save(nifti_mask, save_path)

masks_creation()

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
