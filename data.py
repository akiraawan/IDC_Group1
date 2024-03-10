from utils import *

CROP_SIZE = (224,224,10)
img = nib_from_int(30, Frame.END_DIASTOLIC)

img_norm = resample_volume(img)
img_data = normalise_img(img_norm.get_fdata())
img_data = resize_img(img_data, CROP_SIZE)

center_data = center_crop(img_data, CROP_SIZE)
random_data = random_crop(img_data, CROP_SIZE)

imgs = np.stack((center_data, random_data), axis=3)
plot_nimg_data(imgs, 5)