from utils import *

CROP_SIZE = (224,224,10)

#Could make img_standard() output both img and mask, make it async, a lot more efficient
def load_img_train(pt_num:int, frame:Frame):
    img = nib_from_int(pt_num, frame)
    img_mask = nib_from_int(pt_num, frame, True)
    img_data = img_standard(img, CROP_SIZE)
    img_mask_data = img_standard(img_mask, CROP_SIZE)
    return img_data, img_mask_data

img_data, img_mask_data = load_img_train(30, Frame.END_DIASTOLIC)
#img_data = np.flip(img_data, 0) #flip horizontally(0), vertically(1)
#img_data = np.rot90(img_data) #set k, 1k =90 degrees

plot_nimg_data(5, img_data, img_mask_data)