import os
import nibabel as nib
import nibabel.processing as nib_pro
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import elasticdeform
#gt = mask
#4d = cine-mri over time

TEST_DIR = "../database/testing"
TEST_PT_NUM = 50
TRAINING_DIR = "../database/training"
PT_NUM = 100

class Frame(Enum):
    FULL = 0
    END_DIASTOLIC = 1
    END_SYSTOLIC = 2

def get_pd_data(testing:bool=False):
    columns = ["PtNum", "ED", "ES", "Group", "Height", "NbFrame", "Weight", "XLen", "YLen", "ZLen", "Time"]
    global data
    data = pd.DataFrame(columns=columns)
    x_start = 1 
    x_end = PT_NUM + 1
    if testing:
        x_start += PT_NUM
        x_end += TEST_PT_NUM
    for i in range(x_start, x_end):
        info = label_reader(i, testing=testing)
        info.insert(0, i)
        pt_dir = pt_dir_from_int(i, testing=testing)
        img_4d = nib.nifti1.load(os.path.join(pt_dir, pt_dir.split("/")[-1]+"_4d.nii")).shape
        for j in img_4d:
            info.append(j)
        data.loc[len(data.index)] = info

def pt_dir_from_int(pt_num:int, testing:bool=False):
    if testing:
        directory = TEST_DIR
    else:
        directory = TRAINING_DIR
    num = str(pt_num).zfill(3)
    filename = "patient"+num
    return os.path.join(directory, filename)

def filepath_from_int(pt_num:int, frame=Frame.FULL, mask=False, testing:bool=False):
    pt_dir = pt_dir_from_int(pt_num, testing=testing)
    filename = pt_dir.split("/")[-1]
    match frame:
        case Frame.FULL:
            filename += "_4d"
        case Frame.END_DIASTOLIC:
            frame_num = int(data.loc[data["PtNum"] == pt_num, "ED"].values[0])
            if frame_num < 10:
                frame_num = "0" + str(frame_num)
            filename += ("_frame" + str(frame_num))
        case Frame.END_SYSTOLIC:
            frame_num = int(data.loc[data["PtNum"] == pt_num, "ES"].values[0])
            if frame_num < 10:
                frame_num = "0" + str(frame_num)
            filename += ("_frame" + str(frame_num))
    if mask and frame.value:
        filename += "_gt.nii"
        return os.path.join(pt_dir, filename)
    filename += ".nii"
    return os.path.join(pt_dir, filename)

def nib_from_int(pt_num:int, frame=Frame.FULL, mask=False, testing:bool=False):
    path = filepath_from_int(pt_num, frame, mask, testing=testing)
    return nib.nifti1.load(path)

def normalise_img(img:np.ndarray): #Normalised to Zero mean and unit variance
    mean_intensity = np.mean(img)
    std_intensity = np.std(img)
    norm_img = (img - mean_intensity) / std_intensity
    return norm_img

def resample_volume(img:nib.nifti1.Nifti1Image, voxel_size=[1.25,1.25,10]):
    voxel_size = np.array(voxel_size) / np.array(list(img.header.get_zooms()))
    resampled_img = nib_pro.resample_to_output(img, voxel_size, mode='wrap')
    return resampled_img.get_fdata()

def resize_img(img:np.ndarray, new_shape):
    shape = img.shape
    if len(shape) == 4:
        shape = img.shape[:3]
        pad_value = img[0, 0, 0, 0]
    else:
        pad_value = img[0, 0, 0]
    if np.any(np.array(shape) < np.array(new_shape)):
        new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
        if len(img.shape) == 4:
            new_shape = new_shape + (img.shape[3],)
    else:
        return img
    res = np.ones(new_shape, dtype=img.dtype) * pad_value
    start = np.array(new_shape) / 2. - np.array(shape) / 2.
    if len(img.shape) == 4:
        res[int(start[0]):int(start[0]) + int(shape[0]),
            int(start[1]):int(start[1]) + int(shape[1]),
            int(start[2]):int(start[2]) + int(shape[2]),
            :] = img
    else:
        res[int(start[0]):int(start[0]) + int(shape[0]),
            int(start[1]):int(start[1]) + int(shape[1]),
            int(start[2]):int(start[2]) + int(shape[2])] = img
    return res

def center_crop(img:np.ndarray, crop_size):
    if len(img.shape) == 4:
        center = np.array(img.shape[:3]) / 2.
        return img[int(center[0] - crop_size[0] / 2.):int(center[0] + crop_size[0] / 2.),
           int(center[1] - crop_size[1] / 2.):int(center[1] + crop_size[1] / 2.),
           int(center[2] - crop_size[2] / 2.):int(center[2] + crop_size[2] / 2.),
           :]
    else:
        center = np.array(img.shape) / 2.
        return img[int(center[0] - crop_size[0] / 2.):int(center[0] + crop_size[0] / 2.),
           int(center[1] - crop_size[1] / 2.):int(center[1] + crop_size[1] / 2.),
           int(center[2] - crop_size[2] / 2.):int(center[2] + crop_size[2] / 2.)]

def random_crop(img:np.ndarray, img_mask:np.ndarray, crop_size):
    if crop_size[0] < img.shape[0]:
        lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
    elif crop_size[0] == img.shape[0]:
        lb_x = 0
    if crop_size[1] < img.shape[1]:
        lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
    elif crop_size[1] == img.shape[1]:
        lb_y = 0
    if crop_size[2] < img.shape[2]:
        lb_z = np.random.randint(0, img.shape[2] - crop_size[2])
    elif crop_size[2] == img.shape[2]:
        lb_z = 0
    return (img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]], 
            img_mask[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1], lb_z:lb_z + crop_size[2]])

def img4d_extraction(pt_num:int, crop_size, testing:bool=False):
    img = resample_volume(nib_from_int(pt_num, testing=testing))
    img = normalise_img(img)
    img = resize_img(img, crop_size)
    img = center_crop(img, crop_size).astype(np.float32)
    return img

def img_extraction(pt_num:int, frame:Frame, crop_size, random:bool=False):
    img = resample_volume(nib_from_int(pt_num, frame))
    img_mask = resample_volume(nib_from_int(pt_num, frame, True))
    img = normalise_img(img)
    img = resize_img(img, crop_size)
    img_mask = resize_img(img_mask, crop_size)
    if random:
        img, img_mask = random_crop(img, img_mask, crop_size)
        img = img.astype(np.float32)
        img_mask = img_mask.astype(np.int8)
    else:
        img = center_crop(img, crop_size).astype(np.float32)
        img_mask = center_crop(img, crop_size).astype(np.int8)
    return img, img_mask

def img_standard(pt_num:int, frame:Frame, crop_size, random:bool=False, n_classes:int=4, testing:bool=False):
    img = resample_volume(nib_from_int(pt_num, frame, testing=testing))
    img_mask = resample_volume(nib_from_int(pt_num, frame, True, testing=testing))
    img = normalise_img(img)
    img = resize_img(img, crop_size)
    img_mask = resize_img(img_mask, crop_size)
    if random:
        img, img_mask = random_crop(img, img_mask, crop_size)
        img = img.astype(np.float32)
        img_mask = img_mask.astype(np.int8)
    else:
        img = center_crop(img, crop_size).astype(np.float32)
        img_mask = center_crop(img, crop_size).astype(np.int8)
    img, img_mask = img_aug(img, img_mask)
    img_mask = tf.one_hot(img_mask, depth=n_classes)
    return img, img_mask

def img_aug(img:np.ndarray, img_mask:np.ndarray):
    flip = np.random.randint(0,1)
    img = np.flip(img, flip) #flip horizontally(0), vertically(1)#
    img_mask = np.flip(img_mask, flip)
    rot = np.random.randint(0,3)
    img = np.rot90(img, rot) #set k, 1k =90 degrees
    img_mask = np.rot90(img_mask, rot)
    img = gamma_correction(img)
    return img, img_mask

def gamma_correction(img:np.ndarray):
    gamma = np.random.uniform(0.5, 2)
    min_value = np.min(img)
    max_value = np.max(img)
    img = (img - min_value) / (max_value - min_value)
    img = np.power(img, gamma)
    return normalise_img(img)

def get_spacing(pt_num:int, testing:bool=False):
    img = nib_from_int(pt_num, testing=testing)
    affine = img.affine
    spacing = affine.diagonal()[:3]
    return spacing

def plot_nimg_data(layer:int, *args): #Max 4 imgs
    if len(args) == 0: return
    imgs = []
    for arg in args:
        if type(arg) == np.ndarray:
            imgs.append(arg)
    if len(imgs) > 1:
        plt.figure(figsize = (10,10))
        for i in range(len(imgs)):
            plt.subplot(2,2,i+1)
            plt.imshow(imgs[i][:,:,layer], cmap = 'gray')
            text = "Image Data" + str(i)
            plt.title(text)
    else:
        plt.figure(figsize = (10,10))
        plt.imshow(imgs[:,:,layer], cmap = 'gray')
        plt.title("Image Data")
    plt.show()

def plot_flat_nimg_data(*args): #Max 4 imgs
    if len(args) == 0: return
    imgs = []
    for arg in args:
        if type(arg) == np.ndarray:
            imgs.append(arg)
    if len(imgs) > 1:
        plt.figure(figsize = (10,10))
        for i in range(len(imgs)):
            plt.subplot(2,2,i+1)
            plt.imshow(imgs[i], cmap = 'gray')
            text = "Image Data" + str(i)
            plt.title(text)
    else:
        plt.figure(figsize = (10,10))
        plt.imshow(imgs, cmap = 'gray')
        plt.title("Image Data")
    plt.show()

def plot_img_overlay(img, overlay): #Overlay is a 2D array, containing x,y coordinates
    plt.figure(figsize = (10,10))
    plt.imshow(img, cmap = 'gray')
    plt.scatter(overlay[:,1], overlay[:,0], color='red', marker=',', alpha=0.4)
    plt.title("Image Data")
    plt.show()

def plot_img_data(img:np.ndarray, layer:int):
    plt.figure(figsize = (10,10))
    plt.imshow(img[:,:,layer], cmap = 'gray')
    plt.title("Image Data")
    plt.show()
    
def plot_ed_es(pt_num:int, layer:int, testing:bool=False): #Layer = y-axis
    ed = nib_from_int(pt_num, Frame.END_DIASTOLIC, testing=testing).get_fdata()
    es = nib_from_int(pt_num, Frame.END_SYSTOLIC, testing=testing).get_fdata()
    ed_mask = nib_from_int(pt_num, Frame.END_DIASTOLIC, True, testing=testing).get_fdata()
    es_mask = nib_from_int(pt_num, Frame.END_SYSTOLIC, True, testing=testing).get_fdata()

    print("Patient:",pt_num)
    print("Image Shape:",ed.shape)
    print("Mask Shape:",ed_mask.shape)

    plt.figure(figsize = (10,10))
    plt.subplot(2,2,1)
    plt.imshow(ed[:,:,layer], cmap = 'gray')
    plt.title("End Diastolic")
    plt.subplot(2,2,2)
    plt.imshow(ed_mask[:,:,layer], cmap = 'gray')
    plt.title("Mask ED")

    plt.subplot(2,2,3)
    plt.imshow(es[:,:,layer], cmap = 'gray')
    plt.title("End Systolic")
    plt.subplot(2,2,4)
    plt.imshow(es_mask[:,:,layer], cmap = 'gray')
    plt.title("Mask ES")

    plt.show()

def label_reader(pt_num:int, testing:bool=False):
    f = open(os.path.join(pt_dir_from_int(pt_num, testing=testing), "Info.cfg"), "r")
    text = f.read()
    f.close()
    lines = text.split('\n')
    info = []
    for i in lines:
        j = i.split(": ")
        if len(j) < 2: continue
        info.append(j[1])
    return info

get_pd_data()