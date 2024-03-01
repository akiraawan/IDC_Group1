import os
import nibabel as nib
from enum import Enum
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#gt = mask
#4d = cine-mri over time

EST_DIR = "../database/testing"
TEST_PT_NUM = 50
DIR = "../database/training"
PT_NUM = 100


class Frame(Enum):
    FULL = 0
    END_DIASTOLIC = 1
    END_SYSTOLIC = 2


class Utils:
    def __init__(self):
        self.columns_precnn = ["PtNum", "ED", "ES", "Group", "Height", "NbFrame", "Weight", "XLen", "YLen", "ZLen", "Time"]
        self.train_val = pd.DataFrame(columns=self.columns_precnn)
        self.get_pd_data()
        self.plot_ed_es(26, 9)
        #self.plot_volume_over_frames()

    def get_pd_data(self):
        for i in range(1, PT_NUM+1):
            info = self.label_reader(i)
            info.insert(0, i)
            pt_dir = self.pt_dir_from_int(i)
            img_4d = nib.nifti1.load(os.path.join(pt_dir, pt_dir.split("/")[-1] + "_4d.nii")).shape
            for j in img_4d:
                info.append(j)
            self.train_val.loc[len(self.train_val.index)] = info

    def pt_dir_from_int(self, pt_num: int):
        num = str(pt_num).zfill(3)
        filename = "patient" + num
        return os.path.join(DIR, filename)

    def filepath_from_int(self, pt_num: int, frame=Frame.FULL, mask=False):
        pt_dir = self.pt_dir_from_int(pt_num)
        filename = pt_dir.split("/")[-1]
        match frame:
            case Frame.FULL:
                filename += "_4d"
            case Frame.END_DIASTOLIC:
                frame_num = int(self.train_val.loc[self.train_val["PtNum"] == pt_num, "ED"].values[0])
                if frame_num < 10:
                    frame_num = "0" + str(frame_num)
                filename += ("_frame" + str(frame_num))
            case Frame.END_SYSTOLIC:
                frame_num = int(self.train_val.loc[self.train_val["PtNum"] == pt_num, "ES"].values[0])
                if frame_num < 10:
                    frame_num = "0" + str(frame_num)
                filename += ("_frame" + str(frame_num))
        if mask and frame.value:
            filename += "_gt.nii"
            return os.path.join(pt_dir, filename)
        filename += ".nii"
        return os.path.join(pt_dir, filename)

    def nib_from_int(self, pt_num: int, frame=Frame.FULL, mask=False):
        path = self.filepath_from_int(pt_num, frame, mask)
        return nib.nifti1.load(path)

    def get_spacing(self, pt_num: int):
        img = self.nib_from_int(pt_num)
        affine = img.affine
        spacing = affine.diagonal()[:3]s
        return spacing

    def volume_from_a_frame(self, pt_num):
        nib_data = self.nib_from_int(pt_num, Frame.END_SYSTOLIC, True).get_fdata()
        spacing = self.get_spacing(pt_num)
        thickness = spacing[2]
        area = spacing[0]*spacing[0]
        volume_per_voxel = area*thickness
        num_wall_voxel = 0
        num_cavity_voxel = 0
        for layer in range(nib_data.shape[2]-1):
            for row in range(nib_data.shape[1]-1):
                for column in range(nib_data.shape[0]-1):
                    if nib_data[column, row, layer] == 2:
                        num_wall_voxel += 1
                    if nib_data[column, row, layer] == 3:
                        num_cavity_voxel += 1
        print("volume per voxel: ", volume_per_voxel)
        return (num_wall_voxel*volume_per_voxel, num_cavity_voxel*volume_per_voxel)

    def plot_volume_over_frames(self):  # Plot the volume changes over time according to patient number
        plt.legend()
        plt.xlabel("Volume")
        plt.ylabel('Time-stepts')
        plt.title('Time-series segmentation for RVC (red), LVM (green), LVC (blue) and their corresponding volume dynamics.')
        # Displaying the plot
        plt.show(block=False)

    def plot_ed_es(self, pt_num: int, layer: int):
        ed = self.nib_from_int(pt_num, Frame.END_DIASTOLIC).get_fdata()
        es = self.nib_from_int(pt_num, Frame.END_SYSTOLIC).get_fdata()
        ed_mask = self.nib_from_int(pt_num, Frame.END_DIASTOLIC, True).get_fdata()
        es_mask = self.nib_from_int(pt_num, Frame.END_SYSTOLIC, True).get_fdata()

        self.volume_from_a_frame(26)

        print("Patient:", pt_num)
        print("Image Shape:", ed.shape)
        print("Mask Shape:", ed_mask.shape)

        plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        plt.imshow(ed[:, :, layer], cmap='gray')
        plt.title("End Diastolic")
        plt.subplot(2, 2, 2)
        plt.imshow(ed_mask[:, :, layer], cmap='gray')
        plt.title("Mask ED")

        plt.subplot(2, 2, 3)
        plt.imshow(es[:, :, layer], cmap='gray')
        plt.title("End Systolic")
        plt.subplot(2, 2, 4)
        plt.imshow(es_mask[:, :, layer], cmap='gray')
        plt.title("Mask ES")

        plt.show()

    def label_reader(self, pt_num: int):
        f = open(os.path.join(self.pt_dir_from_int(pt_num), "Info.cfg"), "r")
        text = f.read()
        f.close()
        lines = text.split('\n')
        info = []
        for i in lines:
            j = i.split(": ")
            if len(j) < 2:
                continue
            info.append(j[1])
        return info


A = Utils()
