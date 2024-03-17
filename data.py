from utils import *
from extraction import *
import tensorflow as tf
from matplotlib.colors import ListedColormap
import math

if not os.path.exists(MASK_TESTING_DIR):
    os.makedirs(MASK_TESTING_DIR)
if not os.path.exists(MASK_TRAINING_DIR):
    os.makedirs(MASK_TRAINING_DIR)

def tensor_train_data(random:bool=True):
    x_train = []
    y_train = []
    for i in range(1, PT_NUM+1):
        dia = img_standard(i, Frame.END_DIASTOLIC, CROP_SIZE, random)
        sys = img_standard(i, Frame.END_SYSTOLIC, CROP_SIZE, random)
        x_train += [dia[0], sys[0]]
        y_train += [dia[1], sys[1]]
    return x_train, y_train

def simple_MLP_train_data(structure:Structure):
    x_test = []
    y_test = []
    label_to_number = {label: number for number, label in enumerate(CLASS_LABELS)}
    for i in range(1, PT_NUM+1):
        dia_mask = img_mask_extraction(i, Frame.END_DIASTOLIC, CROP_SIZE)
        sys_mask = img_mask_extraction(i, Frame.END_SYSTOLIC, CROP_SIZE)
        dia_dict = get_features(dia_mask, structure)
        sys_dict = get_features(sys_mask, structure)
        static_dict = {}
        for key, value in dia_dict.items():
            match key:
                case "mean_circularity":
                    static_dict["mean_circularity"] = np.mean([value, sys_dict[key]])
                case "max_circumference":
                    static_dict["max_circumference"] = np.max([value, sys_dict[key]])
                case "mean_circumference":
                    static_dict["mean_circumference"] = np.mean([value, sys_dict[key]])
            if structure == Structure.LVM:
                match key:
                    case "max_thickness":
                        static_dict["max_thickness"] = np.max([value, sys_dict[key]])
                    case "min_thickness":
                        static_dict["min_thickness"] = np.min([value, sys_dict[key]])
                    case "std_thickness":
                        static_dict["std_thickness"] = math.sqrt((value**2 + sys_dict[key]**2) / 2)
                    case "mean_thickness":
                        static_dict["mean_thickness"] = np.mean([value, sys_dict[key]])
        df = training_data_DF[['Group', 'Height', 'Weight']].loc[i-1]
        static_dict['bmi'] = bmi(float(df['Height']), float(df['Weight']))
        static_dict['body_surface_area'] = mosteller_method(float(df['Height']), float(df['Weight']))
        features = np.array(list(static_dict.values()))
        print(features)
        x_test.append(features)
        y_test.append(label_to_number[str(df['Group'])])
    return x_test, y_test

def MLP_train_data(structure:Structure):
    x_test = []
    y_test = []
    label_to_number = {label: number for number, label in enumerate(CLASS_LABELS)}
    for i in range(1, PT_NUM+1):
        mask_4d = load_mask(i)
        dia_mask = img_mask_extraction(i, Frame.END_DIASTOLIC, CROP_SIZE)
        sys_mask = img_mask_extraction(i, Frame.END_SYSTOLIC, CROP_SIZE)
        print(i)
        dyn_dict = get_dynamic_features(mask_4d, structure)
        dia_dict = get_features(dia_mask, structure)
        sys_dict = get_features(sys_mask, structure)
        static_dict = {}
        for key, value in dia_dict.items():
            match key:
                case "mean_circularity":
                    static_dict["mean_circularity"] = np.mean([value, sys_dict[key]])
                case "max_circumference":
                    static_dict["max_circumference"] = np.max([value, sys_dict[key]])
                case "mean_circumference":
                    static_dict["mean_circumference"] = np.mean([value, sys_dict[key]])
            if structure == Structure.LVM:
                match key:
                    case "max_thickness":
                        static_dict["max_thickness"] = np.max([value, sys_dict[key]])
                    case "min_thickness":
                        static_dict["min_thickness"] = np.min([value, sys_dict[key]])
                    case "std_thickness":
                        static_dict["std_thickness"] = math.sqrt((value**2 + sys_dict[key]**2) / 2)
                    case "mean_thickness":
                        static_dict["mean_thickness"] = np.mean([value, sys_dict[key]])
        feature_dict = {**dyn_dict, **static_dict}
        df = training_data_DF[['Group', 'Height', 'Weight']].loc[i-1]
        feature_dict['bmi'] = bmi(float(df['Height']), float(df['Weight']))
        feature_dict['body_surface_area'] = mosteller_method(float(df['Height']), float(df['Weight']))
        features = np.array(list(feature_dict.values()))
        x_test.append(features)
        y_test.append(label_to_number[str(df['Group'])])
    return x_test, y_test

def masks_creation():
    model = tf.keras.models.load_model(MODEL_PATH)
    for i in range(1, PT_NUM+1):
        ptnum = str(i).zfill(3)
        save_path = os.path.join(MASK_TRAINING_DIR, f"patient{ptnum}_gt.nii")
        if os.path.exists(save_path):
            continue;
        img = nib_from_int(i)
        total_frames = training_data_DF.loc[i-1,'NbFrame']
        all_frames = []
        for j in range(int(total_frames)):
            frame = img.get_fdata()[:,:,:,j]
            frame = img4d_extraction(nib.Nifti1Image(frame, img.affine), CROP_SIZE)
            frame = np.expand_dims(frame, axis=0)
            img_mask = model.predict(frame)
            img_mask = img_mask[0,:,:,:,:]
            img_mask = standardise_mask(img_mask)
            all_frames.append(img_mask)
        all_frames = np.transpose(np.asarray(all_frames), axes=(1, 2, 3, 0))
        affine = np.diag([1.25, 1.25, 10.0, 1.0])
        nifti_mask = nib.Nifti1Image(all_frames, affine=affine, dtype=np.int8)
        nib.save(nifti_mask, save_path)
    for i in range(PT_NUM+1, PT_NUM+TEST_PT_NUM+1):
        ptnum = str(i).zfill(3)
        save_path = os.path.join(MASK_TESTING_DIR, f"patient{ptnum}_gt.nii")
        if os.path.exists(save_path):
            continue;
        img = nib_from_int(i, testing=True)
        total_frames = testing_data_DF.loc[i-PT_NUM-1,'NbFrame']
        all_frames = []
        for j in range(int(total_frames)):
            frame = img.get_fdata()[:,:,:,j]
            frame = img4d_extraction(nib.Nifti1Image(frame, img.affine), CROP_SIZE)
            frame = np.expand_dims(frame, axis=0)
            img_mask = model.predict(frame)
            img_mask = img_mask[0,:,:,:,:]
            img_mask = standardise_mask(img_mask)
            all_frames.append(img_mask)
        all_frames = np.transpose(np.asarray(all_frames), axes=(1, 2, 3, 0))
        affine = np.diag([1.25, 1.25, 10.0, 1.0])
        nifti_mask = nib.Nifti1Image(all_frames, affine=affine, dtype=np.int8)
        nib.save(nifti_mask, save_path)

# x_train, y_train = MLP_train_data(Structure.LVM)
# x_train = np.asarray(x_train)
# y_train = np.asarray(y_train)
# print(x_train.shape)
# print(y_train.shape)

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
