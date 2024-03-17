from utils import *
import cv2
from matplotlib.patches import Polygon
from typing import Literal
from scipy.stats import skew, kurtosis

class Structure(Enum):
    RVC = 1
    LVM = 2
    LVC = 3

def get_chamber_volumes(img:np.ndarray): #takes in 3D arrays
    volume_per_voxel = 15.625 #Img MUST be standardised to 1.25 x 1.25 x 10 mm
    lv_voxel = np.sum(img == 3) #left ventricular volume
    lvm_voxel = np.sum(img == 2) #left ventricular wall
    rv_voxel = np.sum(img == 1) #right ventricular volume
    return lv_voxel*volume_per_voxel, lvm_voxel*volume_per_voxel, rv_voxel*volume_per_voxel

def get_contours(img:np.ndarray, structure:Structure, showFig=False):
    img_data = img.astype(np.uint8)
    binary_img = (img_data == structure.value).astype(np.uint8) #left ventricular wall
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    if structure == Structure.LVM:
        if len(contours) < 2 or len(hierarchy) < 1:
            return False
        for i, h in enumerate(hierarchy[0]):
            if h[3] == -1:
                if i != 0:
                    return False
        inner_contour = contours[1]
        outer_contour = contours[0].squeeze()
        if len(inner_contour.squeeze()) < 3 or len(outer_contour) < 3:
            return False
    else:
        if len(contours) < 1:
            return False
        elif len(contours) > 1:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            contours = contours[0]
            outer_contour = contours.squeeze()
        else:
            outer_contour = np.asarray(contours).squeeze()
    if showFig:
        if structure == Structure.LVM or len(contours) == 2:
            contours = [outer_contour, inner_contour]
        else:
            contours = [outer_contour]
        fig, ax = plt.subplots()
        ax.imshow(img_data, cmap='gray')
        for i in range(len(contours)):
            colour = 'r' if i == 0 else 'b'
            polygon = Polygon(contours[i].squeeze(), edgecolor=colour, linewidth=2, fill=False)
            ax.add_patch(polygon)
        plt.show()
    if structure == Structure.LVM:
        return outer_contour, inner_contour
    else:
        return outer_contour

def slice_thickness(outer_contour, inner_contour, showFig=False):
    thickness_measurements = []
    for pt in outer_contour:
        pt = tuple([int(round(pt[0])), int(round(pt[1]))]) #wont work if its not converted to int type
        distance = cv2.pointPolygonTest(inner_contour, pt, measureDist=True)
        thickness_measurements.append(abs(distance))
    return np.asarray(thickness_measurements)

def get_circumference(contour):
    return np.asarray(cv2.arcLength(contour, closed=True))

def get_circularity(contour):
    area = cv2.contourArea(contour) * 1.25 * 1.25
    perimeter = cv2.arcLength(contour, closed=True) * 1.25
    circularity = float((4 * np.pi * area) / (perimeter ** 2))
    return circularity

def mosteller_method(height, weight): #Height in cm, weight in kg
    return np.sqrt(height * weight / 3600)

def bmi(height, weight): #Height in cm, weight in kg
    height = height / 100
    return weight/height**2

def volume_changes(lv_volumes, rv_volumes, lw_mass):
    time_points = range(len(lv_volumes))
    plt.plot(time_points, lv_volumes, color='blue', label='Left Ventricular Volume')
    plt.plot(time_points, rv_volumes, color='red', label='Right Ventricular Volume')
    plt.plot(time_points, lw_mass, color='green', label='Left Ventricular Mass')
    plt.xlabel('Time (frames)')
    plt.ylabel('Volume (cm3)')
    plt.title('Volumes Over Time')
    plt.legend()
    plt.show()

def volumes_stats(volumes):
    volumes = np.array(volumes)
    stats_dict = {}
    stats_dict['mean_v'] = np.mean(volumes)
    stats_dict['median_v'] = np.median(volumes)
    stats_dict['min_v ']= np.min(volumes)
    stats_dict['max_v'] = np.max(volumes)
    stats_dict['std_dev_v'] = np.std(volumes)
    stats_dict['skewness_v'] = skew(volumes)
    stats_dict['kurtosis_v'] = kurtosis(volumes)
    return stats_dict

def get_ef(end_diastolic_volume, end_systolic_volume): #EDV = max(volume), ESV = min(volume) at TIME of max/min at LVC
    stroke_volume = end_diastolic_volume - end_systolic_volume
    ejection_fraction = (stroke_volume / end_diastolic_volume) * 100
    return ejection_fraction

def draw_thickness(outer_contour, inner_contour, img_data, thickness_measurements):
    contours = [outer_contour, inner_contour]
    fig, ax = plt.subplots()
    ax.imshow(img_data, cmap='gray')
    for i in range(2):
        colour = 'r' if i == 0 else 'b'
        polygon = Polygon(contours[i].squeeze(), edgecolor=colour, linewidth=2, fill=False)
        ax.add_patch(polygon)
    for pt, thickness in zip(outer_contour, thickness_measurements):
        thickness = thickness*1.25
        ax.text(pt[0], pt[1], f'{thickness:.3} mm', color='yellow', fontsize=8, ha='center')
    plt.show()

def standardise_mask(array_4d):
    num_classes = array_4d.shape[3]
    quantized_data = np.empty(array_4d.shape, dtype=np.int8)
    for class_idx in range(num_classes):
        class_data = array_4d[:, :, :, class_idx]
        class_data = np.round(class_data)
        quantized_data[:, :, :, class_idx] = class_data.astype(np.int8)
    return np.argmax(quantized_data, axis=-1)

def get_features(img:np.ndarray, structure:Structure):
    thickness = []
    circumference = []
    circularity = []
    for i in range(img.shape[2]):
        result = get_contours(img[:,:,i], structure, True)
        if type(result) == np.ndarray: pass
        elif result == False: continue
        if structure == Structure.LVM:
            print(img.shape)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            contour_image = img.copy() if len(img.shape) == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(img, result, -1, (0, 255, 0), 2) 
            cv2.imshow('Contours', contour_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            thickness.extend(slice_thickness(result[0], result[1]))
            circumference.append(get_circumference(result[0]))
            circularity.append(get_circularity(result[0]))
        else:
            circumference.append(get_circumference(result))
            circularity.append(get_circularity(result))
    feature_dict = {}
    print("Circumference:")
    print(circumference)
    print("Circularity:")
    print(circularity)
    circularity = np.asarray(circularity).squeeze()
    circumference = np.asarray(circumference).squeeze()
    circumference = circumference * 1.25
    feature_dict['mean_circularity'] = circularity.mean()
    feature_dict['max_circumference'] = circumference.max()
    feature_dict['mean_circumference'] = circumference.mean()
    if structure == Structure.LVM:
        thickness = np.asarray(thickness).squeeze()
        thickness = thickness * 1.25
        feature_dict['max_thickness'] = thickness.max()
        feature_dict['min_thickness'] = thickness.min()
        feature_dict['std_thickness'] = thickness.std()
        feature_dict['mean_thickness'] = thickness.mean()
    return feature_dict

def get_dynamic_features(img_4d:nib.nifti1.Nifti1Image, structure:Structure):
    lv_volumes = []
    rv_volumes = []
    lv_mass = []
    img_data = img_4d.get_fdata()
    for i in range(img_4d.shape[3]):
        img = img_data[:,:,:,i]
        lv_V, lvw_V, rv_V = get_chamber_volumes(img)
        if float(lv_V) != 0.0:
            lv_volumes.append(float(lv_V))
        if float(rv_V) != 0.0:
            rv_volumes.append(float(rv_V))
        if float(lvw_V) != 0.0:
            lv_mass.append(float(lvw_V))
    feature_dict = {}
    lv_volumes = np.asarray(lv_volumes)
    rv_volumes = np.asarray(rv_volumes)
    lv_mass = np.asarray(lv_mass)
    print(f"lv_volume: {lv_volumes}")
    print(f"rv_volume: {rv_volumes}")
    print(f"lv_mass: {lv_mass}")
    match structure:
        case Structure.RVC:
            feature_dict['max_v'] = float(rv_volumes.max())
            feature_dict['min_v'] = float(rv_volumes.min())
            stats_dict = volumes_stats(rv_volumes)
            feature_dict = {**feature_dict, **stats_dict}
        case Structure.LVC:
            feature_dict['max_v'] = float(lv_volumes.max())
            feature_dict['min_v'] = float(lv_volumes.min())
            stats_dict = volumes_stats(lv_volumes)
            feature_dict = {**feature_dict, **stats_dict}
        case Structure.LVM:
            index = np.where(lv_volumes == lv_volumes.min())[0]
            feature_dict['max_v'] = float(lv_mass.max())
            feature_dict['min_v'] = float(lv_mass[index])
            stats_dict = volumes_stats(lv_mass)
            feature_dict = {**feature_dict, **stats_dict}
    feature_dict['ef'] = float(get_ef(feature_dict['max_v'], feature_dict['min_v']))
    feature_dict['ratio_min_lv_rv'] = float(lv_volumes.min()) / float(rv_volumes.min())
    feature_dict['ratio_min_rv_lm'] = float(rv_volumes.min()) / float(lv_mass.min())
    feature_dict['ratio_min_lm_lv'] = float(lv_mass.min()) / float(lv_volumes.min())
    feature_dict['stepdiff_min_lv_rv'] = int(np.where(lv_volumes == float(lv_volumes.min()))[0] - np.where(rv_volumes == float(rv_volumes.min()))[0])
    feature_dict['stepdiff_max_lv_rv'] = int(np.where(lv_volumes == float(lv_volumes.max()))[0] - np.where(rv_volumes == float(rv_volumes.max()))[0])
    return feature_dict
    
# img_data, img_mask_data = img_extraction(30, Frame.END_DIASTOLIC, CROP_SIZE, True)
# features = get_features(img_mask_data, Structure.LVM)
# lv_V, lv_M, rv_V = get_chamber_volumes(img_mask_data)
# print(f"THICKNESS: Max = {features['max_thickness']}, Min = {features['min_thickness']}, Std = {features['std_thickness']}, Mean = {features['mean_thickness']}")
# print(f"CIRCUMFERENCE: Max = {features['max_circumference']}, Mean = {features['mean_circumference']}")
# print(f"CIRCULARITY: Mean = {features['mean_circularity']}")
# print(f"Left ventricular volume: {lv_V}, Left ventricular mass: {lv_M} Right ventricular volume: {rv_V}")