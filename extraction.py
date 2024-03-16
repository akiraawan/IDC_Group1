from utils import *
from data import load_img_train_async
import cv2
from matplotlib.patches import Polygon
from typing import Literal

class Structure(Enum):
    RVC = 1
    LVM = 2
    LVC = 3

def get_chamber_volumes(img:np.ndarray):
    volume_per_voxel = 15.625 #Img MUST be standardised to 1.25 x 1.25 x 10 mm
    lv_voxel = np.sum(img == 3) #left ventricular volume
    rv_voxel = np.sum(img == 1) #left ventricular volume
    return lv_voxel*volume_per_voxel, rv_voxel*volume_per_voxel

def get_contours(img:np.ndarray, structure:Structure, showFig=False):
    img_data = img.astype(np.uint8)
    binary_img = (img_data == structure.value).astype(np.uint8) #left ventricular wall
    contours, _ = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if structure == Structure.LVM:
        if len(contours) < 2:
            return False
        inner_contour = contours[1]
        outer_contour = contours[0].squeeze()
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
        if structure == Structure.LVM:
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
    if showFig:
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
    return np.asarray(thickness_measurements)

def get_circumference(contour):
    return np.asarray(cv2.arcLength(contour, closed=True))

def get_circularity(contour):
    area = cv2.contourArea(contour) * 1.25 * 1.25
    perimeter = cv2.arcLength(contour, closed=True) * 1.25
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

def get_features(img:np.ndarray, structure:Structure):
    thickness = []
    circumference = []
    circularity = []
    for i in range(img.shape[2]):
        result = get_contours(img[:,:,i], structure)
        if type(result) == np.ndarray: pass
        elif result == False: continue
        if structure == Structure.LVM:
            thickness.extend(slice_thickness(result[0], result[1]))
            circumference.append(get_circumference(result[0]))
            circularity.append(get_circularity(result[0]))
        else:
            circumference.append(get_circumference(result))
            circularity.append(get_circularity(result))
    feature_dict = {}
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

def mosteller_method(height, weight): #Height in cm, weight in kg
    return np.sqrt(height * weight / 3600)

def bmi(height, weight): #Height in cm, weight in kg
    height = height / 100
    return weight/height**2

img_data, img_mask_data = asyncio.run(load_img_train_async(30, Frame.END_DIASTOLIC))
features = get_features(img_mask_data, Structure.LVM)
lv_V, rv_V = get_chamber_volumes(img_mask_data)
print(f"THICKNESS: Max = {features['max_thickness']}, Min = {features['min_thickness']}, Std = {features['std_thickness']}, Mean = {features['mean_thickness']}")
print(f"CIRCUMFERENCE: Max = {features['max_circumference']}, Mean = {features['mean_circumference']}")
print(f"CIRCULARITY: Mean = {features['mean_circularity']}")
print(f"Total left ventricular volume: {lv_V}, Total right ventricular volume: {rv_V}")