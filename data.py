from utils import *
import asyncio
import tensorflow as tf

CROP_SIZE = (224,224,10)

async def load_img_train_async(pt_num:int, frame:Frame):
    img = await asyncio.create_task(get_img_train(pt_num, frame))
    img_mask = await asyncio.create_task(get_mask_train(pt_num, frame))
    img = await asyncio.create_task(img_standard(img, CROP_SIZE))
    img_mask = await asyncio.create_task(img_standard(img_mask, CROP_SIZE))
    return img, img_mask

def panda_train_data(series, frame:Frame):
    pt_num = series['PtNum']
    print(pt_num)
    future = asyncio.run_coroutine_threadsafe(load_img_train_async(pt_num, frame), loop)
    print(future.result())
    return future.result()

async def tensor_train_data():
    ptnums = data.loc[:, ['PtNum']]
    results = []
    #loop = asyncio.get_event_loop()
    for _, row in ptnums.iterrows():
        task = asyncio.create_task(load_img_train_async(row['PtNum'], Frame.END_DIASTOLIC))
        results.append(task)
        task = asyncio.create_task(load_img_train_async(row['PtNum'], Frame.END_SYSTOLIC))
        results.append(task)
    rows = await asyncio.gather(*results)
    return rows
    # df = pd.concat([df1, df2], axis=0)
    # x = df.iloc[:, 0]
    # y = df.iloc[:, 1]
    # print(x)
    # print(y)
    # x_tensor = tf.constant(x.values, tf.float32)
    # y_tensor = tf.constant(y.values, tf.float32)
    # return x_tensor, y_tensor

#tensor_train_data()

#img_data, img_mask_data = load_img_train(30, Frame.END_DIASTOLIC)
#img_data = np.flip(img_data, 0) #flip horizontally(0), vertically(1)
#img_data = np.rot90(img_data) #set k, 1k =90 degrees
#plot_nimg_data(5, img_data, img_mask_data)