from utils import *
from data import *
import tensorflow as tf

def test_UNET():
    x, y = tensor_test_data()
    x = np.array(x)
    y = np.array(y)
    print(x.shape)
    print(y.shape)
    model = tf.keras.models.load_model(MODEL_PATH)
    results = model.evaluate(x,y,verbose=1)
    print(results)

test_UNET()