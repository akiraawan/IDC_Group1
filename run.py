from network import model_builder
from keras.utils import plot_model

model = model_builder()
# plot_model(model, to_file='model_plot.png', show_shapes=True)