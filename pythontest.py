import tensorflow as tf
import numpy as np

# class MyModel(tf.keras.Model):
# 
#   def __init__(self):
#     super().__init__()
#     self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
#     self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
# 
#   def call(self, inputs):
#     x = self.dense1(inputs)
#     return self.dense2(x)
# 
# model = MyModel()
# x = np.random.random((2, 3))
# y = np.random.randint(0, 2, (2, 2))
# model.metrics_names
# inputs = tf.keras.layers.Input(shape=(3,))
# d = tf.keras.layers.Dense(2, name='out')
# output_1 = d(inputs)
# output_2 = d(inputs)
# model = tf.keras.models.Model(
#    inputs=inputs, outputs=[output_1, output_2])
# model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
# model.fit(x, (y, y))


r.model.compile(optimizer="Adam", loss="categorical_crossentropy")
r.model.fit( r.x_name, r.y_name, batch_size = 64, epochs = 25 )

r.model.save("model.h5")
r.model.save('my_model.keras')
