import tensorflow as tf


class MADALINE(tf.keras.Model):
    def __init__(self, num_units):
        super(MADALINE, self).__init__()
        self.layers_list = [
            tf.keras.layers.Dense(1, activation="linear") for _ in range(num_units)
        ]
        self.threshold = tf.keras.layers.Activation("sigmoid")

    def call(self, inputs):
        outputs = [layer(inputs) for layer in self.layers_list]
        combined_output = tf.reduce_sum(outputs, axis=0)
        return self.threshold(combined_output)
