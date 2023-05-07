import cv2
import numpy as np
import tensorflow as tf


class ModelCNN:

    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            tf.keras.layers.MaxPool2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(6, activation='softmax')
        ])

        self.model.build(input_shape=(None, 70, 70, 1))

    def get_number_of_weights(self):
        return self.model.count_params()

    def set_weights(self, best_weights):
        inserted = 0
        for i, layer in enumerate(self.model.layers):
            if i in [1, 3, 5, 6]:
                continue

            layer_weights_shape = layer.get_weights()[0].shape
            layer_biases_shape = layer.get_weights()[1].shape

            # total number of weights in layer = product of all dimensions
            total_n_weights = np.prod(layer_weights_shape)
            total_n_biases = np.prod(layer_biases_shape)

            # get weights and biases from individual
            layer_weights = np.array(best_weights[inserted:inserted + total_n_weights]).reshape(layer_weights_shape)
            layer_biases = np.array(
                best_weights[inserted + total_n_weights:inserted + total_n_weights + total_n_biases]).reshape(
                layer_biases_shape)

            # set weights and biases in model
            layer.set_weights([layer_weights, layer_biases])

            # update inserted
            inserted += total_n_weights + total_n_biases

    def play(self, env, best_weights) -> float:

        self.set_weights(best_weights)

        observation = env.reset()[0]
        done = False
        action_count = 0
        total_reward = 0
        while not done:
            observation = observation[35:170, 20:]
            observation = cv2.resize(observation, dsize=(70, 70), interpolation=cv2.INTER_CUBIC)
            action = self.model.predict(np.array([observation]), verbose=0)

            action = np.argmax(action)

            observation, reward, truncted, terminated, info = env.step(action)
            action_count += 1
            done = terminated or truncted
            total_reward += reward
        env.close()

        return total_reward
