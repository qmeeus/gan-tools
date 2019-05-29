import keras.backend as K


def soft_accuracy(y_true, y_pred):
    soft_probability = K.sigmoid(y_pred)
    hard_prediction = K.cast(K.greater_equal(soft_probability, 0.5), 'float32')
    return K.mean(K.equal(y_true, hard_prediction))

