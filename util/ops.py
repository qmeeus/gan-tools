import keras.backend as K


def soft_accuracy(y_true, y_pred):
    soft_probability = K.sigmoid(y_pred)
    hard_prediction = K.cast(K.greater_equal(soft_probability, 0.5), 'float32')
    return K.mean(K.equal(y_true, hard_prediction))


def test_soft_accuracy():
    import numpy as np
    import tensorflow as tf
    from keras.layers import Input

    eps = 1e-12

    def sigmoid(x):
        return (1 + np.exp(-x)) ** -1

    labels = np.random.randint(0, 2, 32)
    predictions = np.random.normal(0, 100, size=(32,))

    expected = np.mean(labels == (sigmoid(predictions) >= 0.5).astype('float'))

    y_true = Input(batch_shape=(32,))
    y_pred = Input(batch_shape=(32,))
    acc = soft_accuracy(y_true, y_pred)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        fd = {y_true: labels, y_pred: predictions}
        result = sess.run(acc, feed_dict=fd)
        print(result, expected)
        assert result - expected < eps