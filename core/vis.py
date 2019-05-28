import numpy as np
import matplotlib.pyplot as plt


def image_grid(array, columns):
    nr, height, width, channels = array.shape
    rows = nr // columns
    assert nr == rows * columns  # otherwise not a rectangle
    result = array.reshape(rows, columns, height, width, channels) \
        .swapaxes(1, 2) \
        .reshape(height * rows, width * columns, channels)
    return result


def show_image_grid(array, columns):
    grid = image_grid(array, columns)
    plt.imshow(grid)


def show_gan_image_predictions(gan, nr, columns=8, image_shape=None, filename=None, ax=None):
    images = gan.generate(nr)
    ax = ax or plt.subplot()
    if image_shape is not None:
        if len(image_shape) == 2:
            image_shape = image_shape + (1,)
        images = images.reshape(-1, image_shape[0],image_shape[1], image_shape[2])
    grid = image_grid(images, columns)
    grid = 0.5 * grid + 0.5
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(np.squeeze(grid))
    plt.tight_layout()
    # plt.savefig(filename or 'image_{}.eps'.format(np.random.randint(10**6, 10**7-1)))
