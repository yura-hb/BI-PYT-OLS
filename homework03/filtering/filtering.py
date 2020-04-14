import numpy as np
from collections import namedtuple

CropResultObject = namedtuple('CropResultObject', 'matrix multi_index')

def apply_filter(image: np.array, kernel: np.array) -> np.array:
    # A given image has to have either 2 (grayscale) or 3 (RGB) dimensions
    assert image.ndim in [2, 3]
    # A given filter has to be 2 dimensional and square
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]

    result = np.zeros(image.shape).astype(np.float)

    rot_kernel = np.flip(kernel, (0, 1))

    if image.ndim == 2:
        result = convolution(image, rot_kernel)
    else:
        for channel in range(0, image.shape[2]):
            result[:, :, channel] = convolution(image[:, :, channel], rot_kernel)

    np.clip(result, 0, 255, out=result)

    return result.astype(np.uint8)


def relativeMatrix(shape: np.shape) -> np.array:
    result = np.zeros(shape, dtype=[('x', int), ('y', int)])
    width, height = shape

    center_x, center_y = (int((width - 1) / 2), int((height - 1) / 2))

    it = np.nditer(result, flags=['multi_index'], op_flags=['readwrite'])

    with it:
        for pixel in it:
            x, y = it.multi_index
            it[0]['x'], it[0]['y'] = x - center_x, y - center_y

    return result


def crop(channel: np.array, shape: np.ndarray.shape) -> np.array:
    rel_matrix = relativeMatrix(shape)

    result = np.zeros(shape)

    width, height = channel.shape

    it = np.nditer(channel.copy(), flags=['multi_index'], op_flags=['readonly'])

    left_x, top_y = rel_matrix[0, 0]
    right_x, bottom_y = rel_matrix[-1, -1]

    for pixel in it:

        x, y = it.multi_index

        if x + left_x >= 0 and x + right_x < width and y + top_y >= 0 and y + bottom_y < height:
            result[:] = channel[x + left_x : x + right_x + 1, y + top_y : y + bottom_y + 1]
        else:
            result_it = np.nditer(rel_matrix, flags=['multi_index'], op_flags=['readwrite'])

            for kernel in result_it:
                rel_x, rel_y = kernel['x'], kernel['y']
                masked_x, masked_y = x + rel_x, y + rel_y
                result[result_it.multi_index] = channel[masked_x, masked_y] if masked_x >= 0 and masked_y >= 0 and masked_x < width  and masked_y < height else 0

        yield CropResultObject(result, it.multi_index)


def convolution(channel: np.array, kernel: np.array) -> np.array:
    cropper = crop(channel.astype(np.float), kernel.shape)

    result = np.zeros(channel.shape)

    for crop_res in cropper:
        result[crop_res.multi_index] = np.sum(kernel * crop_res.matrix)

    return result

