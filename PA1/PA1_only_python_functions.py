import numpy as np

def only_python_softmax(input :np.ndarray):
    input = input - np.max(input, axis=1, keepdims=True)
    output = np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
    return output


def only_python_cross_entropy_loss(x_softmax:np.ndarray, label:np.ndarray):
    batch_size = x_softmax.shape[0]
    return -np.sum( label * np.log(x_softmax + 1e-7 ) ) / batch_size


def img2col(input: np.ndarray, kernel_h, kernel_w, stride=1, padding=0):
    batch_size, channel_size, height, width = input.shape
    out_h = (height + 2*padding - kernel_h) // stride + 1
    out_w = (height + 2*padding - kernel_w) // stride + 1

    img = np.pad(input, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((batch_size, channel_size, kernel_h, kernel_w, out_h, out_w))

    for y in range(kernel_h):
        y_max = y + stride * out_h
        for x in range(kernel_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0,4,5,1,2,3).reshape(batch_size*out_h*out_w, -1)

    return col


def col2img(col: np.ndarray, input_shape, kernel_h, kernel_w, stride=1, padding=0):
    batch_size, channel_size, height, width = input_shape
    out_h = (height + 2*padding - kernel_h) // stride + 1
    out_w = (height + 2*padding - kernel_w) // stride + 1
    col = col.reshape(batch_size, out_h, out_w, channel_size, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((batch_size, channel_size, height + 2*padding + stride - 1, width + 2*padding + stride - 1))
    for y in range(kernel_h):
        y_max = y + stride*out_h
        for x in range(kernel_w):
            x_max = x + stride*out_w
            img[:,:,y:y_max:stride, x:x_max:stride] += col[:,:,y,x,:,:]

    return img[:,:,padding:height+padding, padding:width + padding]