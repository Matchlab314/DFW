import numpy as np


# He initialization
# return standard deviation
def he(input_size, output_size):
    return np.sqrt(2 / input_size)


# Xavier initialization
# return standard deviation
def xavier(input_size, output_size):
    return np.sqrt(2 / (input_size + output_size))


class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def set_W_and_b(self, W, b):
        self.W = W
        self.b = b
    
    def update(self, W_grad, b_grad):
        self.W -= self.lr * W_grad
        self.b -= self.lr * b_grad


class Momentum:
    def __init__(self, learning_rate=0.01, alpha=0.9):
        self.lr = learning_rate
        self.alpha = alpha
    
    def set_W_and_b(self, W, b):
        self.W = W
        self.b = b
        self.W_v = np.zeros_like(self.W)
        self.b_v = np.zeros_like(self.b)
    
    def update(self, W_grad, b_grad):
        self.W_v = self.alpha * self.W_v - self.lr * W_grad
        self.W += self.W_v
        self.b_v = self.alpha * self.b_v - self.lr * b_grad
        self.b += self.b_v


class AdaGrad:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
    
    def set_W_and_b(self, W, b):
        self.W = W
        self.b = b
        self.W_h = np.zeros_like(self.W)
        self.b_h = np.zeros_like(self.b)
    
    def update(self, W_grad, b_grad):
        delta = 1e-7
        self.W_h += W_grad * W_grad
        self.W -= self.lr * (1 / (np.sqrt(self.W_h) + delta)) * W_grad
        self.b_h += b_grad * b_grad
        self.b -= self.lr * (1 / (np.sqrt(self.b_h) + delta)) * b_grad


# y = ReLU(x) = max(0, x)
class ReLU:
    def forward(self, x):
        self.x = x
        y = np.maximum(0, x)
        return y
    
    # Please call backward after forward is called.
    def backward(self, dldy):
        dldx = dldy * (self.x > 0)
        return dldx
        
    def __call__(self, x):
        return self.forward(x)


# y = sigmoid(x) = 1 / (1 + exp(-x))
class Sigmoid:
    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        self.y = y
        return y
    
    # Please call backward after forward is called.
    def backward(self, dldy):
        dldx = dldy * self.y * (1 - self.y)
        return dldx

    def __call__(self, x):
        return self.forward(x)


# y = softmax(x)
# x : 2-dimensional array
class Softmax:
    def forward(self, x):
        delta = 1e-7  # to prevent zero division and log(0)
        exp_x = np.exp(x - x.max(axis=1).reshape(x.shape[0], 1))
        y = exp_x / (exp_x.sum(axis=1).reshape(x.shape[0], 1) + delta)
        return y
    
    def __call__(self, x):
        return self.forward(x)


# x, t : 2-dimensional array
class SoftmaxCrossEntropy:
    def forward(self, x, t):
        delta = 1e-7  # to prevent zero division and log(0)
        exp_x = np.exp(x - x.max(axis=1).reshape(x.shape[0], 1))
        y = exp_x / (exp_x.sum(axis=1).reshape(x.shape[0], 1) + delta)
        loss = -(t * np.log(y + delta)).sum() / y.shape[0]
        self.y = y
        self.t = t
        return loss
    
    # Please call backward after forward is called.
    # SoftmaxCrossEntropy.backward works properly only if t is one-hot.
    def backward(self):
        return (self.y - self.t) / self.y.shape[0]
    
    def __call__(self, x, t):
        return self.forward(x, t)


# y = xW + b
# x, dldy : 2-dimensional array
class Affine:
    # initializer : function
    # optimizer : instance
    def __init__(self, input_size, output_size, initializer, optimizer):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.random.normal(0, initializer(self.input_size, self.output_size), (input_size, output_size))
        self.b = np.zeros(output_size)
        self.optimizer = optimizer
        self.optimizer.set_W_and_b(self.W, self.b)

    def forward(self, x):
        self.x = x
        return self.x @ self.W + self.b
    
    # Please call backward after forward is called.
    def backward(self, dldy):
        self.dldb = dldy.sum(axis=0)
        self.dldW = self.x.T @ dldy
        dldx = dldy @ self.W.T
        return dldx
    
    # Please call update after backward is called.
    def update(self):
        self.optimizer.update(self.dldW, self.dldb)
    
    def __call__(self, x):
        return self.forward(x)


# x, dldy : 4-dimensional array
class Conv:
    # input_shape : 3-tuple (channel of data, height of data, width of data)
    # output_shape : 3-tuple (channel of data, height of data, width of data)
    # filter_shape : 4-tuple (number of filters, channel of filter, height of filter, width of filter)
    # initializer : function
    # optimizer : instance
    def __init__(self, input_shape, filter_shape, output_shape, padding, stride, initializer, optimizer):
        # shape check
        if filter_shape[1] != input_shape[0]:
            raise ValueError("Channel of input data and channel of filter don't match.")
        if output_shape[0] != filter_shape[0]:
            raise ValueError("Number of filters and channel of output data don't match.")
        if (input_shape[1] - filter_shape[2] + 2*padding) % stride != 0 or output_shape[1] != (input_shape[1] - filter_shape[2] + 2*padding) // stride + 1:
            raise ValueError("Height of output data is invalid.")
        if (input_shape[2] - filter_shape[3] + 2*padding) % stride != 0 or output_shape[2] != (input_shape[2] - filter_shape[3] + 2*padding) // stride + 1:
            raise ValueError("Width of output data is invalid.")
        
        self.c, self.h, self.w = input_shape
        self.fn, self.c, self.fh, self.fw = filter_shape
        self.fn, self.oh, self.ow = output_shape
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.padding = padding
        self.stride = stride
        
        self.W = np.random.normal(0, initializer(input_shape[0] * input_shape[1] * input_shape[2], output_shape[0] * output_shape[1] * output_shape[2]), filter_shape)
        self.b = np.zeros(filter_shape[0])

        self.optimizer = optimizer
        self.optimizer.set_W_and_b(self.W, self.b)
    
    def forward(self, x):
        # shape check
        if x.shape[1:] != self.input_shape:
            raise ValueError("Shape of x doesn't match input_shape.")
        
        self.n = x.shape[0]

        # pad
        if self.padding != 0:
            x_padded = np.zeros((self.n, self.c, self.h + 2*self.padding, self.w + 2*self.padding))
            x_padded[:, :, self.padding:(-self.padding), self.padding:(-self.padding)] = x
        else:
            x_padded = x
        
        # convert x and W to matrix
        self.x_matrix = np.empty((self.n * self.oh * self.ow, self.c * self.fh * self.fw))
        self.W_matrix = np.empty((self.c * self.fh * self.fw, self.fn))

        for i in range(self.n):
            for j in range(self.oh):
                for k in range(self.ow):
                    self.x_matrix[i*self.oh*self.ow + j*self.oh + k] = x_padded[i, :, (j*self.stride):(j*self.stride + self.fh), (k*self.stride):(k*self.stride + self.fw)].flatten()
        
        for i in range(self.fn):
            self.W_matrix[:, i] = self.W[i].flatten()

        # calculate matrix product
        y_matrix = self.x_matrix @ self.W_matrix

        # convert y to the proper shape
        y_matrix = y_matrix.transpose()
        y = np.empty((self.n, self.fn, self.oh, self.ow))
        for i in range(self.n):
            y[i] = y_matrix[:, (i * self.oh * self.ow):((i + 1) * self.oh * self.ow)].reshape(self.fn, self.oh, self.ow)
        
        # add bias
        y += self.b.reshape(1, self.fn, 1, 1)

        return y
    
    # Please call backward after forward is called.
    def backward(self, dldy):
        # shape check
        if dldy.shape != (self.n, *self.output_shape):
            raise ValueError("Shape of dldy doesn't match output_shape.")
        
        # caluculate gradient
        self.dldb = dldy.sum(axis=(0, 2, 3))

        # convert dldy to matrix
        dldy_matrix = np.empty((self.fn, self.n * self.oh * self.ow))
        for i in range(self.n):
            dldy_matrix[:, (i * self.oh * self.ow):((i + 1) * self.oh * self.ow)] = dldy[i].reshape(self.fn, -1)
        dldy_matrix = dldy_matrix.transpose()

        # calculate gradient
        dldW_matrix = self.x_matrix.T @ dldy_matrix
        dldx_matrix = dldy_matrix @ self.W_matrix.T

        # convert dldW and dldx to proper shape
        self.dldW = np.empty((self.fn, self.c, self.fh, self.fw))
        dldx_padded = np.zeros((self.n, self.c, self.h + 2*self.padding, self.w + 2*self.padding))

        for i in range(self.fn):
            self.dldW[i] = dldW_matrix[:, i].reshape(self.c, self.fh, self.fw)
        
        for i in range(self.n):
            for j in range(self.oh):
                for k in range(self.ow):
                    dldx_padded[i, :, (j*self.stride):(j*self.stride + self.fh), (k*self.stride):(k*self.stride + self.fw)] += dldx_matrix[i*self.oh*self.ow + j*self.oh + k].reshape(self.c, self.fh, self.fw)
        
        # unpad
        if self.padding != 0:
            dldx = dldx_padded[:, :, self.padding:(-self.padding), self.padding:(-self.padding)]
        else:
            dldx = dldx_padded
        
        return dldx
    
    # Please call update after backward is called.
    def update(self):
        self.optimizer.update(self.dldW, self.dldb)
    
    def __call__(self, x):
        return self.forward(x)


# Max Pooling
# x, dldy : 4-dimensional array
class Pooling:
    # size : pooling size
    def __init__(self, size):
        self.size = size
    
    def forward(self, x):
        self.x_shape = x.shape

        if x.shape[2] % self.size != 0:
            raise ValueError("Height of input data is invalid.")
        if  x.shape[3] % self.size != 0:
            raise ValueError("Width of input data is invalid.")
        
        y = np.empty((x.shape[0], x.shape[1], x.shape[2] // self.size, x.shape[3] // self.size))
        self.mask = np.empty_like(x)
        for i in range(x.shape[2] // self.size):
            for j in range(x.shape[3] // self.size):
                y[:, :, i, j] = x[:, :, (i * self.size):((i + 1) * self.size), (j * self.size):((j + 1) * self.size)].max(axis=(2, 3))
                self.mask[:, :, (i * self.size):((i + 1) * self.size), (j * self.size):((j + 1) * self.size)] = (x[:, :, (i * self.size):((i + 1) * self.size), (j * self.size):((j + 1) * self.size)] == y[:, :, i, j].reshape(y.shape[0], y.shape[1], 1, 1))
        return y
    
    def backward(self, dldy):
        dldy_expanded = np.empty(self.x_shape)
        zeros_to_expand = np.zeros((dldy.shape[0], dldy.shape[1], self.size, self.size))
        for i in range(dldy.shape[2]):
            for j in range(dldy.shape[3]):
                dldy_expanded[:, :, (i * self.size):((i + 1) * self.size), (j * self.size):((j + 1) * self.size)] = dldy[:, :, i, j].reshape(dldy.shape[0], dldy.shape[1], 1, 1) + zeros_to_expand
        dldx = self.mask * dldy_expanded
        return dldx
    
    def __call__(self, x):
        return self.forward(x)


class Layer:
    # affine_or_conv, activation_function, pooling : instance
    def __init__(self, affine_or_conv, activation_function, pooling=None):
        self.affine_or_conv = affine_or_conv
        self.activation_function = activation_function
        self.pooling = pooling
    
    def forward(self, x):
        if self.pooling is None:
            return self.activation_function(self.affine_or_conv(x))
        else:
            return self.pooling(self.activation_function(self.affine_or_conv(x)))
    
    # Please call backward after forward is called.
    def backward(self, dldy):
        if self.pooling is None:
            return self.affine_or_conv.backward(self.activation_function.backward(dldy))
        else:
            return self.affine_or_conv.backward(self.activation_function.backward(self.pooling.backward(dldy)))
    
    # Please call update after backward is called.
    def update(self):
        self.affine_or_conv.update()
    
    def __call__(self, x):
        return self.forward(x)


# x : 4-dimensional array
# dldy : 2-dimensional array
class CNN_to_FC:
    def forward(self, x):
        self.x_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    # Please call backward after forward is called.
    def backward(self, dldy):
        return dldy.reshape(self.x_shape)
    
    # Please call update after backward is called.
    def update(self):
        pass
    
    def __call__(self, x):
        return self.forward(x)


class LastLayer:
    # affine, activation_function, activation_function_and_loss_function : instance
    def __init__(self, affine, activation_function, activation_function_and_loss_function):
        self.affine = affine
        self.activation_function = activation_function
        self.activation_function_and_loss_function = activation_function_and_loss_function
    
    def infer(self, x):
        return self.activation_function(self.affine(x))
    
    def loss(self, x, t):
        return self.activation_function_and_loss_function.forward(self.affine(x), t)
    
    # Please call backward after loss is called.
    def backward(self):
        return self.affine.backward(self.activation_function_and_loss_function.backward())
    
    # Please call update after backward is called.
    def update(self):
        self.affine.update()


class Net:
    def __init__(self, *args):
        self.layers = args

    def infer(self, x):
        y = x
        for layer in self.layers:
            if isinstance(layer, LastLayer):
                y = layer.infer(y)
            else:
                y = layer(y)
        return y
    
    def accuracy(self, x, t):
        y = self.infer(x)
        return (y.argmax(axis=1) == t.argmax(axis=1)).sum() / y.shape[0]
    
    def loss(self, x, t):
        y = x
        for layer in self.layers:
            if isinstance(layer, LastLayer):
                y = layer.loss(y, t)
            else:
                y = layer(y)
        return y
    
    # Please call backward after loss is called.
    def backward(self):
        dldx = self.layers[-1].backward()
        for layer in self.layers[-2::-1]:
            dldx = layer.backward(dldx)
        return dldx
    
    # Please call update after backward is called.
    def update(self):
        for layer in self.layers:
            layer.update()