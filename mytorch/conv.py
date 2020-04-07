# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np


class Conv1D():
    def __init__(self, in_channel, out_channel, kernel_size, stride,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channel, in_channel, kernel_size))
        else:
            self.W = weight_init_fn(out_channel, in_channel, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channel)
        else:
            self.b = bias_init_fn(out_channel)

        self.dW = np.zeros(self.W.shape)
        self.db = np.zeros(self.b.shape)
        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, input_size)
        Return:
            out (np.array): (batch_size, out_channel, output_size)
        """
        # scan
        output_size = int(1 + (x.shape[2] - self.kernel_size) // self.stride)
        out = np.zeros((x.shape[0], self.out_channel, output_size))
        for position in range(output_size):
            out[:, :, position] = np.einsum('jkl, ikjl->ij', self.W, x.reshape(x.shape[0], x.shape[1], 1, x.shape[2])
            [:, :, :, position * self.stride:position * self.stride + self.kernel_size])
        self.x = x
        return out

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch_size, out_channel, output_size)
        Return:
            dx (np.array): (batch_size, in_channel, input_size)
        """
        input_size = self.x.shape[2]
        batch_size = delta.shape[0]
        output_size = delta.shape[2]
        dx = np.zeros(self.x.shape)
        print(batch_size, input_size, output_size)
        index = 0
        for time in range(0, input_size - self.kernel_size + 1, self.stride):
            print(delta[:, :, index].shape)
            print(self.x[:, :, time:time + self.kernel_size].shape)
            self.dW += np.tensordot(delta[:, :, index], self.x[:, :, time:time + self.kernel_size], axes=([0], [0]))
            dx[:, :, time:time + self.kernel_size] += np.tensordot(np.transpose(delta, (1, 0, 2))[:, :, index], self.W,
                                                                   axes=([0], [0]))
            index += 1

        # for position in range(input_size):
        #     if position - self.kernel_size < 0:
        #         dx[:, :, position] = np.einsum('ijkl, jkl->ik', delta.reshape(batch_size, self.out_channel, 1, output_size)
        #                                        [:, :, :, :(self.kernel_size - position)], self.W[:, :, ::-1][:, :, :(self.kernel_size - position)]
        #                                        )
        #     else:
        #         dx[:, :, position] = np.einsum('ijkl, jkl->ik', delta.reshape(batch_size, self.out_channel, 1, output_size)
        #                                        [:, :, :, ], self.W[:, :, ::-1])
        #
        # for position in range(self.kernel_size):
        #     self.dW[:, :, position] = np.einsum('ijkl, jkl->ik',
        #                                         self.x.reshape(batch_size, 1, self.in_channel, input_size)
        #                                         [:, :, :, position:position+output_size-1], delta.reshape(batch_size, self.out_channel, 1, output_size))

        # for j in range(self.out_channel):
        #     for x in range(0, input_size):
        #         m = x - self.kernel_size
        #         for i in range(self.in_channel):
        #             for xprime in range(self.kernel_size):
        #                 if 0 <= m+xprime+1 < output_size:
        #                     dx[:, i, x] += self.W[j, i, self.kernel_size-1-xprime] * delta[:, j, m+xprime+1]

        # self.dW[j, i, xprime] += np.dot(delta[:, j, m+xprime+1], self.x[:, i, x])/batch_size

        # print("my weight is ")
        # print(self.W)
        # print(self.W.shape)

        return dx


class Flatten():
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        """
        Argument:
            x (np.array): (batch_size, in_channel, in_width)
        Return:
            out (np.array): (batch_size, in_channel * in width)
        """
        self.b, self.c, self.w = x.shape
        return x.reshape(self.b, self.c * self.w)

    def backward(self, delta):
        """
        Argument:
            delta (np.array): (batch size, in channel * in width)
        Return:
            dx (np.array): (batch size, in channel, in width)
        """
        return delta.reshape(self.b, self.c, self.w)
