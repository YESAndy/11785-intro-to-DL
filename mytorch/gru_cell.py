import numpy as np
from activation import *


class GRU_Cell:
    """docstring for GRU_Cell"""

    def __init__(self, in_dim, hidden_dim):
        self.d = in_dim
        self.h = hidden_dim
        h = self.h
        d = self.d
        self.x = 0

        self.Wzh = np.random.randn(h, h)
        self.Wrh = np.random.randn(h, h)
        self.Wh = np.random.randn(h, h)

        self.Wzx = np.random.randn(h, d)
        self.Wrx = np.random.randn(h, d)
        self.Wx = np.random.randn(h, d)

        self.dWzh = np.zeros((h, h))
        self.dWrh = np.zeros((h, h))
        self.dWh = np.zeros((h, h))

        self.dWzx = np.zeros((h, d))
        self.dWrx = np.zeros((h, d))
        self.dWx = np.zeros((h, d))

        self.z_act = Sigmoid()
        self.r_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wzh, Wrh, Wh, Wzx, Wrx, Wx):
        self.Wzh = Wzh
        self.Wrh = Wrh
        self.Wh = Wh
        self.Wzx = Wzx
        self.Wrx = Wrx
        self.Wx = Wx

    def __call__(self, x, h):
        return self.forward(x, h)

    def forward(self, x, h):
        # input:
        #   - x: shape(input dim),  observation at current time-step
        #   - h: shape(hidden dim), hidden-state at previous time-step
        #
        # output:
        #   - h_t: hidden state at current time-step

        self.x = x
        self.hidden = h

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.

        self.z = self.z_act.forward(self.Wzh.dot(h) + self.Wzx.dot(x))
        self.r = self.r_act.forward(self.Wrh.dot(h) + self.Wrx.dot(x))
        self.h_state = self.Wh.dot(self.r * h) + self.Wx.dot(x)
        self.h_tilda = self.h_act.forward(self.h_state)
        h_t = (1 - self.z) * h + self.z * self.h_tilda

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.h_tilda.shape == (self.h,)
        assert h_t.shape == (self.h,)

        # return h_t
        return h_t

    # This must calculate the gradients wrt the parameters and return the
    # derivative wrt the inputs, xt and ht, to the cell.
    def backward(self, delta):
        # input:
        #  - delta:  shape (hidden dim), summation of derivative wrt loss from next layer at
        #            the same time-step and derivative wrt loss from same layer at
        #            next time-step
        # output:
        #  - dx: Derivative of loss wrt the input x
        #  - dh: Derivative  of loss wrt the input hidden h

        # 1) Reshape everything you saved in the forward pass.
        # 2) Compute all of the derivatives
        # 3) Know that the autograders the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ht = (1-zt) * h_{t-1} + zt * ht_curly
        dh = delta * (1 - self.z)
        dzt = delta * (self.h_tilda - self.hidden)
        dh_tilda = delta * self.z

        # ht_curly = tanh(Wh(rt * h_{t-1}) + WxXt)
        drt = (dh_tilda * self.h_act.derivative()).dot(self.Wh) * self.hidden
        dh += (dh_tilda * self.h_act.derivative()).dot(self.Wh) * self.r
        self.dWh += (dh_tilda * self.h_act.derivative()).reshape(self.h, 1).dot((self.r * self.hidden).reshape(1, self.h))
        self.dWx += (dh_tilda * self.h_act.derivative()).reshape(self.h, 1).dot(self.x.reshape(1, self.d))
        dx = (dh_tilda * self.h_act.derivative()).dot(self.Wx)

        # zt = alpha(Wzx Xt + Wzh h_{t-1} + bz)p;
        dh += (dzt * self.z_act.derivative()).dot(self.Wzh)
        self.dWzh += (dzt * self.z_act.derivative()).reshape(self.h, 1).dot(self.hidden.reshape(1, self.h))
        self.dWzx += (dzt * self.z_act.derivative()).reshape(self.h, 1).dot(self.x.reshape(1, self.d))
        dx += (dzt * self.z_act.derivative()).dot(self.Wzx)
        # dbz?

        # rt = alpha(Wrx Xt + Wrh h_{t-1} + br)
        dh += (drt * self.r_act.derivative()).dot(self.Wrh)
        self.dWrh += (drt * self.r_act.derivative()).reshape(self.h, 1).dot(self.hidden.reshape(1, self.h))
        self.dWrx += (drt * self.r_act.derivative()).reshape(self.h, 1).dot(self.x.reshape(1, self.d))
        dx += (drt * self.r_act.derivative()).dot(self.Wrx)
        # dbz?

        assert dx.shape == (1, self.d)
        assert dh.shape == (1, self.h)

        # return dx, dh
        return dx, dh

