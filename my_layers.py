# -*- coding: utf-8 -*-

import numpy as np

class fully_connected_layer(object):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 weight_scale):
        
        self.W = np.random.normal(0, weight_scale, (input_dim, output_dim))
        self.b = np.zeros(output_dim)
        self.x = None
    
    def fc_forward(self, x):
        w = self.W
        b = self.b
        out = np.dot(x.reshape(x.shape[0], -1), w) + b
        self.x = x
        return out
    
    def fc_backward(self, dout):
        x = self.x
        w = self.W
        
        dx = np.dot(dout, w.T).reshape(x.shape)
        dw = np.dot(x.reshape(x.shape[0], -1).T, dout)
        db = np.dot(dout.T, np.ones(x.shape[0]))    
        return dx, dw, db


class convolutional_layer(object):
    def __init__(self, 
                 input_dim,
                 num_filters,
                 filter_size,
                 weight_scale):        
        C = input_dim[0]        #channel值在前
        
        self.stride = 1
        self.pad = (filter_size - 1) // 2
        self.W = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.b = np.zeros(num_filters)
        self.x = None
        
    def conv_forward(self, x):
        w = self.W
        b = self.b
        stride = self.stride
        pad = self.pad
        
        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')
        H_pad = x_pad.shape[2]
        W_pad = x_pad.shape[3]

        H_out = (H + 2 * pad - HH) // stride + 1
        W_out = (W + 2 * pad - WW) // stride + 1
        out = np.zeros((N, F, H_out, W_out))

        w_row = w.reshape(F, C * HH * WW)

        x_col = np.zeros((C * HH * WW, H_out * W_out))
        for idx in range(N):
            col = 0
            for i in range(0, H_pad - HH + 1, stride):
                for j in range(0, W_pad - WW + 1, stride):
                    x_col[:, col] = x_pad[idx, :, i:i + HH, j:j + WW].reshape(C * HH * WW)
                    col += 1
            out[idx] = (np.dot(w_row, x_col) + b.reshape(F, 1)).reshape(F, H_out, W_out)

        self.x = x
        return out


    def conv_backward(self, dout):
        w = self.W
        b = self.b
        x = self.x
        stride = self.stride
        pad = self.pad

        N, C, H, W = x.shape
        F, C, HH, WW = w.shape
        x_pad = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), 'constant')

        dx_pad = np.zeros_like(x_pad)
        dx = np.zeros_like(x)
        dw = np.zeros_like(w)
        db = np.zeros_like(b)

        H_out = dout.shape[2]
        W_out = dout.shape[3]
        
        for n in range(N):
            for f in range(F):
                db[f] += np.sum(dout[n, f])
                for i in range(H_out):
                    for j in range(W_out):
                        dw[f] += x_pad[n, :, i * stride:i * stride + HH, j * stride:j*stride + WW] * dout[n, f, i, j]
                        dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += w[f] * dout[n, f, i, j]
        dx = dx_pad[:, :, pad:-pad, pad:-pad]

        return dx, dw, db


class relu_layer(object):
    def __init__(self):
        self.x = None
        
    def relu_forward(self, x):
        out = np.maximum(0, x)
        self.x = x
        return out
    
    def relu_backward(self, dout):
        dx = self.x
        dx[dx < 0] = 0
        dx[dx > 0] = 1
        dx *= dout    
        return dx


class batch_normalization_layer(object):
    def __init__(self, input_dim):
        self.gamma = np.ones(input_dim)
        self.beta = np.zeros(input_dim)
        self.eps = 1e-5
        self.momentum = 0.9
        
        self.running_mean = None
        self.running_var = None       
        
        self.x = None
        self.x_norm = None
        self.mean = None
        self.var = None
        self.std = None
        
    def batchnorm_forward(self, x, mode):
        gamma = self.gamma
        beta = self.beta        
        eps = self.eps
        momentum = self.momentum
        
        running_mean = np.zeros(x.shape[1], dtype=x.dtype)
        running_var = np.zeros(x.shape[1], dtype=x.dtype)
    
        if mode == "train":    
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0) + eps
            std = np.sqrt(var)
            x_norm = (x - mean) / std
            out = x_norm * gamma + beta
    
            running_mean = momentum * running_mean + (1 - momentum) * mean
            running_var = momentum * running_var + (1 - momentum) * var
    
            self.x = x
            self.x_norm = x_norm
            self.mean = mean
            self.var = var
            self.std = std    
        elif mode == "test":
            x_norm = (x - running_mean) / np.sqrt(running_var + eps)
            out = x_norm * gamma + beta

        self.running_mean = running_mean
        self.running_var = running_var
    
        return out    
    
    def batchnorm_backward(self, dout):
        gamma = self.gamma
        x = self.x
        x_norm = self.x_norm
        mean = self.mean
        var = self.var 
        std = self.std
      
        N = x.shape[0] * 1.0
        dfdu = dout * gamma
        dfdv = np.sum(dfdu * (x - mean) * -0.5 * var ** -1.5, axis=0)
        dfdw = np.sum(dfdu * -1 / std, axis=0) + dfdv * np.sum(-2/N * (x - mean), axis=0)
        dx = dfdu / std + dfdv * 2/N * (x - mean) + dfdw / N
    
        dgamma = np.sum(dout * x_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
    
        return dx, dgamma, dbeta


class dropout_layer(object):
    def __init__(self, keep_ratio=0.5):        
        self.p = keep_ratio
        self.mask = None

    def dropout_forward(self, x, mode):
        p = self.p        
        if mode == "train":
            mask = np.random.random(x.shape) < p
            out = x * mask / p
            self.mask = mask
        elif mode == "test":
            out = x         
        return out
    
    def dropout_backward(self, dout):
        mask = self.mask
        p = self.p        
        dx = dout * mask / p
        return dx


def softmax_loss(x, y):
    num_train = x.shape[0]
    score = x - np.max(x, axis=1).reshape(num_train, 1)
    prob = np.exp(score) / np.sum(np.exp(score), axis=1).reshape(num_train ,1)
    loss = -np.sum(np.log(prob[np.arange(num_train), y]))
    loss /= num_train
    prob[np.arange(num_train), y] -= 1
    dx = prob / num_train
    return loss, dx
