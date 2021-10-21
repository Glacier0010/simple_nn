# -*- coding: utf-8 -*-

import numpy as np
import my_layers as ml

class My_net(object):
    def __init__(self,
                input_dim=(3, 32, 32),
                num_filters=32,
                filter_size=7,
                hidden_dim=100,
                num_classes=10,
                weight_scale=1e-3,
                dropout_keep_ratio=0.5,
                reg=0.0):
        self.reg = reg        
        C, H, W = input_dim
        
        #各层定义
        self.conv = ml.convolutional_layer(input_dim, num_filters, filter_size, weight_scale)
        self.fc1 = ml.fully_connected_layer(num_filters*H*W, hidden_dim, weight_scale)        
        self.bn = ml.batch_normalization_layer(hidden_dim)        
        self.relu = ml.relu_layer()
        self.dp = ml.dropout_layer()
        self.fc2 = ml.fully_connected_layer(hidden_dim, num_classes, weight_scale)
        
        #待更新的参数集
        self.params = {"W1": self.conv.W,
                       "b1": self.conv.b,
                       "W2": self.fc1.W,
                       "b2": self.fc1.b,
                       "W3": self.fc2.W,
                       "b3": self.fc2.b,
                       "gamma": self.bn.gamma,
                       "beta": self.bn.beta
                    }
        for k, v in self.params.items():    #强制转换格式
            self.params[k] = v.astype(np.float32)

            
    def loss(self, X, y=None):      #计算损失和梯度
        mode = "test" if y is None else "train"
        
        #前向传播
        out1 = self.conv.conv_forward(X)
        out2 = self.fc1.fc_forward(out1)
        out3 = self.bn.batchnorm_forward(out2, mode)
        out4 = self.relu.relu_forward(out3)
        out5 = self.dp.dropout_forward(out4, mode)
        scores = self.fc2.fc_forward(out5)
        
        if mode == "test":      #测试集上只需要前向传播
            return scores
        
        W1 = self.conv.W
        W2 = self.fc1.W     
        W3 = self.fc2.W
        grads = {}          #字典形式记录所有待更新的参数值

        #反向传播
        loss, softmax_grad = ml.softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(W1 ** 2)        #L2 regularization
        loss += 0.5 * self.reg * np.sum(W2 ** 2)
        loss += 0.5 * self.reg * np.sum(W3 ** 2)
        
        dout, grads["W3"], grads["b3"] = self.fc2.fc_backward(softmax_grad)
        dout = self.dp.dropout_backward(dout)
        dout = self.relu.relu_backward(dout)
        dout, grads["gamma"], grads["beta"] = self.bn.batchnorm_backward(dout)
        dout, grads["W2"], grads["b2"] = self.fc1.fc_backward(dout)
        dout, grads["W1"], grads["b1"] = self.conv.conv_backward(dout)
        
        grads["W1"] += self.reg * W1
        grads["W2"] += self.reg * W2
        grads["W3"] += self.reg * W3

        return loss, grads
    
    def update_params(self, config):    #反向传播完成后更新各参数
        self.conv.W = config["W1"]
        self.conv.b = config["b1"]
        self.fc1.W = config["W2"]
        self.fc1.b = config["b2"]
        self.fc2.W = config["W3"]
        self.fc2.b = config["b3"]
        self.bn.gamma = config["gamma"]
        self.bn.beta = config["beta"]
    


