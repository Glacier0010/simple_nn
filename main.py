# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt

from my_net import My_net
from solver import Solver
from load_data import get_CIFAR10_data


def main():
    np.random.seed(1012)
    
    #加载cifar-10数据
    data = get_CIFAR10_data()
    for k, v in list(data.items()):
        print(('%s: ' % k, v.shape))

    model = My_net()            #创建模型
    
    solver = Solver(
        model,
        data,
        num_epochs=5,
        batch_size=100,
        optim_config={'learning_rate': 1e-4},
        print_every=1
    )
    
    t0 = time.asctime(time.localtime(time.time()))      #用于记录训练时间
    print("\nBefore training:", t0)
    
    solver.train()              #训练模型
    
    t1 = time.asctime(time.localtime(time.time()))
    print("After training:", t1)
    
    #作loss和accuracy图
    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history, 'o')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history, '-o')
    plt.plot(solver.val_acc_history, '-o')
    plt.legend(['train', 'val'], loc='upper left')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    
    #计算发展集上准确率
    y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
    print('\nValidation set accuracy= ', (y_val_pred == data['y_val']).mean())
    
    #计算测试集上准确率
    y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
    print('\nTest set accuracy= ', (y_test_pred == data['y_test']).mean())
    

if __name__ == '__main__':
    
    main()
