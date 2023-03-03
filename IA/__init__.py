from IA import Neurone, 
from IA.Res_Neur import network
from IA.Res_Neur.network import Network
from IA.Res_Neur.conv_layer import ConvLayer
from IA.Res_Neur.activation_layer import ActivationLayer
from IA.Res_Neur.activations import tanh, tanh_prime
from IA.Res_Neur.losses import mse, mse_prime
from IA.Res_Neur.fc_layer import FCLayer
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

class IA_home:
  def __init__(self):
    pass
  
  def create_res(self,*args):
    return Neurone.reseau(*args)
  
  def conv_res(self, layers, *args):
    # network
    net = Network()
    for lay in layers:
      net.add(ConvLayer(*lay))
      net.add(ActivationLayer(tanh, tanh_prime))
    # train
    net.use(mse, mse_prime)
    net.fit(*args)
    return net
  
  def lstm_res(self, layers, *args):
    net = Network()

  def example_xor(self):
    # training data
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
    
    # network
    net = Network()
    net.add(FCLayer(2, 3))
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(3, 1))
    net.add(ActivationLayer(tanh, tanh_prime))

    # train
    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

    # test
    out = net.predict(x_train)
    return out
  
  def example_mnist(self):
    # load MNIST from server
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # training data : 60000 samples
    # reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
    x_train = x_train.astype('float32')
    x_train /= 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y_train = np_utils.to_categorical(y_train)
    
    # same for test data : 10000 samples
    x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    # Network
    net = Network()
    net.add(FCLayer(28*28, 100))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 50))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(50, 10))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)
    net.add(ActivationLayer(tanh, tanh_prime))

    # train on 1000 samples
    # as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
    net.use(mse, mse_prime)
    net.fit(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

    # test on 3 samples
    out = net.predict(x_test[0:3])
    return {'predicted': out, 'true': y_test[0:3]}
    print("\n")
    print("predicted values : ")
    print(out, end="\n")
    print("true values : ")
    print(y_test[0:3])
