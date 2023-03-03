from IA import Neurone
from IA import Res_Neur
from Res_Neur.Network import Network
from Res_Neur.conv_layer import ConvLayer
from Res_Neur.activation_layer import ActivationLayer
from Res_Neur.activations import tanh, tanh_prime
from Res_Neur.losses import mse, mse_prime
from Res_Neur.fc_layer import FCLayer
import numpy as np


  
def create_res(*args):
  return Neurone.reseau(*args)

def conv_res(layers, *args):
  # network
  net = Network()
  for lay in layers:
    net.add(ConvLayer(*lay))
    net.add(ActivationLayer(tanh, tanh_prime))
  # train
  net.use(mse, mse_prime)
  net.fit(*args)
  return net

def lstm_res(layers, *args):
  net = Network()

def example_xor():
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
