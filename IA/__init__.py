from IA import Neurone
from IA.Network import Network
from IA.Res_neur.conv_layer import ConvLayer
from IA.Res_neur.activation_layer import ActivationLayer
from IA.Res_neur.activations import tanh, tanh_prime
from IA.Res_neur.losses import mse, mse_prime
from IA.Res_neur.fc_layer import FCLayer
import numpy as np


  
def create_res(X, y, X_t = None, y_t = None, learning_rate = 0.01, n_iter = 3000, loss = Neurone.log_loss, act = Neurone.sigm, hidden_layers = (16, 16, 16)):
  classes = set(y.flatten())
  if len(set) == 2:
    return Neurone.reseau(X, y, X_t = X_t, y_t = y_t, learning_rate = learning_rate, n_iter = n_iter, loss = loss, act = act, hidden_layers = hidden_layers)
  else:
    return Neurone.multiclass_reseau(X, y, X_t = X_t, y_t = y_t, learning_rate = learning_rate, n_iter = n_iter, loss = loss, act = act, hidden_layers = hidden_layers)

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
