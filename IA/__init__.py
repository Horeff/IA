from IA import Neurone
from IA import Res_Neur
import numpy as np

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

  
  
home = IA_home()
