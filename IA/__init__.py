from IA import Neurone, 
from IA.Res_Neur import network
from IA.Res_Neur.network import Network
from IA.Res_Neur.conv_layer import ConvLayer
from IA.Res_Neur.activation_layer import ActivationLayer
from IA.Res_Neur.activations import tanh, tanh_prime
from IA.Res_Neur.losses import mse, mse_prime

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
