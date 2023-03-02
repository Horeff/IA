from IA import Neurone, 
from IA.res_neur import network
from IA.Res_neur.network import Network
from IA.Res_neur.conv_layer import ConvLayer
from IA.Res_neur.activation_layer import ActivationLayer
from IA.Res_neur.activations import tanh, tanh_prime
from IA.Res_neur.losses import mse, mse_prime

class IA_home:
  def __init__(self):
    pass
  
  def create_res(self,*args):
    return Neurone.reseau(*args)
