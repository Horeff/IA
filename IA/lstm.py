from Res_neur.layer import Layer
from Res_neur.activations import sigmoid, tanh, tanh_prime
import numpy as np

class lstm(Layer):
  def __init__(self, input_size, output_size):
    self.weightf = np.random.rand(input_size, output_size) - 0.5
    self.weightc = np.random.rand(input_size, output_size) - 0.5
    self.weighti = np.random.rand(input_size, output_size) - 0.5
    self.weighto = np.random.rand(input_size, output_size) - 0.5
    self.biasf = np.random.rand(1, output_size) - 0.5
    self.biasc = np.random.rand(1, output_size) - 0.5
    self.biasi = np.random.rand(1, output_size) - 0.5
    self.biaso = np.random.rand(1, output_size) - 0.5

  def forward_propagation(self, input, previous_input, previous_mem):
    self.input = input
    self.prev_input = previous_input
    self.prev_mem = previous_mem
    # Calcul du vecteur d'entree
    data = np.hstack(input, previous_input)
    # Calcul des trois portes
    forgot_gate = sigmoid(np.dot(self.weightf,data) + self.biasf)
    input_gate = sigmoid(np.dot(self.weighti,data) + self.biasi) * tanh(np.dot(self.weightc,data) + self.biasc)
    output_gate = sigmoid(np.dot(self.weighto,data) + self.biaso)
    # Mise à  jour de la mémoire
    mem = (previous_mem * forgot_gate) + input_gate
    # Calcul de la prédiciton
    output = output_gate * tanh(mem)
    return (output, mem)
    
    
  def backward_propagation(self, cache, d_ht, d_ct, learning_rate):
    d_ot = np.dot(d_ht,cache['c'])
    d_ct += np.dot(np.dot(d_ht, d_ot),(1-(tanh(cache['c']))**2))
    d_it = np.dot(d_ct,cache['a'])
    d_ft = np.dot(d_ct,cache['ct-1'])
    d_at = np.dot(d_ct,cache['i'])
    d_ct_1 = np.dot(d_ct,cache['f'])
    


