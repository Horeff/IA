import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.datasets import make_blobs, make_circles
from tqdm import tqdm

def sigm(Z):
    return 1/(1+np.exp(-Z))

def log_loss_sm(A,y):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))

def norm(X, max = None, min = None):
    if max is None:
        max = X.max()
    if min is None:
        min = X.min()
    return (X - min)/(max - min)

def flatten(X):
    return X.reshape(X.shape[0], -1)

class neurone:
    def __init__(self, X, y, Xt, Yt, lr = 0.01, act = sigm, loss = log_loss_sm, epoch = 1000):
        self.dim = X.shape[1]
        self.W = np.random.randn(self.dim, 1)
        self.b = np.random.randn(1)
        self.act = act
        self.loss = loss
        self.c_loss = None
        self.c_acc = None
        self.t_loss = None
        self.t_acc = None
        self.train(X, y, Xt, Yt, lr, epoch)

    def model(self, X):
        Z = X.dot(self.W) + self.b
        A = self.act(Z)
        return A

    def gradients(self, A, X, y):
        dW = 1/len(y) * np.dot(X.T, A-y)
        db = 1/len(y) * np.sum(A - y)
        return dW,db

    def update(self, dW, db, W, b, lr):
        self.W = self.W - lr * dW
        self.b = self.b - lr * db

    def train(self, X, y, X_t, y_t, lr, epoch):
        self.c_loss = []
        self.c_acc = []
        self.t_loss = []
        self.t_acc = []
        print("** ENTRAINEMENT **")
        for i in tqdm(range(epoch)):
            A = self.model(X)
            if i%10 == 0:
                #Train
                self.c_loss.append(self.loss(A, y))
                self.c_acc.append(accuracy_score(y, self.predict(X)))
                #Test
                A_t = self.model(X_t)
                self.t_loss.append(self.loss(A_t, y_t))
                self.t_acc.append(accuracy_score(y_t, self.predict(X_t)))
            dW, db = self.gradients(A, X, y)
            self.update(dW, db, self.W, self.b, lr)

        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(self.c_loss, label="train loss")
        plt.plot(self.t_loss, label = "test loss")
        plt.legend()
        plt.subplot(1,2,2)
        plt.plot(self.c_acc, label="train accuracy")
        plt.plot(self.t_acc, label="test accuracy")
        plt.legend()
        plt.show()

    def predict(self, X):
        A = self.model(X)
        return A >= 0.5

class reseau:
    def __init__(self,X, y, X_t = None, y_t = None, learning_rate = 0.01, n_iter = 3000, loss = log_loss, act = sigm, hidden_layers = (16, 16, 16)):
        self.loss = loss
        self.act = act
        self.dimensions = list(hidden_layers)
        self.dimensions.insert(0, X.shape[0])
        self.dimensions.append(y.shape[0])
        np.random.seed(1)
        self.l_dim = len(self.dimensions)
        self.dim = self.dimensions
        self.parametres = {}
        for c in range(1,self.l_dim):
            self.parametres['W'+str(c)] = np.random.randn(self.dimensions[c], self.dimensions[c-1])
            self.parametres['b'+str(c)] = np.random.randn(self.dimensions[c], 1)
        self.C = len(self.parametres) // 2
        self.training_history = np.zeros((int(n_iter)//10, 4))

        # gradient descent
        for i in tqdm(range(n_iter)):
            activations = self.forward_propagation(X)
            gradients = self.back_propagation(y, activations)
            self.update(gradients, learning_rate)
            Af = activations['A' + str(self.C)]
            if i%10 == 0:
                # calcul du log_loss et de l'accuracy
                self.training_history[i//10, 0] = (self.loss(y.flatten(), Af.flatten()))
                y_pred = self.predict(X)
                self.training_history[i//10, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))
                # Pour le test set
                if X_t is not None and y_t is not None:
                    act = self.forward_propagation(X_t)
                    self.training_history[i//10, 2] = (self.loss(y_t.flatten(), act['A' + str(self.C)].flatten()))
                    y_pred = self.predict(X_t)
                    self.training_history[i//10, 3] = (accuracy_score(y_t.flatten(), y_pred.flatten()))
        # Plot courbe d'apprentissage
        self.fig = plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history[:, 0], label='train loss')
        if X_t is not None and y_t is not None:
            plt.plot(self.training_history[:, 2], label='test loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history[:, 1], label='train acc')
        if X_t is not None and y_t is not None:
            plt.plot(self.training_history[:, 3], label='test acc')
        plt.legend()
        plt.show()

    def forward_propagation(self, X):
        activations = {'A0': X}
        for c in range(1, self.C + 1):
            Z = self.parametres['W' + str(c)].dot(activations['A' + str(c - 1)]) + self.parametres['b' + str(c)]
            activations['A' + str(c)] = self.act(Z)
        return activations

    def back_propagation(self, y, activations):
        m = y.shape[1]
        dZ = activations['A' + str(self.C)] - y
        gradients = {}

        for c in reversed(range(1, self.C + 1)):
            gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
            gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            if c > 1:
                dZ = np.dot(self.parametres['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])
        return gradients

    def update(self, gradients, learning_rate):
        for c in range(1, self.C + 1):
            self.parametres['W' + str(c)] = self.parametres['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
            self.parametres['b' + str(c)] = self.parametres['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    def predict(self, X):
        activations = self.forward_propagation(X)
        Af = activations['A' + str(self.C)]
        return Af >= 0.5

class multiclass_reseau:
    def __init__(self, X, y, X_t = None, y_t = None, learning_rate = 0.01, n_iter = 3000, loss = log_loss, act = sigm, hidden_layers = (16, 16, 16)):
        self.classes = np.unique(y)
        self.reseaux = []
        for elem in range(len(self.classes)-1):
            y_p = y == self.classes[elem]
            if y_t is not None: 
                y_t_p = y_t == self.classes[elem]
            else:
                y_t_p = None
            self.reseaux.append(reseau(X, y_p, X_t, y_t_p, learning_rate, n_iter, loss, act, hidden_layers))

    def predict(self, X):
        prediction = np.array([])
        for res in self.reseaux:
            prediction += res.predict(X)[0][0]
        indexes = np.where(prediction == 1)[0]
        print(indexes, prediction)
        if len(indexes) == 0:
            return self.classes[-1]
        else:
            return [self.classes[i] for i in indexes]
            
