# IA

## Instalation
```pip install git+[https://https://github.com/Horeff/IA](https://github.com/Horeff/IA).git#egg=IA```

## Utilisation
```
from IA import home
```
* Pour un reseau simple

```
reseau = home.create_res(X_train, y_train, X_test = None, y_test = None, hidden_layers = (16, 16, 16), learning_rate = 0.001, n_iter = 3000, act = sigm, loss = log_loss)
```
* Pour un reseau convolutionnel
```
Conv = home.conv_res(layers, x_train, y_train, n_iter, learning_rate)
```
On recommande n_iter = 1000 et learning_rate = 0.3.
example pour layers : [[(10,10,1), (3,3), 1], 
                       [(8,8,1), (3,3), 1], 
                       [(6,6,1), (3,3), 2)]]
* Pour un reseau lstm
```
lstm = home.lstm_res(*args)
```
* Prediction
```
A = reseau.predict(X_p)
```
Les résultats des reseaux ne sont pas garantis...
