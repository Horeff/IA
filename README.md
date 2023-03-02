# IA

* Instalation
```pip install git+https://https://github.com/Horeff/IA.git#egg=IA```

* Utilisation
```
from IA import home
# Pour un reseau simple
reseau = home.create_res(X_train, y_train, X_test = None, y_test = None, hidden_layers = (16, 16, 16), learning_rate = 0.001, n_iter = 3000, act = sigm, loss = log_loss
# Prediction
A = reseau.predict(X_p)

# Pour un reseau convolutionnel
Conv = home.conv_res(*args)

# Pour un reseau lstm
lstm = home.lstm_res(*args)
```

Les résultats des reseaux ne sont pas garantis...
