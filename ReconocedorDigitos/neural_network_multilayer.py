# coding: utf-8
import numpy as np
import pandas as pd
import image

class neuro_network():

    def __init__(self, layers, activation = 'tanh'):
        if activation == 'logistic':
            self.activation = lambda x : 1 / (1 + np.exp(-x))
            self.deriv = lambda x : self.activation(x) * (1 - self.activation(x))
        elif activation == 'tanh':
            self.activation = lambda x : np.tanh(x)
            self.deriv = lambda x : 1.0 - np.tanh(x) ** 2

        self.weights = []
        for i in range(1, len(layers) - 1):
            self.weights.append((2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1) * 0.25)
        self.weights.append((2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1) * 0.25)

    def fit(self, X, y, learning_rate = 0.2, epochs = 10000) :
        X = np.atleast_2d(X)
        temp  = np.ones([X.shape[0], X.shape[1] + 1])
        temp[:, 0 : -1] = X
        X = temp
        y = np.array(y)
        for k in range(epochs):
            i = np.random.randint(y.shape[0]);
            a = [X[i]]
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l], self.weights[l])))
            error = y[i] - a[-1]
            deltas = [error * self.deriv(a[-1])]

            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T) * self.deriv(a[l]))
            deltas.reverse()
            for i in range(len(self.weights)) :
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

    def predict(self, x):
        x = np.array(x)
        temp = np.ones(x.shape[0]+1)
        temp[0:-1] = x
        a = temp
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

def convertDecimal(n):
    a = [0]*10
    a[n] = 1
    return a

def DecimalToBinary(n):
    ans = [int(i) for i in str(bin(n)[2:])]
    a = 4 - len(ans)
    return [0] * a + ans

if __name__ == "__main__":
    capas = [64,100,4] # 64 porque representa la imagen de entrada, 100 porque si, y 4 que son los
    # numeros de bits que estamos utilizando
    # https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
    df = pd.read_csv('digits.csv',header=None)
    digitos = neuro_network(capas, 'logistic') # define la red neuronal
    rows = 1797 # del archivo
    x = np.array(df.iloc[0:rows,[i for i in range(0,64)]].values)
    # iloc es para el slicing: en este caso corta desde 0 hasta rows y de las columnas desde 0 a 64
    # 64 es la representación de los números de cada imagen
    y = df.iloc[0:rows,64].values # se toma el ultimo numero de la entrada de los datos, que
    # es el dato que representa los 64 numeros anteriores
    y = np.array([DecimalToBinary(int(y[i])) for i in range(0,len(y))])
    # convierte cada dato de y en binario
    digitos.fit(x,y) # entrena la red
    # muestra el resultado
    """for X in x:
        print (X, [round(i) for i in digitos.predict(X)])"""

    cero = [0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0]
    #cuatro = [0,0,0,4,15,12,0,0,0,0,3,16,15,14,0,0,0,0,8,13,8,16,0,0,0,0,1,6,15,11,0,0,0,1,8,13,15,1,0,0,0,9,16,16,5,0,0,0,0,3,13,16,16,11,5,0,0,0,0,3,11,16,9,0]
    cuatro = [0,0,0,1,12,6,0,0,0,0,0,11,15,2,0,0,0,0,8,16,6,1,2,0,0,4,16,9,1,15,9,0,0,13,15,6,10,16,6,0,0,12,16,16,16,16,1,0,0,1,7,4,14,13,0,0,0,0,0,0,14,9,0,0]
    ocho = [0,0,7,12,10,0,0,0,0,3,16,16,16,9,1,0,0,0,8,16,16,11,1,0,0,0,10,16,16,0,0,0,0,3,16,14,16,4,0,0,0,4,13,0,7,15,0,0,0,4,14,2,2,16,0,0,0,0,6,11,10,5,0,0]
    nueve = [0,0,0,12,14,1,0,0,0,0,9,16,10,5,0,0,0,0,8,13,5,14,0,0,0,0,2,14,16,16,4,0,0,0,0,0,4,10,10,0,0,0,0,0,0,4,16,0,0,0,2,6,4,9,16,0,0,0,1,11,16,15,7,0]

    ans = digitos.predict(cero)
    ans2 = []
    # muestra el resultado en binario
    for i in ans:
        ans2.append(int(round(i)))
    ans2 = list(reversed(ans2)) # lo invierte
    ans = ""
    for i in ans2:
        ans = ans + str(i)
    print (int(ans,2)) # convierte de binario a decimal
    image.showDigit(cero,8,8)
