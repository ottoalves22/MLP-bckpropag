import numpy as np
import pandas as pd


def sigmoide(x):
    return 1 / (1 + np.exp(-x))  # f(x) = 1/1+e^-x


# essa merda servira pra checar o erro da predicao do sigmoide
def sigmoide_derivada(sx):
    # derivacao aqui https://math.stackexchange.com/a/1225116
    return sx * (1 - sx)  # f'(x) = f(x) . (1-f(x))


def custo(predizido, verdadeiro):
    return verdadeiro - predizido


# cria um dataframe com os valores do csv:
entrada = pd.read_csv('./problemXOR.csv', header=None, names=['e1', 'e2', 's'])

# cria vetor de entrada:
X = np.array([
    [entrada['e1'][0], entrada['e2'][0]],
    [entrada['e1'][1], entrada['e2'][1]],
    [entrada['e1'][2], entrada['e2'][2]],
    [entrada['e1'][3], entrada['e2'][3]]
])

Y = np.array([
    entrada['s'][0],
    entrada['s'][1],
    entrada['s'][2],
    entrada['s'][3]
]).T

# define o formato dos vetor de peso
dados_pesos, camada_input = X.shape

camada_escondida = 1

# inicializar os pesos entre o input e a camada escondida com valores aleatorios
P1 = np.random.random((camada_input, camada_escondida))

camada_saida = len(Y.T)

# inicializar os pesos entre a escondida e a camada de saida com valores aleatorios
P2 = np.random.random((camada_escondida, camada_saida))

num_epocas = 1000

# taxa de aprendizado
alfa = 1.0

for epoca in range(num_epocas):
    camada0 = X
    # foward propagation, daqui pra baixo os pesos vao do inicio pra saida

    # dentro do perceprola passo2
    camada1 = sigmoide(np.dot(camada0, P1))
    camada2 = sigmoide(np.dot(camada1, P2))

    # back propagation, daqui pra baixo os pesos vao do fim pro inicio

    # taxa de erro das predicoes, a diferença entre os resultados em camada2 para os valores corredos em Y_and
    # camada2_erro = custo(camada2, Y)

for x, y in zip(X, Y):
    layer1_prediction = sigmoide(np.dot(P1.T, x))  # Feed the unseen input into trained W.
    prediction = layer2_prediction = sigmoide(np.dot(P2.T, layer1_prediction))  # Feed the unseen input into trained W.
    print((prediction > 0.5), y)
