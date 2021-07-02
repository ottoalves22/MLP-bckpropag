import math

import numpy as np
import pandas as pd


class MLP_OTIN69:
    def __init__(self, alfa, epocas, num_camadas_escondidas=1):
        self.num_camadas_escondidas = num_camadas_escondidas
        self.epocas = epocas
        self.taxa_aprendizado = alfa

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

    def sigmoide(self, t):
        return 1 / (1 + np.exp(-t))

    def sigmoide_derivada(self, t):
        return t * (1.0 - t)

    def eqm(self, target, saida):
        return np.average((target - saida) ** 2)

    def inicializa_pesos(self, x, y):
        num_exemplos, num_atributos = x.shape
        _, num_saidas = y.shape
        # camada oculta
        limite = 1 / math.sqrt(num_atributos)
        self.pesosEscondidos = np.random.uniform(-limite, limite, (
            num_atributos, self.num_camadas_escondidas))  # pesos do meio terao formato entrada x camadas escondidas
        self.pesosEscondidos0 = np.zeros((1, self.num_camadas_escondidas)) # bias

        # camada de saida
        limite = 1 / math.sqrt(self.num_camadas_escondidas)
        self.pesosSaida = np.random.uniform(-limite, limite, (
            self.num_camadas_escondidas,
            num_saidas))  # pesos do meio terao formato camadas escondidas x rotulos de saida
        self.pesosSaida0 = np.zeros((1, num_saidas)) # bias

    def feedfoward(self, x):
        # feed
        entrada_camada_escondida = x.dot(self.pesosEscondidos)  # + self.pesosEscondidos0 # SUM xi x wj + (bias)
        resultados_camada_escondida = self.sigmoide(entrada_camada_escondida)  # chama função de ativação (sigmoide)

        # foward
        entrada_camada_saida = resultados_camada_escondida.dot(self.pesosSaida)  # sem bias, vide linha 32
        predicao = self.sigmoide(entrada_camada_saida)  # solta uma lista de 7 elementos

        return predicao

    def backpropagation(self, x, y):
        for i in range(self.epocas):
            entrada_camada_escondida = x.dot(self.pesosEscondidos)  # + self.pesosEscondidos0 # SUM xi x wj + (bias)
            resultados_camada_escondida = self.sigmoide(entrada_camada_escondida)  # chama função de ativação (sigmoide)

            # foward
            entrada_camada_saida = resultados_camada_escondida.dot(self.pesosSaida)  # sem bias, vide linha 32
            feedfoward_y = self.sigmoide(entrada_camada_saida)  # solta uma lista de 7 elementos

            # camada de saida
            gradiente_erro_saida = self.gradient(y, feedfoward_y)  * self.sigmoide_derivada(resultados_camada_escondida.dot(self.pesosSaida))# calculo do erro
            print(gradiente_erro_saida)
            # nao sei bem que porra é essa
            gradiente_s = x.dot(self.pesosEscondidos).T.dot(gradiente_erro_saida)
            gradiente_s0 = np.sum(gradiente_erro_saida, axis=0, keepdims=True)

            # camada escondida
            gradiente_erro_escondida = (gradiente_erro_saida.dot(self.pesosSaida.T) * self.sigmoide(x.dot(self.pesosEscondidos)))
            gradiente_e = x.T.dot(gradiente_erro_escondida)
            gradiente_e0 = np.sum(gradiente_erro_escondida, axis=0, keepdims=True)

            # Atualizacao de pesos
            self.pesosSaida -= self.taxa_aprendizado * gradiente_s
            self.pesosSaida0 -= self.taxa_aprendizado * gradiente_s0 # atualiza os bias
            self.pesosEscondidos -= self.taxa_aprendizado * gradiente_e
            self.pesosEscondidos0 -= self.taxa_aprendizado * gradiente_e0 # atualiza os bias

    def treino(self, x, y):
        self.inicializa_pesos(x, y)
        self.backpropagation(x, y)

    def predizer(self, x, y):
        resultado = self.feedfoward(x)
        return resultado


def matriz_confusao_multiclasse(preditos, targets):
    matriz = [[0 for i in range(7)] for j in range(7)]
    resultado = []
    for i in range(len(preditos)):
        resultado.append(contabilizar(preditos[i]))

    for i in range(len(targets)):
        d = resultado[i].index(1)
        j = targets[i].index(1)
        if d == j:
            matriz[j][j] += 1
        else:
            matriz[d][j] += 1
    matriz_final = pd.DataFrame(matriz, index=['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                                columns=['Ap', 'Bp', 'Cp', 'Dp', 'Ep', 'Fp', 'Gp'])
    matriz_conf = np.array(matriz)
    logger(f'Matriz de Confusão: \n{matriz_final}\n', 'resultado.txt')
    return matriz_conf  # retorna como arranjos


def contabilizar(resultados):  # traz resultados em termos de caracter para cada resultado da mlp
    caracteres = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    previsoes = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(resultados)):
        if resultados[i] == max(resultados):
            previsoes[i] += 1
    return previsoes


def precisao_matriz_confusao(rotulos, matriz_conf):
    coluna = matriz_conf[:, rotulos]  # corta verticalmente e pega valores da coluna rotulos
    return matriz_conf[rotulos, rotulos] / coluna.sum()


def recall_matriz_conf(rotulos, matriz_conf):
    linha = matriz_conf[rotulos, :]  # corta horizontalmente e pega valores da linha rotulos
    return matriz_conf[rotulos, rotulos] / linha.sum()


def precisao_media(matriz):
    linhas, colunas = matriz.shape
    soma_precisao = 0
    for rotulo in range(linhas):
        soma_precisao += precisao_matriz_confusao(rotulo, matriz)
    return soma_precisao / linhas


def recall_medio(matriz):
    linhas, colunas = matriz.shape
    soma_recalls = 0
    for rotulo in range(linhas):
        soma_recalls += recall_matriz_conf(rotulo, matriz)
    return soma_recalls / linhas


def acuracia(matriz):
    soma_diagonal = matriz.trace()  # soma os valores diagonais da matriz
    soma_total_elementos = matriz.sum()  # soma quantidade total de predicoes
    return soma_diagonal / soma_total_elementos


def estatisticas_matriz_confusao(matriz):
    aux = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    print("Rotulo    Precisao  Revocação")
    for i in range(7):
        print(f"{aux[i]}     {precisao_matriz_confusao(i, matriz):9.2f}     {recall_matriz_conf(i, matriz):6.3f}")

    print("Precisão total:", precisao_media(matriz))

    print("Recall total:", recall_medio(matriz))

    print("Acuracia total:", acuracia(matriz))


def separa_colunas(entrada: pd.DataFrame, linhas, col):
    target = []
    saida = []
    tgt = None
    arr = None
    for i in range(linhas):
        aux1 = []
        aux2 = []
        for j in range(col):
            if j < 63:
                aux1.append(entrada[j][i])  # monta aux 1 da lista da letra
            else:
                aux2.append(entrada[j][i])  # monta aux 2 da lista de label
        arr = np.array(aux1)
        arr = arr.tolist()
        tgt = aux2

        saida.append(arr)
        target.append(tgt)

    return saida, target


def logger(mensagem, arquivo):
    file = open(arquivo, "a")
    file.write(mensagem)
    file.close()


def logger_predicao(resultado, base):
    aux = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    logger(f"Predições no CSV {base}:\n", "resultado.txt")
    for i in range(linhas_execucao):
        for j in range(7):
            logger(f"\n{resultado[i][j]} {aux[j]} \n", "resultado.txt")
        logger(f"------------------------------------", "resultado.txt")


if __name__ == '__main__':
    """------ TREINAMENTO ------"""
    data_input = pd.read_csv('caracteres-limpo.csv', header=None, usecols=[i for i in range(70)])
    linhas_treinamento = 21  # 21 caracteres-limpo originalmente tinha 21 linhas, cortei algumas fora pra fazer o csv de predicao
    colunas = 70
    X, labels = separa_colunas(data_input, linhas_treinamento, colunas)  # x numero de linhas  y num colunas
    entradas = np.array(X)
    # print(entradas)
    targets = np.array(labels)
    epocas = 8000
    alfa = 0.852  # varia de 0.1 à 1

    mlp = MLP_OTIN69(alfa, epocas)

    mlp.inicializa_pesos(entradas, targets)

    logger(f'Épocas: {epocas}, taxa de aprendizado: {alfa} ', 'parametros_iniciais.txt')
    mlp.treino(entradas, targets)

    nome_csv = 'caracteres-ruido.csv'
    data_input = pd.read_csv(nome_csv, header=None, usecols=[i for i in range(70)])
    X, labels = separa_colunas(data_input, linhas_treinamento, colunas)  # x numero de linhas  y num colunas
    entradas = np.array(X)
    targets = np.array(labels)
    mlp.treino(entradas, targets)

    """------ EXECUCAO ------"""

    nome_csv = 'caracteres-ruido.csv'
    entrada_execucao = pd.read_csv(nome_csv, header=None, usecols=[i for i in range(70)])
    linhas_execucao = 21
    X_teste, labels_teste = separa_colunas(entrada_execucao, linhas_execucao, colunas)
    entradas = np.array(X_teste)

    resultados = mlp.predizer(entradas, nome_csv)  # aqui entrariam os caracteres sujos
    # resultado_precisao = precisao(resultados, np.array(labels_teste))
    # matriz_confusao(resultados, labels_teste)  # Caso queira o log da matriz de confusão no resultado.txt
    matriz = matriz_confusao_multiclasse(resultados, labels_teste)
    estatisticas_matriz_confusao(matriz)
