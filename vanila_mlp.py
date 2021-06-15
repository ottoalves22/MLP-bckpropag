import numpy as np
from numpy.ma import count
import pandas as pd


class MLP:
    def __init__(self, entrada=63, escondida=[1], saida=7):
        self.entradas = entrada
        self.escondida = escondida
        self.saida = saida

        camadas = [self.entradas] + self.escondida + [self.saida]

        pesos = []
        for i in range(len(camadas) - 1):
            w = np.random.rand(camadas[i], camadas[i + 1])  # Gerando a matriz de pesos
            pesos.append(w)
        self.pesos = pesos

        ativacoes = []
        for i in range(len(camadas)):
            temp = np.zeros(camadas[i])  # Return a list of 0 (0, 0, 0, 0)
            ativacoes.append(temp)
        self.ativacoes = ativacoes

        derivadas = []
        for i in range(len(camadas) - 1):
            aux = np.zeros((camadas[i], camadas[i + 1]))
            derivadas.append(aux)
        self.derivadas = derivadas

    def sigmoide(self, t):
        return 1 / (1 + np.exp(-t))

    def sigmoide_derivada(self, t):
        return t * (1.0 - t)

    def feed_foward(self, entradas):
        ativacao = entradas
        self.ativacoes[0] = entradas

        for i, w in enumerate(self.pesos):
            temp = np.dot(ativacao, w)
            ativacao = self.sigmoide(temp)
            self.ativacoes[i + 1] = ativacao

        return ativacao

    def back_propagation(self, erro, v=False):
        for i in reversed(range(len(self.derivadas))):
            ativacoes = self.ativacoes[i + 1]
            delta = erro * self.sigmoide_derivada(ativacoes)
            delta_transformada = delta.reshape(delta.shape[0], -1).T

            ativacao = self.ativacoes[i]
            ativacao = ativacao.reshape(ativacao.shape[0], -1)

            self.derivadas[i] = np.dot(ativacao, delta_transformada)

            erro = np.dot(delta, self.pesos[i].T)

            if v:
                print(f'Derivadas para {i}: {self.derivadas[i]}')

    def gradiente_descendente(self, taxa_erro):
        for i in range(len(self.pesos)):
            pesos = self.pesos[i]
            # print(f'Original w{i} {pesos}')

            derivadas = self.derivadas[i]

            pesos += derivadas * taxa_erro
            # print(f'Atualizado w{i} {pesos}')

    def eqm(self, target, saida):
        return np.average((target - saida) ** 2)

    def treinamento(self, entradas, targets, epocas, taxa_erro):
        for epoca in range(epocas):
            somatorio_erros = 0
            for entrada, target in zip(entradas, targets):
                saida = self.feed_foward(entrada)

                erro = target - saida

                self.back_propagation(erro, v=False)

                self.gradiente_descendente(taxa_erro)

                somatorio_erros += self.eqm(target, saida)

            print(f'Erro {somatorio_erros / len(entradas)} na epoca {epoca}')


def separa_colunas(entrada: pd.DataFrame, x, y, target):
    for i in range(x):
        aux1 = []
        aux2 = []
        for j in range(y):
            if j<63: 
                aux1.append(entrada[j][i]) # monta aux 1 da lista da letra
            else:
                aux2.append(entrada[j][i]) # monta aux 2 da lista de label
        arr = np.array(aux1) # transforma em array
        target = aux2
        print("\n")
        print(arr.reshape(9,7)) #transforma em matrix letra
        print("\n")
        print(target)

if __name__ == '__main__':
    data_input = pd.read_csv('caracteres-limpo.csv', header=None, usecols=[i for i in range(70)])
    target = []
    separa_colunas(data_input, 3, 70, target) # x numero de linhas  y num colunas
