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

            # TODO:  isso nao parece certo, deveria ser self.pesos?
            pesos += derivadas * taxa_erro
            logger(f'Atualizando w{i} {pesos}\n', "pesos_grad_desc.txt")
            # print(f'Atualizado w{i} {pesos}')

    def eqm(self, target, saida):
        return np.average((target - saida) ** 2)

    def treinamento(self, entradas, targets, epocas, taxa_erro):
        saida = None
        for epoca in range(epocas):
            somatorio_erros = 0
            for entrada, target in zip(entradas, targets):
                saida = self.feed_foward(entrada / np.linalg.norm(entrada))

                erro = target - saida

                self.back_propagation(erro, v=False)

                self.gradiente_descendente(taxa_erro)

                somatorio_erros += self.eqm(target, saida)
            logger(f'Erro {somatorio_erros / len(entradas)} na epoca {epoca} \n', "erro_epoca.txt")

    def predizer(self, x_teste, base):
        resultado = self.feed_foward(x_teste)

        # TODO: ta logando, preciso encapsular depois
        aux = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        logger(f"Predições no CSV {base}:\n", "resultado.txt")
        for i in range(linhas_execucao):
            for j in range(7):
                logger(f"\n{resultado[i][j]} {aux[j]} \n", "resultado.txt")
            logger(f"------------------------------------", "resultado.txt")

        return resultado


def separa_colunas(entrada: pd.DataFrame, linhas, colunas):
    target = []
    saida = []
    tgt = None
    arr = None
    for i in range(linhas):
        aux1 = []
        aux2 = []
        for j in range(colunas):
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


"""
Arquivos possíveis: 
    pesos_iniciais.txt
    pesos_finais.txt
    erro_epoca.txt
"""


def logger(mensagem, arquivo):
    file = open(arquivo, "a")
    file.write(mensagem)
    file.close()


if __name__ == '__main__':
    """------ TREINAMENTO ------"""
    data_input = pd.read_csv('caracteres-limpo.csv', header=None, usecols=[i for i in range(70)])
    linhas_treinamento = 14 # 21 caracteres-limpo originalmente tinha 21 linhas, cortei algumas fora pra fazer o csv de predicao
    colunas = 70
    X, labels = separa_colunas(data_input, linhas_treinamento, colunas)  # x numero de linhas  y num colunas
    entradas = np.array(X)
    targets = np.array(labels)

    mlp = MLP()

    epocas = 1000
    alfa = 0.1
    mlp.treinamento(entradas, targets, epocas, alfa)

    """------ EXECUCAO ------"""

    nome_csv = 'caraceteres_teste_otto.csv'
    entrada_execucao = pd.read_csv(nome_csv, header=None, usecols=[i for i in range(70)])
    linhas_execucao = 7
    X_teste, labels_teste = separa_colunas(entrada_execucao, linhas_execucao, colunas)
    entradas = np.array(X_teste)

    resultados = mlp.predizer(entradas, nome_csv) # aqui entrariam os caracteres sujos


