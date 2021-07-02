import numpy as np
import pandas as pd


class MLP:
    def __init__(self, entrada=63, escondida=[1], saida=7):

        logger(f'Neuronios de entrada: {entrada}, camadas escondidas: {escondida}, neuronios de saida: {saida}, ',
               'parametros_iniciais.txt')

        self.entradas = entrada
        self.escondida = escondida
        self.saida = saida

        self.camadas = [self.entradas] + self.escondida + [self.saida]

        pesos = []
        for i in range(len(self.camadas) - 1):
            w = np.random.rand(self.camadas[i], self.camadas[i + 1])  # Gerando a matriz de pesos
            pesos.append(w)
        self.pesos = pesos

        # TODO: Ta feiao, preciso arrumar
        logger(f'Pesos inciais:\n {pesos}', 'pesos_iniciais.txt')

        ativacoes = []
        for i in range(len(self.camadas)):
            temp = np.zeros(self.camadas[i])  # Return a list of 0 (0, 0, 0, 0)
            ativacoes.append(temp)
        self.ativacoes = ativacoes

        derivadas = []
        for i in range(len(self.camadas) - 1):
            aux = np.zeros((self.camadas[i], self.camadas[i + 1]))
            derivadas.append(aux)
        self.derivadas = derivadas

    def sigmoide(self, t):
        return 1 / (1 + np.exp(-t))

    def sigmoide_derivada(self, t):
        return t * (1.0 - t)

    def feed_foward(self, entradas):
        outputs = []
        for i in self.camadas:
            output = self.sigmoide(entradas)
            outputs.append(output)

        return outputs

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

            # logger(f'Atualizando w{i} {pesos}\n', "pesos_grad_desc.txt")
            # print(f'Atualizado w{i} {pesos}')

    def eqm(self, target, saida):
        return np.average((target - saida) ** 2)

    def treinamento(self, entradas, targets, epocas, taxa_erro, parada_antecipada=False):
        saida = None
        for epoca in range(epocas):
            # print(epoca)
            somatorio_erros = 0
            somatorio_erros_aux = 0
            for entrada, target in zip(entradas, targets):
                somatorio_erros_aux = somatorio_erros

                saida = self.feed_foward(entrada)

                erro = target - saida

                self.back_propagation(erro, v=False)

                self.gradiente_descendente(taxa_erro)

                somatorio_erros += self.eqm(target, saida)
                print(epoca)
                if somatorio_erros/len(entradas) - somatorio_erros_aux/len(entradas) < 0.00001 and somatorio_erros > 0:
                    break

            logger(f'Erro {somatorio_erros/len(entradas)} na epoca {epoca} \n', "erro_epoca.txt")

        logger(f'Pesos finais:\n {self.pesos} \n', 'pesos_finais.txt')

    def predizer(self, x_teste, base):
        resultado = self.feed_foward(x_teste)

        logger_predicao(resultado, base)  # logando certinho os resultados

        return resultado


""" ----- Funções auxiliares -----"""


def precisao(resultado, y_teste):
    for y in range(len(resultado)):
        if np.array_equal(resultado[y], y_teste[y]):
            resultado += 1
    accuracy = resultado / y_teste.shape[0]

    logger(f'\n Precisão:\n', 'resultado.txt')
    for i in accuracy:
        for j in i:
            logger(f'{j} ', 'resultado.txt')
        logger(f'\n------------------------------\n', 'resultado.txt')
    return accuracy


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
    matriz_final = pd.DataFrame(matriz, index=['A', 'B', 'C', 'D', 'E', 'F', 'G'], columns=['Ap', 'Bp', 'Cp', 'Dp', 'Ep', 'Fp', 'Gp'])
    matriz_conf = np.array(matriz)
    logger(f'Matriz de Confusão: \n{matriz_final}\n', 'resultado.txt')
    return matriz_conf # retorna como arranjos


def contabilizar(resultados): # traz resultados em termos de caracter para cada resultado da mlp
    caracteres = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    previsoes = [0, 0, 0, 0, 0, 0, 0]
    for i in range(len(resultados)):
        if resultados[i] == max(resultados):
            previsoes[i] += 1
    return previsoes


def precisao_matriz_confusao(rotulos, matriz_conf):
    coluna = matriz_conf[:, rotulos] # corta verticalmente e pega valores da coluna rotulos
    return matriz_conf[rotulos, rotulos]/coluna.sum()


def recall_matriz_conf(rotulos, matriz_conf):
    linha = matriz_conf[rotulos, :] # corta horizontalmente e pega valores da linha rotulos
    return matriz_conf[rotulos, rotulos]/linha.sum()


def precisao_media(matriz):
    linhas, colunas = matriz.shape
    soma_precisao = 0
    for rotulo in range(linhas):
        soma_precisao += precisao_matriz_confusao(rotulo, matriz)
    return soma_precisao/linhas


def recall_medio(matriz):
    linhas, colunas = matriz.shape
    soma_recalls = 0
    for rotulo in range(linhas):
        soma_recalls += recall_matriz_conf(rotulo, matriz)
    return soma_recalls / linhas


def acuracia(matriz):
    soma_diagonal = matriz.trace() # soma os valores diagonais da matriz
    soma_total_elementos = matriz.sum() # soma quantidade total de predicoes
    return soma_diagonal/soma_total_elementos


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
    linhas_treinamento = 1  # 21 caracteres-limpo originalmente tinha 21 linhas, cortei algumas fora pra fazer o csv de predicao
    colunas = 70
    X, labels = separa_colunas(data_input, linhas_treinamento, colunas)  # x numero de linhas  y num colunas
    entradas = np.array(X)
    # print(entradas)
    targets = np.array(labels)

    mlp = MLP()

    epocas = 80000
    alfa = 0.6452  # varia de 0.1 à 1
    logger(f'Épocas: {epocas}, taxa de aprendizado: {alfa} ', 'parametros_iniciais.txt')
    mlp.treinamento(entradas, targets, epocas, alfa)

    nome_csv = 'caracteres-ruido.csv'
    data_input = pd.read_csv(nome_csv, header=None, usecols=[i for i in range(70)])
    X, labels = separa_colunas(data_input, linhas_treinamento, colunas)  # x numero de linhas  y num colunas
    entradas = np.array(X)
    targets = np.array(labels)
    # mlp.treinamento(entradas, targets, epocas, alfa)

    """------ EXECUCAO ------"""

    nome_csv = 'caracteres-ruido.csv'
    entrada_execucao = pd.read_csv(nome_csv, header=None, usecols=[i for i in range(70)])
    linhas_execucao = 1
    X_teste, labels_teste = separa_colunas(entrada_execucao, linhas_execucao, colunas)
    entradas = np.array(X_teste)

    resultados = mlp.predizer(entradas, nome_csv)  # aqui entrariam os caracteres sujos
    resultado_precisao = precisao(resultados, np.array(labels_teste))
    # matriz_confusao(resultados, labels_teste)  # Caso queira o log da matriz de confusão no resultado.txt
    matriz = matriz_confusao_multiclasse(resultados, labels_teste)
    estatisticas_matriz_confusao(matriz)
