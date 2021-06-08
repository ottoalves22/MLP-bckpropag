import numpy as np


class Modelo:
    def __init__(self, entradas, num_saida, saidas):
        self.entrada = entradas  # matriz com dados
        self.saida = saidas  # vetor numpy com valores de output
        self.num_saidas = num_saida  # numero de neuronios de output
        self.vetor_pesos = []  # Creating all the wij weights matrices
        self.vetor_escondidos = []  # Creating model's hidden vectors
        self.vetor_bias = []  # bias para camadas escondidas

        self.output_final = np.zeros(self.num_saidas)  # vetor de output inicial (predicao)
        self.output_indicador = True  # para inicializar pesos e bias de output
        self.momentum_indicator = True  # inicializa termo de momentum
        self.dropout_indicador = []  # vetor de booleans para indicar dropout

    def adicionar_camada_escondida(self, num_neurons, dropout=0.):
        """adiciona camadas escondidas ao modelo criando novas matrizes de pesos,
        os pesos serao inicializados aleatoriamente"""

        if not self.vetor_pesos:  # primeira camada escondida
            pesos = np.random.rand(num_neurons, self.entrada.shape[1]) \
                    * np.sqrt(2. / self.entrada.shape[1] + num_neurons)  # Ajuste na variancia dos pesos

        else:  # demais camadas escondidas
            pesos = np.random.rand(num_neurons, self.vetor_pesos[-1].shape[0]) \
                    * np.sqrt(2. / self.vetor_pesos[-1].shape[0] + num_neurons)

        vetor_escondido = np.random.rand(num_neurons)  # ativacoes aleatorias criando a camada escondida
        vetor_bias = np.ones(num_neurons, 1)  # cria vetores de bias

        self.vetor_escondidos.append(vetor_escondido)
        self.vetor_bias.append(vetor_bias)
        self.vetor_pesos.append(pesos)

        '''
        # inicializa dropout
        if dropout == 0.:
            self.dropout_indicador.append((False, 0))
        else:
            self.dropout_indicador.append((True, dropout))
        '''
        return self

