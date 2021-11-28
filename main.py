import numpy as np
import matplotlib.pyplot as mpl
import ga
# Equação escolhida
# Y = w1x1 + w2x2 + w3x3 + w4x4 + w5x5 + w6x6
# (x1,x2,x3,x4,x5,x6) = (4,-2,3.5,5,-11,-4.7)
# A equação possui 6 inputs e 6 pesos

# Entradas da equação
entradas_eq = [4,-2,3.5,5,-11,-4.7] # [4,-2,3.5,5,-11]

# Número de pesos
pesosQtd = len(entradas_eq) # Nesse caso 5 inputs

# Solução por pop
solPop = 8

num_parents_mating = 4

# Definindo o tamanho da população, pop terá solPop cromossomo que tera gene pesosQtd
popTam = (solPop, pesosQtd)

# Criando a pop inicial de forma aleatória
novaPop = np.random.uniform(low=-4.0, high=4.0, size=popTam)
print(novaPop)

melhoresSaidas = []
numGeracoes = 5

for generation in range(numGeracoes):
    print("Geração : ", generation)

    # Medir a fitness de cada cromossomo na população
    fitness = ga.cal_pop_fitness(entradas_eq, novaPop)
    print("Fitness")
    print(fitness)

    melhoresSaidas.append(np.max(np.sum(novaPop*entradas_eq, axis=1)))
    # O melhor resultado na iteração atual
    print("Melhor Resultado : ", np.max(np.sum(novaPop*entradas_eq, axis=1)))

    # Seleção dos melhores parentes da população para o acasalamento
    parents = ga.select_mating_pool(novaPop, fitness, num_parents_mating)
    print("Parents")
    print(parents)

    # Gerando próxima geração usando crossover
    offspring_crossover = ga.crossover(parents, offspring_size=(popTam[0]-parents.shape[0], pesosQtd))
    print("Crossover")
    print(offspring_crossover)

    # Adicionando algumas variações ao offspring usando mutação
    offspring_mutation = ga.mutation(offspring_crossover, num_mutations=2)
    print("Mutação")
    print(offspring_mutation)

    # Criar a nova população com base nos parentes e offspring
    novaPop[0:parents.shape[0], :] = parents
    novaPop[parents.shape[0]:, :] = offspring_mutation

# Obtendo a melhor solução após a iteração finalizando todas as gerações.
# Primeiro, a primeira fitness é calculada para cada solução na geração final
fitness = ga.cal_pop_fitness(entradas_eq, novaPop)

# O retorno do index da solução correspondendo ao melhor fitness
melhorIdx = np.where(fitness == np.max(fitness))

print("Best solution : ", novaPop[melhorIdx, :])
print("Best solution fitness : ", fitness[melhorIdx])

mpl.plot(melhoresSaidas)
mpl.xlabel("Iteration")
mpl.ylabel("Fitness")
mpl.show()