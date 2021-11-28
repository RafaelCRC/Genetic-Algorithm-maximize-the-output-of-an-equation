# Modulo GA

import numpy

def cal_pop_fitness(equation_inputs, pop):
    # Calculando o valor da fitness para cada solução na população atual
    # A função da fitness calcula a soma dos produtos entre cada entrada e seu peso correspondente
    fitness = numpy.sum(pop*equation_inputs, axis=1)
    return fitness

def select_mating_pool(pop, fitness, num_parents):
    # Selecionando os melhores indivíduos na geração atual como parentes para produzir o offspring da próxima geração
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # O ponto em que qual crossover toma lugar entre dois parentes. Geralmente, é no centro
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        # Index do primeiro parente a acasalar
        parent1_idx = k%parents.shape[0]
        # Index do segundo parente a acasalar
        parent2_idx = (k+1)%parents.shape[0]
        # O novo offspring vai ter a primeira metade dos genes tomada do primeiro parente
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # O novo offspring vai ter a segunda metade dos genes tomada do primeiro parente
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

def mutation(offspring_crossover, num_mutations=1):
    mutations_counter = numpy.uint8(offspring_crossover.shape[1] / num_mutations)
    # Mutação troca o número de genes definidos pelo argumento num_mutations. As trocas são aleatórias.
    for idx in range(offspring_crossover.shape[0]):
        gene_idx = mutations_counter - 1
        for mutation_num in range(num_mutations):
            # O valor aleatório vai ser adicionado no gene
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            offspring_crossover[idx, gene_idx] = offspring_crossover[idx, gene_idx] + random_value
            gene_idx = gene_idx + mutations_counter
    return offspring_crossover