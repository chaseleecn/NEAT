# -*- coding: utf-8 -*-
__author__ = 'Chason'

from NEAT import NEAT
import random
import copy
import pickle
import os
import sys

class Environment(object):
    """This is an ecological environment that can control the propagation of evolutionary neural networks.
    Attributes:
        input_size: The input size of the genomes.
        output_size: The output size of the genomes.
        init_population: The initial population of genomes in the environment.
        max_generation: The maximum number of genomes generations
        genomes: The list of NEAT(NeuroEvolution of Augmenting Topologies)
    """
    def __init__(self,input_size, output_size, init_population, max_generation, comp_threshold, avg_comp_num,
                 mating_prob, copy_mutate_pro, self_mutate_pro,excess, disjoint, weight, survive, task,
                 file_name = None):
        self.input_size = input_size
        self.output_size = output_size
        self.population = init_population
        self.evaluation = init_population
        self.max_generation = max_generation
        self.next_generation = []
        self.outcomes = []
        self.generation_iter = 0
        self.species = [[NEAT(i, input_size, output_size) for i in range(init_population)]]
        self.comp_threshold = comp_threshold
        self.avg_comp_num = avg_comp_num
        self.mating_prob = mating_prob
        self.copy_mutate_pro = copy_mutate_pro
        self.self_mutate_pro = self_mutate_pro
        self.excess = excess
        self.disjoint = disjoint
        self.weight = weight
        self.survive = survive
        self.file_name = file_name
        self.adversarial_genomes = []

        # Load the environment parameters, if you saved it before.
        if self.file_name != None and os.path.exists(self.file_name + '.env'):
            print "Loading environment parameters...",
            self.load()
            print "\tDone!"

        for sp in self.species:
            for gen in sp:
                task.get_fitness(gen)

    def update_adversarial_genomes(self, task):
        self.adversarial_genomes = []
        for sp in self.species:
            self.adversarial_genomes.extend(sp[:])
        self.adversarial_genomes.sort(key=lambda NEAT: NEAT.fitness, reverse=True)
        self.adversarial_genomes = self.adversarial_genomes[0:(task.play_times/5)]

    def save(self):
        if self.file_name != None:
            print "Saving...",
            with open(self.file_name + '.env', "wb") as f:
                pickle.dump([ self.generation_iter,
                              self.species,
                              self.next_generation,
                              self.outcomes,
                              self.adversarial_genomes,
                              NEAT.connection_list], f)
            print "\tDone!"

    def load(self):
        if self.file_name != None:
            with open(self.file_name + '.env', "rb") as f:
                self.generation_iter, self.species, self.next_generation, self.outcomes, self.adversarial_genomes, NEAT.connection_list = pickle.load(f)

    def produce_offspring(self, genome):
        """Produce a new offspring."""
        offspring = copy.deepcopy(genome)
        offspring.id = self.evaluation
        self.next_generation.append(offspring)
        self.evaluation += 1
        return offspring

    def add_outcome(self, genome):
        """Collecting outcomes."""
        gen = copy.deepcopy(genome)
        self.outcomes.append(gen)
        # print "Generation:%d\tFound outcome %d,\thidden node = %d,\tconnections = %d"%(self.generation_iter,
        #                                                                                len(self.outcomes),
        #                                                                                len(gen.hidden_nodes),
        #                                                                                gen.connection_count())

    def mating_pair(self, pair, task):
        """Mating two genomes."""
        p1 = pair[0]
        p2 = pair[1]
        p1_len = len(p1.connections)
        p2_len = len(p2.connections)
        offspring = self.produce_offspring(NEAT(self.evaluation, self.input_size, self.output_size, offspring=True))

        # Generate the same number of nodes as the larger genome
        max_hidden_node = max(len(p1.hidden_nodes), len(p2.hidden_nodes))
        for i in range(max_hidden_node):
            offspring.add_hidden_node()

        # Crossing over
        i, j = 0, 0
        while i < p1_len or j < p2_len:
            if i < p1_len and j < p2_len:
                if p1.connections[i].innovation == p2.connections[j].innovation:
                    if NEAT.probability(0.5):
                        con = p1.connections[i]
                    else:
                        con = p2.connections[j]
                    i += 1
                    j += 1
                elif p1.connections[i].innovation < p2.connections[j].innovation:
                    con = p1.connections[i]
                    i += 1
                else:
                    con = p2.connections[j]
                    j += 1
            elif i >= p1_len:
                con = p2.connections[j]
                j += 1
            else:
                con = p1.connections[i]
                i += 1
            offspring.add_connection_id(input_node_id=con.input.id,
                                        output_node_id=con.output.id,
                                        weight=con.weight,
                                        enable=con.enable)
        # task.get_fitness(offspring)
        task.get_adversarial_fitness(offspring, self.adversarial_genomes)
        return offspring

    def mating_genomes(self, task):
        """Randomly mating two genomes."""
        mating_pool = []
        for sp in self.species:
            for gen in sp:
                # The higher the fitness, the higher the probability of mating.
                if NEAT.probability(self.mating_prob):
                    mating_pool.append(gen)

            while len(mating_pool) > 1:
                pair = random.sample(mating_pool, 2)
                self.mating_pair(pair, task)
                for p in pair:
                    mating_pool.remove(p)

    def mutation(self, task):
        """Genome mutation."""
        for k, sp in enumerate(self.species):
            for gen in self.species[k]:
                if gen.fitness < task.best_fitness:
                    if NEAT.probability(self.copy_mutate_pro):
                        offspring = self.produce_offspring(gen)
                        offspring.mutation()
                        # task.get_fitness(offspring)
                        task.get_adversarial_fitness(offspring, self.adversarial_genomes)
                    if NEAT.probability(self.self_mutate_pro):
                        gen.mutation(new_node=False)
                # task.get_fitness(gen)
                task.get_adversarial_fitness(gen, self.adversarial_genomes)

    def compatibility(self, gen1, gen2):
        """Calculating compatibility between two genomes."""
        g1_len = len(gen1.connections)
        g2_len = len(gen2.connections)
        E, D = 0, 0
        i, j = 0, 0
        w1, w2 = 0, 0
        while i < g1_len or j < g2_len:
            if i < g1_len and j < g2_len:
                if gen1.connections[i].innovation == gen2.connections[j].innovation:
                    w1 += abs(gen1.connections[i].weight)
                    w2 += abs(gen2.connections[j].weight)
                    i += 1
                    j += 1
                elif gen1.connections[i].innovation < gen2.connections[j].innovation:
                    D += 1
                    w1 += abs(gen1.connections[i].weight)
                    i += 1
                else:
                    D += 1
                    w2 += abs(gen2.connections[j].weight)
                    j += 1
            elif i >= g1_len:
                E += 1
                w2 += abs(gen2.connections[j].weight)
                j += 1
            else:
                E += 1
                w2 += abs(gen1.connections[i].weight)
                i += 1
        W = abs(w1 - w2)
        distance = self.excess * E + self.disjoint * D + self.weight * W
        return distance

    def speciation(self, genome):
        """Assign a genome to compatible species."""
        for sp in self.species:
            avg_comp = 0
            for gen in sp[:self.avg_comp_num]:
                avg_comp += self.compatibility(gen, genome)
            avg_comp /= min(self.avg_comp_num, len(sp))
            if avg_comp < self.comp_threshold:
                sp.append(genome)
                return
        # If there is no compatible species, create a new species for the genome.
        if len(self.species) < 15:
            self.species.append([genome])

    def surviving_rule(self):
        """Set the surviving rules."""

        for gen in self.next_generation:
            self.speciation(gen)

        for k, sp in enumerate(self.species):
            sp.sort(key=lambda NEAT: NEAT.fitness, reverse=True)
            # sp = sp[:20] + self.next_generation + [NEAT(i,
            #                                             self.input_size,
            #                                             self.output_size)
            #                                        for i in range(10)]
            self.species[k] = self.species[k][:self.survive]

    def run(self, task, showResult=False):
        """Run the environment."""
        print "Running Environment...(Initial population = %d, Maximum generation = %d)"%(self.population, self.max_generation)
        # Generational change
        for self.generation_iter in range(self.generation_iter + 1, self.max_generation):
            self.next_generation = []

            # Mutation
            self.mutation(task)

            # Mating genomes
            self.mating_genomes(task)

            # Killing bad genomes
            self.surviving_rule()

            # Logging outcome information
            outcome = [gen for sp in self.species for gen in sp if gen.fitness >= task.best_fitness]
            self.population = sum([len(sp) for sp in self.species])
            hidden_distribution = [0]
            max_fitness = -sys.maxint
            best_outcome = None
            for sp in self.species:
                genome_len = len(sp)
                if genome_len > 0:
                    for gen in sp:
                        hid = len(gen.hidden_nodes)
                        while hid >= len(hidden_distribution):
                            hidden_distribution.append(0)
                        hidden_distribution[hid] += 1
                        if gen.fitness > max_fitness:
                            max_fitness = gen.fitness
                            best_outcome = gen

            print "Generation %d:\tpopulation = %d,\tspecies = %d,\toutcome = %d,\tbest_fitness(%d_%d) = %.2f%%,\thidden node distribution:%s"%(
                self.generation_iter,
                self.population,
                len(self.species),
                len(outcome),
                best_outcome.id,
                len(best_outcome.hidden_nodes),
                100.0 * max_fitness / task.best_fitness,
                hidden_distribution)

            # Update adversarial genomes
            # if self.generation_iter >= 100 and self.generation_iter % 10 == 0:
            #     self.update_adversarial_genomes(task)
            #     print "Adversarial genomes updated."

            # Save genome
            if self.file_name != None:
                with open(self.file_name + '.gen', 'wb') as file_out:
                    pickle.dump(best_outcome, file_out)

            # Save environment parameters
            if self.generation_iter % 10 == 0:
                self.save()

        # Collecting outcomes
        max_fitness = -sys.maxint
        best_outcome = None
        for sp in self.species:
            for gen in sp:
                if gen.fitness >= task.best_fitness:
                    self.add_outcome(gen)
                if gen.fitness > max_fitness:
                    max_fitness = gen.fitness
                    best_outcome = gen
        # if len(self.outcomes) == 0:
        #     self.add_outcome(best_outcome)

        # Save best genome
        if self.file_name != None:
            with open(self.file_name + '.gen', 'wb') as file_out:
                pickle.dump(best_outcome, file_out)

        print "Species distribution:"
        for k, sp in enumerate(self.species):
            hidden_node = []
            con = []
            for gen in sp:
                hidden_node.append(len(gen.hidden_nodes))
                con.append(len(gen.connections))
            print "\t%d:\tnode:\t%s\n\t\tcons:\t%s"%(k, hidden_node, con)
        print

        if showResult:
            print "Completed Genomes:",
            self.outcomes.sort(key=lambda NEAT:NEAT.hidden_nodes)
            outcomes_len = len(self.outcomes)
            if outcomes_len > 0:
                print outcomes_len
            else:
                print "There are no completed genomes!"
            avg_hid = 0.0
            avg_con = 0.0
            if outcomes_len > 0:
                for gen in self.outcomes:
                    gen.show_structure()
                    avg_hid += len(gen.hidden_nodes)
                    avg_con += gen.connection_count()
                avg_hid /= outcomes_len
                avg_con /= outcomes_len
            print "Evaluation: %d,\tPopulation: %d,\tAverage Hidden node = %f,\tAverage Connection = %f"%(
                self.evaluation, self.population, avg_hid, avg_con)
