# -*- coding: utf-8 -*-
__author__ = 'Chason'

from Environment import *
import sys
import argparse

class XorTest(object):
    best_fitness = 4
    input_size = 2
    output_size = 1
    @staticmethod
    def get_fitness(genome):
        fitness = 0
        for case in [[[0, 0], [0]], [[0, 1], [1]], [[1, 0], [1]], [[1, 1], [0]]]:
            for i in range(len(genome.input_nodes)):
                genome.input_nodes[i].value = case[0][i]
            genome.forward_propagation()
            if round(genome.output_nodes[0].value) != case[1][0]:
                break
            else:
                fitness += 1
        genome.fitness = fitness

def main(args=None):
    env = Environment(input_size=XorTest.input_size,
                      output_size=XorTest.output_size,
                      init_population=args.pop,
                      max_generation=args.gen,
                      comp_threshold=args.thr,
                      avg_comp_num=args.cmp,
                      mating_prob=args.mat,
                      copy_mutate_pro=args.cpy,
                      self_mutate_pro=args.slf,
                      excess=args.exc,
                      disjoint=args.dsj,
                      weight=args.wgh,
                      survive=args.srv,
                      task=XorTest)

    # env.test()
    env.run(task=XorTest, showResult=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change the evolutionary parameters.')
    parser.add_argument(
        '--pop',
        default=150,
        type=int,
        help='The initial population size.'
    )
    parser.add_argument(
        '--gen',
        default=100,
        type=int,
        help='The maximum generations.'
    )
    parser.add_argument(
        '--thr',
        default=1.5,
        type=float,
        help='The compatibility threshold.'
    )
    parser.add_argument(
        '--cmp',
        default=50,
        type=int,
        help='The number of genomes used to compare compatibility.'
    )
    parser.add_argument(
        '--mat',
        default=0.6,
        type=float,
        help='The mating probability.'
    )
    parser.add_argument(
        '--cpy',
        default=0.1,
        type=float,
        help='The copy mutation probability.'
    )
    parser.add_argument(
        '--slf',
        default=0.99,
        type=float,
        help='The self mutation probability.'
    )
    parser.add_argument(
        '--exc',
        default=0.9,
        type=float,
        help='The excess weight.'
    )
    parser.add_argument(
        '--dsj',
        default=0.1,
        type=float,
        help='The disjoint weight.'
    )
    parser.add_argument(
        '--wgh',
        default=0.001,
        type=float,
        help='The average weight differences weight.'
    )
    parser.add_argument(
        '--srv',
        default=15,
        type=int,
        help='The number of survivors per generation.'
    )
    args = parser.parse_args()
    sys.exit(main(args))