# -*- coding: utf-8 -*-
__author__ = 'Chason'

from Environment import *
import sys
import argparse
import time

class TictactoeTest:
    win_reward = 10
    loss_penalty = 100
    draw_reward = 1
    play_times = 50
    best_fitness = play_times * 2 * win_reward

    ROW = 3
    COL = 3
    WIN_NUM = 3
    DRAW = 3

    PLAYER1 = 1
    PLAYER2 = -1

    PLAYER1_CHAR = '#'
    PLAYER2_CHAR = '*'
    MAPS = '.'

    input_size = ROW * COL
    output_size = ROW * COL
    board = [[0 for c in range(COL)] for r in range(ROW)]
    empty = [[r, c] for r in range(ROW) for c in range(COL)]
    turns = 0

    def init_board(self):
        self.board = [[0 for c in range(self.COL)] for r in range(self.ROW)]
        self.empty = [[r, c] for r in range(self.ROW) for c in range(self.COL)]

    def print_piece(self, inx):
        if inx == self.PLAYER1:
            print self.PLAYER1_CHAR,
        elif inx == self.PLAYER2:
            print self.PLAYER2_CHAR,
        else:
            print self.MAPS,

    def show_board(self):
        print "----------------------------------"
        for r in self.board:
            for c in r:
                self.print_piece(c)
            print
        # print

    def is_occupied(self, r, c):
        return self.board[r][c] != 0

    def move(self, player, r, c):
        if not self.is_occupied(r, c):
            self.empty.remove([r, c])
            self.board[r][c] = player
            return True
        return False

    def rnd_move(self, player):
        if len(self.empty) > 0:
            seed = random.random()
            random.seed(seed*time.time())
            p = random.choice(self.empty)
            self.empty.remove(p)
            self.board[p[0]][p[1]] = player
            return p
        return None

    def judge(self, r, c):
        player = self.board[r][c]
        for ddr, ddc in [[-1, -1], [-1, 0], [-1, 1], [0, 1]]:
            count = 1
            for dr, dc in [[ddr, ddc], [-ddr, -ddc]]:
                nr = r + dr
                nc = c + dc
                while nr >= 0 and nr < self.ROW and nc >= 0 and nc < self.COL:
                    if self.board[nr][nc] == player:
                        count += 1
                        nr += dr
                        nc += dc
                    else:
                        break
            if count >= self.WIN_NUM:
                return player
        if self.turns + 1 >= self.ROW * self.COL:
            return self.DRAW
        return None

    def test_case(self, genome, test_time=500, show_board=False):
        print "Test case:"
        wins = 0
        loses = 0
        draw = 0
        foul = 0
        fitness = 0
        for k in range(2):
            for i in range(test_time):
                self.init_board()
                for self.turns in range(self.ROW * self.COL):
                    if self.turns % 2 == k:
                        # input board data
                        for m in range(self.ROW):
                            for n in range(self.COL):
                                genome.input_nodes[m * self.COL + n].value = self.board[m][n]

                        # calculate output location
                        genome.forward_propagation()
                        # output = genome.get_max_output_index()
                        # r, c = int(output / self.COL), output % self.COL
                        r, c = genome.get_legal_output(self.board, self.COL)
                        self.move(self.PLAYER1, r, c)
                        # if not self.move(self.PLAYER1, r, c):
                        #     fitness -= 10
                        #     print "(%d, %d) has been occupied."%(r, c)
                        #     foul += 1
                        #     break
                    else:
                        r, c = self.rnd_move(self.PLAYER2)
                    if show_board:
                        self.show_board()
                    res = self.judge(r, c)
                    if res != None and res != self.DRAW:
                        print "Player %d wins."%res
                        if res == self.PLAYER1:
                            fitness += self.win_reward
                            wins += 1
                        else:
                            loses += 1
                            fitness -= self.loss_penalty
                        break
                    elif res == self.DRAW:
                        print "There is a draw."
                        fitness += self.draw_reward
                        draw += 1
                        break
        print "Test Times: %d\n\tWins: %d\t(%.2f%%)\n\tLoses: %d\t(%.2f%%)\n\tDraws: %d\t(%.2f%%)\n\tFoul = %d\t(%.2f%%)"%(
            test_time*2, wins, 100.0*wins/test_time/2, loses, 100.0*loses/test_time/2,
            draw, 100.0*draw/test_time/2, foul, 100.0*foul/test_time/2)
        genome.show_structure(info_only=True)

    def get_adversarial_fitness(self, genome, adversarial):
        fitness = 0
        for k in range(2):
            for i in range(self.play_times):
                self.init_board()
                for self.turns in range(self.ROW * self.COL):
                    if self.turns % 2 == k:
                        # input board data
                        for m in range(self.ROW):
                            for n in range(self.COL):
                                genome.input_nodes[m * self.COL + n].value = self.board[m][n]

                        # calculate output location
                        genome.forward_propagation()
                        # output = genome.get_max_output_index()
                        # r, c = int(output / self.COL), output % self.COL
                        r, c = genome.get_legal_output(self.board, self.COL)
                        self.move(self.PLAYER1, r, c)
                        # if not self.move(self.PLAYER1, r, c):
                        #     # print "AI randomly move:"
                        #     # r, c = self.rnd_move(self.PLAYER1)
                        #     fitness -= 10
                        #     break
                    else:
                        if i < len(adversarial) and k == 1:
                            for m in range(self.ROW):
                                for n in range(self.COL):
                                    adversarial[i].input_nodes[m * self.COL + n].value = -self.board[m][n]
                            adversarial[i].forward_propagation()
                            r, c = adversarial[i].get_legal_output(self.board, self.COL)
                            self.move(self.PLAYER2, r, c)
                        else:
                            r, c = self.rnd_move(self.PLAYER2)

                    # self.show_board()
                    res = self.judge(r, c)
                    if res != None and res != self.DRAW:
                        # print "Player %d wins."%res
                        if res == self.PLAYER1:
                            fitness += self.win_reward
                        else:
                            fitness -= self.loss_penalty
                        break
                    elif res == self.DRAW:
                        # print "There is a draw."
                        fitness += self.draw_reward
                        break
        genome.fitness = fitness
        return fitness

    def get_fitness(self, genome):
        fitness = 0
        for k in range(2):
            for i in range(self.play_times):
                self.init_board()
                for self.turns in range(self.ROW * self.COL):
                    if self.turns % 2 == k :
                        # input board data
                        for m in range(self.ROW):
                            for n in range(self.COL):
                                genome.input_nodes[m*self.COL+n].value = self.board[m][n]

                        # calculate output location
                        genome.forward_propagation()
                        # output = genome.get_max_output_index()
                        # r, c = int(output / self.COL), output % self.COL
                        r, c = genome.get_legal_output(self.board, self.COL)
                        self.move(self.PLAYER1, r, c)
                        # if not self.move(self.PLAYER1, r, c):
                        #     # print "AI randomly move:"
                        #     # r, c = self.rnd_move(self.PLAYER1)
                        #     fitness -= 10
                        #     break
                    else:
                        r, c = self.rnd_move(self.PLAYER2)
                    # self.show_board()
                    res = self.judge(r, c)
                    if res != None and res != self.DRAW:
                        # print "Player %d wins."%res
                        if res == self.PLAYER1:
                            fitness += self.win_reward
                        else:
                            fitness -= self.loss_penalty
                        break
                    elif res == self.DRAW:
                        # print "There is a draw."
                        fitness += self.draw_reward
                        break
        genome.fitness = fitness
        return fitness

def main(args=None):
    env = Environment(input_size=TictactoeTest.input_size,
                      output_size=TictactoeTest.output_size,
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
                      task=TictactoeTest(),
                      file_name='tictactoe')

    env.run(task=TictactoeTest(), showResult=True)
    TictactoeTest().test_case(env.outcomes[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Change the evolutionary parameters.')
    parser.add_argument(
        '--pop',
        default=20,
        type=int,
        help='The initial population size.'
    )
    parser.add_argument(
        '--gen',
        default=100000,
        type=int,
        help='The maximum generations.'
    )
    parser.add_argument(
        '--thr',
        default=2.0,
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
        default=0.5,
        type=float,
        help='The mating probability.'
    )
    parser.add_argument(
        '--cpy',
        default=0.4,
        type=float,
        help='The copy mutation probability.'
    )
    parser.add_argument(
        '--slf',
        default=0.0,
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
        default=10,
        type=int,
        help='The number of survivors per generation.'
    )
    args = parser.parse_args()
    sys.exit(main(args))
