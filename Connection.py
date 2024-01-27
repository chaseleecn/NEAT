# -*- coding: utf-8 -*-
__author__ = 'Chason'
import random

class Connection(object):
    """The connection of evolutionary neural network.
    Attributes:
        input: The input node of the connection.
        output: The output node of the connection.
        weight: The weight of the connection.
        enable: The enable flag for the connection
        innovation: The number of innovation for the connection.
    """
    def __init__(self, input, output, weight = None, innovation = None, enable = True):
        self.input = input
        self.output = output
        if weight != None:
            self.weight = weight
        else:
            self.weight = self.random_weight()
        self.enable = enable
        self.innovation = innovation

    def __eq__(self, other):
        """If the input node and the output node are the same in both connections,
        then the two connections are considered the same.
        """
        return self.input == other.input and self.output == other.output

    def random_weight(self, range = 10):
        """Randomly generate the weight of the connection. The default range is from -10 to 10."""
        self.weight = random.uniform(-range, range)