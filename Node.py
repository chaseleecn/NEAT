# -*- coding: utf-8 -*-
__author__ = 'Chason'

class Node(object):
    """The node of evolutionary neural network.
    Attributes:
        id: The unique identification number of the node.
        value: The value of the node, ranging from 0 to 1.
        tag: The tag of the node, such as input node, hidden node, and output node.
    """
    def __init__(self, id, tag = '', value = 0):
        self.id = id
        self.value = value
        self.tag = tag

    def __eq__(self, other):
        """If the node id and tag are the same, then the two nodes are considered the same."""
        return self.id == other.id and self.tag == other.tag