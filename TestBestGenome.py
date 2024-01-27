# -*- coding: utf-8 -*-
__author__ = 'Chason'
from TestTictactoe import *

with open("tictactoe.gen", "rb") as file_in:
    model = pickle.load(file_in)
tt = TictactoeTest()
tt.test_case(model,test_time=10000)
# model.show_structure()
# tt.test_case(model,test_time=10, show_board=True)