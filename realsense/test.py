import torch
import numpy as np
from math import pi



class Test:
     def __init__(self):
          self.x = torch.tensor([1, 2, 1])
          self.y = torch.tensor([0, pi, 2])

     def printxy(self):
          print(self.x)
          print(self.y)

if __name__ == "__main__":
     test = Test()
     test.printxy()

