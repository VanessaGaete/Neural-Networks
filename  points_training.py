import numpy as np
from random import uniform


class Perceptron(object):

    def __init__(self,n,bias,lr=0.01):
        self.bias=bias
        self.weights = np.ones(n)
        self.lr= lr
        for i in range(len(self.weights)):
            self.weights[i] = uniform(-1,1)
            while self.weights[i] == 0:
                self.weights[i] = uniform(-1,1)
        

    def output(self, inputs):
        m=inputs * self.weights
        result=0
        for i in range(len(self.weights)):
            result+=m[i]

        if result <= 0:
            return 0
        else:
            return 1

    def training(self, inputs, desired):
        realOutput=self.output(inputs)
        error=desired - realOutput
        for i in range(len(self.weights)):
            self.weights[i]= self.weights[i] + (self.lr * inputs[i] * error)

        self.bias= self.bias + (self.lr * error)

class Example(object):
    
    def __init__(self, x, y):
        self.inputs =np.array([x,y])
        self.output=-1

    def f(self):
      return 3*self.inputs[0]+1

    def above(self):
        if self.inputs[1]>self.f():
            return True
        else: 
            return False

    def setup(self):
        if self.above():
            self.output= 1
        else:
            self.output=0


def exGenerator(n, width, height):
    examples=[]
    for i in range(n):
        ex=Example(uniform(-width,width), uniform(-height,height))
        ex.setup()
        examples.append(ex)

    return examples

def run(n, bias, width, height):
    examples= exGenerator(n,width,height)
    perceptron= Perceptron(2,bias)
    for ex in examples:
        perceptron.training(ex.inputs,ex.output)


    for ex in examples:
        guess=perceptron.output(ex.inputs)
        print("El punto: x=", ex.inputs[0],"y=", ex.inputs[1], "tiene resultado=", guess, "esperado=", ex.output )



run(100, 1, 25,25)