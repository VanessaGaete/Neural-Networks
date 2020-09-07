from numpy import array

class Perceptron(object):
    def __init__(self, bias, w1, w2):
        self.bias=bias
        self.w1=w1
        self.w2=w2

    def output(self, x1,x2):
        m1=x1*self.w1
        m2=x2*self.w2
        result= m1+m2 +self.bias

        if result <= 0:
          return 0
        else:
          return 1

class PerceptronAND(Perceptron):

    def __init__(self):
        Perceptron.__init__(self,-1.5,1,1)

    #weights are always 1, it's no necessary to make a vector of ones. 
    #def output(self, inputs):
     #   result=0
      #  for i in range(0, len(inputs)):
      #      actual = inputs[i]
       #     result = result + actual
        
        #return result + self.bias


class PerceptronOR(Perceptron):
    
    def __init__(self, bias=-0.5):
        Perceptron.__init__(self,-0.5,1,1)



class PerceptronNAND(Perceptron):
    
    def __init__(self, bias=3):
        Perceptron.__init__(self,3,-2,-2)

    #def output(self, inputs):
    #    mult=inputs*self.weight
     #   result=0
     #   for i in range(0, len(mult)):
      #      actual = mult[i]
       #     result = result + actual
        
        #return result + self.bias


#quizas sea mejor que  no sean arreglos y sean dos outputs
class SummingGate:

    def __init__(self, x1, x2):
        self.x1=x1
        self.x2=x2


    def output(self):
        NAND=PerceptronNAND()
        n1=NAND.output(self.x1, self.x2)
        n2=NAND.output(self.x1, n1)
        n3=NAND.output(n1,self.x2)
        carry=NAND.output(n1,n1)
        result=NAND.output(n2,n3)

        return result,carry

SummingGate(1,1).output()