from abc import ABC, abstractmethod
from typing import Any

class AbstractNode(ABC):
    def __init__(self, value: Any, left, right):
        self.value = value
        self.left = left
        self.right = right

        self.father = None

        if self.left:
            self.left.father = self
        if self.right:
            self.right.father = self

    def eval(self, value=None):
        "Evaluates the value of this Node."
        return eval(self.toString(value))

    @abstractmethod
    def copy(self):
        "Returns a deepcopy of this Node."

    @property
    def nodes(self) -> int:
        return  1 + \
                (self.left.nodes if self.left else 0) + \
                (self.right.nodes if self.right else 0)
    
    @property
    def depth(self) -> int:
        return  1 + (self.father.depth if self.father else -1)

    def toString(self, value=None) -> str:
        "Returns a string of this Node."

        string = ""
        string += "( "
        string += self.left.toString(value)
        string += " "
        string += str(self.value)
        string += " "
        string += self.right.toString(value)
        string += " )"

        return string

    def show(self) -> None:
        "Prints on console this Node."
        print(self.toString())
    
    def nodesList(self, lista=None) -> list:
        if lista == None:
            lista = list()
            
        lista += [self]
        if self.left:
            self.left.nodesList(lista)
        if self.right:
            self.right.nodesList(lista)
        return lista

    def replace(self, index, tree):
        if index == 0:
            raise ValueError
        
        nodes_list = self.nodesList()
        self_tree = nodes_list[index]

        if (self_tree==self_tree.father.left):
            self_tree.father.left = tree
        else:
            self_tree.father.right = tree

        tree.father = self_tree.father
        
    def __repr__(self):
        return "<<Node " + self.toString() + ">>"


class Number(AbstractNode):
    def __init__(self, value: int):
        super().__init__(value, None, None)

    def toString(self, value=None) -> str:
        if float(self.value) < 0:
            return "(" + str(self.value) + ")"
        else:
            return str(self.value)

    def copy(self):
        return Number(self.value)

class X(Number):
    def __init__(self):
        super().__init__("X")
    
    def toString(self, value="X") -> str:
        if value == "X" or value == None:
            return "X"
        if value < 0:
            return "(" + value if value else str(value) + ")"
        else:
            return str(value)

class Add(AbstractNode):
    def __init__(self, left:AbstractNode, right:AbstractNode):
        super().__init__('+',left, right)

    def copy(self):
        return Add(
            self.left.copy(),
            self.right.copy()
        )

class Mult(AbstractNode):
    def __init__(self, left:AbstractNode, right:AbstractNode):
        super().__init__('*',left, right)
    
    def copy(self):
        return Mult(
            self.left.copy(),
            self.right.copy()
        )

class Div(AbstractNode):
    def __init__(self,left:AbstractNode, right:AbstractNode):
        super().__init__('/',left, right)
        
    def copy(self):
        return Div(
            self.left.copy(),
            self.right.copy()
        )

class Subs(AbstractNode):
    def __init__(self,left:AbstractNode, right:AbstractNode):
        super().__init__('-',left, right)
        
    def copy(self):
        return Subs(
            self.left.copy(),
            self.right.copy()
        )

if __name__ == "__main__":
    number_a = Number(44)
    number_b = Number(-4)
    number_c = Number(0)
    number_d = Number(40)
    node = Div(number_a, Add(number_b, Mult(number_d, number_c)))
    node.show()
    print(node.nodesList())