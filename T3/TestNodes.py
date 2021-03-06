import unittest

from Nodes import *

class TestNumberNode(unittest.TestCase):
    def test_number(self):
        number_a = Number(-4)
        number_b = Number(42)

        self.assertEqual(number_a.value, -4)
        self.assertEqual(number_b.value, 42)

        self.assertEqual(number_a.eval(), -4)
        self.assertEqual(number_b.eval(), 42)

class TestSimpleOperationNodes(unittest.TestCase):
    def setUp(self):
        self.number_a = Number(42)
        self.number_b = Number(-4)

    def test_add(self):
        add = Add(self.number_a, self.number_b)

        self.assertEqual(add.left, self.number_a)
        self.assertEqual(add.right, self.number_b)
        self.assertEqual(add.eval(), 42 + (-4))
    
    def test_mult(self):
        mult = Mult(self.number_a, self.number_b)

        self.assertEqual(mult.left, self.number_a)
        self.assertEqual(mult.right, self.number_b)
        self.assertEqual(mult.eval(), 42 * (-4))
    
    def test_div(self):
        div = Div(self.number_a, self.number_b)

        self.assertEqual(div.left, self.number_a)
        self.assertEqual(div.right, self.number_b)
        self.assertEqual(div.eval(), 42 / (-4))

    def test_subs(self):
        subs = Subs(self.number_a, self.number_b)

        self.assertEqual(subs.left, self.number_a)
        self.assertEqual(subs.right, self.number_b)
        self.assertEqual(subs.eval(), 42 - (-4))

class TestComplexOperationNodes(unittest.TestCase):
    def setUp(self):
        self.number_a = Number(44)
        self.number_b = Number(-4)
        self.number_c = Number(0)
        self.number_d = Number(40)
    
    def test_largerOperation(self):
        node = \
        Div(
            self.number_a,
            Add(
                self.number_b,
                Mult(
                    self.number_d, 
                    self.number_c
                )
            )
        )

        self.assertEqual( node.eval(), 44 / (-4 + (40 * 0)) )

class TestToString(unittest.TestCase):
    def setUp(self):
        self.number_a = Number(42)
        self.number_b = Number(-4)
    
    def test_numberToString(self):
        self.assertEqual(self.number_a.toString(), "42")
        self.assertEqual(self.number_b.toString(), "(-4)")
    
    def test_addToString(self):
        add = Add(self.number_a, self.number_b)
        self.assertEqual(add.toString(), "( 42 + (-4) )")
        
    def test_multToString(self):
        mult = Mult(self.number_a,self.number_b)
        self.assertEqual(mult.toString(), "( 42 * (-4) )")
    
    def test_divToString(self):
        div = Div(self.number_a,self.number_b)
        self.assertEqual(div.toString(), "( 42 / (-4) )")

class TestCopy(unittest.TestCase):
    def setUp(self):
        self.number_a = Number(42)
        self.number_b = Number(-4)
        self.number_c = Number(5)
        self.number_d = Number(2)
        self.tree=Add(self.number_b,Mult(self.number_c,self.number_d))

    def own_test_childsCopy(self,node1,node2):
        if node1 is None and node2 is None:
            return
        self.assertEqual(node1.value,node2.value)
        self.own_test_childsCopy(node1.left,node2.left)
        self.own_test_childsCopy(node1.right,node2.right)
        
    def test_copyNumber(self):
        n=Number(self.number_a)
        n2=n.copy()
        self.assertNotEqual(n,n2)
        self.own_test_childsCopy(n,n2)

    def test_copyAdd(self):
        add=Add(self.number_a,self.number_b)
        add2=add.copy()
        self.assertNotEqual(add,add2)
        self.own_test_childsCopy(add,add2)
    
    def test_copyMult(self):
        mult=Mult(self.number_a,self.number_b)
        mult2=mult.copy()
        self.assertNotEqual(mult,mult2)
        self.own_test_childsCopy(mult,mult2)

    def test_copyDiv(self):
        div=Div(self.number_a,self.number_b)
        div2=div.copy()
        self.assertNotEqual(div,div2)
        self.own_test_childsCopy(div,div2)

    def test_copyTree(self):
        tree2=self.tree.copy()
        self.assertNotEqual(self.tree,tree2)
        self.own_test_childsCopy(self.tree,tree2)


class TestProperties(unittest.TestCase):
    def setUp(self):
        self.number_a = Number(44)
        self.number_b = Number(-4)
        self.number_c = Number(0)
        self.number_d = Number(40)
        self.node = Div(self.number_a, Add(self.number_b, Mult(self.number_d, self.number_c)))

        self.number_x = Number(44)
        self.number_z = Number(-4)
        self.node2 = Add(self.number_x, self.number_z)

    def test_nodes(self):
        self.assertEqual(self.number_a.nodes, 1)
        self.assertEqual(Add(self.number_a, self.number_b).nodes, 3)
        self.assertEqual(self.node.nodes, 7)

    def test_nodesList(self):
        self.assertEqual(len(self.node.nodesList()), 7)

    def test_depth(self):
        self.assertEqual(self.number_a.depth, 1)
        self.assertEqual(self.node.depth, 0)
        self.assertEqual(self.node.right.right.depth, 2)
        self.assertEqual(self.number_c.depth, 3)

        self.assertEqual(self.node2, self.number_x.father)
        self.assertEqual(self.node2, self.number_z.father)
        self.assertEqual(self.node2.left.depth, 1)
        

class TestReplace(unittest.TestCase):
    def setUp(self):
        self.number_a = Number(44)
        self.number_b = Number(-4)
        self.number_c = Number(0)
        self.number_d = Number(40)
        self.node1 = Add(self.number_a, self.number_b)
        self.node = Div(self.number_a, Add(self.number_b, Mult(self.number_d, self.number_c)))
    
    def test_big_replace(self):
        self.assertEqual(len(self.node.nodesList()), 7)
        original=self.node
        self.node.replace(2,self.node1)
        
        self.assertEqual(self.node.value, original.value)
        self.assertEqual(self.node.left, original.left)
        self.assertEqual(self.node.right.value, self.node1.value)
        self.assertEqual(self.node.right.left, self.node1.left)
        self.assertEqual(self.node.right.right, self.node1.right)

        self.assertEqual(len(self.node.nodesList()), 5)

if __name__ == "__main__":
    unittest.main()