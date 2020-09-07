from perceptron import *
import unittest

#tests
class TestPerceptron(unittest.TestCase):
  def test_and(self):
    self.assertAlmostEqual(PerceptronAND().output(0,0), 0)
    self.assertAlmostEqual(PerceptronAND().output(0,1), 0)
    self.assertAlmostEqual(PerceptronAND().output(1,0), 0)
    self.assertAlmostEqual(PerceptronAND().output(1,1), 1)

  def test_or(self):
    self.assertAlmostEqual(PerceptronOR().output(0,0), 0)
    self.assertAlmostEqual(PerceptronOR().output(0,1), 1)
    self.assertAlmostEqual(PerceptronOR().output(1,0), 1)
    self.assertAlmostEqual(PerceptronOR().output(1,1), 1)

  def test_NAND(self):
    self.assertAlmostEqual(PerceptronOR().output(0,0), 1)
    self.assertAlmostEqual(PerceptronOR().output(0,1), 1)
    self.assertAlmostEqual(PerceptronOR().output(1,0), 1)
    self.assertAlmostEqual(PerceptronOR().output(1,1), 0)

  def test_NAND(self):
    self.assertAlmostEqual(SummingGate(1,1).output(), 0,1)
    self.assertAlmostEqual(SummingGate(0,0).output(), 0,0)
    self.assertAlmostEqual(SummingGate(0,1).output(), 1,0)
    self.assertAlmostEqual(SummingGate(1,0).output(), 1,0)
    
if __name__ == "__main__":
   unittest.main()
