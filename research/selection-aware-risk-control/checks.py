import unittest
import numpy as np
from study import upper_binomial,policy_losses,calibrate,synthetic
class RiskTests(unittest.TestCase):
    def test_zero(self):self.assertAlmostEqual(upper_binomial(0,100,.05),1-.05**.01)
    def test_all(self):self.assertEqual(upper_binomial(10,10,.05),1)
    def test_select_not_random(self):
        loss,cov,cost=policy_losses([[.9,.9]],[[2,1]],[[0,1]],.8)
        self.assertTrue(loss[0]);self.assertTrue(cov[0]);self.assertEqual(cost[0],1)
    def test_abstain(self):
        loss,cov,_=policy_losses([[.1,.2]],[[1,2]],[[1,1]],.9)
        self.assertFalse(loss[0]);self.assertFalse(cov[0])
    def test_no_certification(self):
        r=calibrate(np.ones((5,2)),np.ones((5,2)),np.ones((5,2)),[.5,.9])
        self.assertIsNone(r['selected'])
    def test_family_penalty(self):
        s=np.ones((100,1));c=s.copy();u=np.zeros_like(s)
        a=calibrate(s,c,u,[.5]);b=calibrate(s,c,u,[.5,.6])
        self.assertGreater(b['candidates'][0]['upper_risk'],a['candidates'][0]['upper_risk'])
    def test_reproducible(self):
        a=synthetic(1,10,3);b=synthetic(1,10,3)
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    def test_invalid(self):
        with self.assertRaises(ValueError):upper_binomial(2,1,.05)
        with self.assertRaises(ValueError):calibrate([[.5]],[[1]],[[0]],[])
if __name__=='__main__':unittest.main()
