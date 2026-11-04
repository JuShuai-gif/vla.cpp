import unittest
import numpy as np
from int8_compensation import prepare,simulate,error_metrics

class CompensationTests(unittest.TestCase):
    def test_zero_input_and_outlier_reconstruction(self):
        rng=np.random.default_rng(3)
        w=rng.normal(size=(32,64)).astype(np.float16).astype(np.float32)
        x=np.zeros((4,64),np.float32);x[:,7]=[1,2,4,8]
        p=prepare(w,np.max(np.abs(x),axis=0),outlier_count=1)
        self.assertEqual(p["indices"].tolist(),[7])
        np.testing.assert_allclose(simulate(x,p),x@w.T,rtol=0,atol=0)
        self.assertTrue(np.all(simulate(np.zeros_like(x),p)==0))

    def test_smoothing_and_compensation_reduce_outlier_error(self):
        rng=np.random.default_rng(5)
        w=rng.normal(size=(64,128)).astype(np.float16).astype(np.float32)
        x=rng.normal(size=(32,128)).astype(np.float32);x[:,3]*=200
        ref=x@w.T;amax=np.max(np.abs(x),axis=0)
        plain=error_metrics(ref,simulate(x,prepare(w,amax)))["relative_l2"]
        smooth=error_metrics(ref,simulate(x,prepare(w,amax,.5,1)))["relative_l2"]
        self.assertLess(smooth,plain*.2)

    def test_awq_style_search_requires_separate_calibration_statistics(self):
        w=np.ones((4,32),np.float32);a=np.ones(32,np.float32)
        with self.assertRaises(ValueError):prepare(w,a,.5,method="awq")
        p=prepare(w,a,.5,method="awq",activation_mean=a)
        np.testing.assert_array_equal(p["inverse"],np.ones(32))
        self.assertEqual(p["method"],"awq")

    def test_bad_calibration_rejected(self):
        w=np.ones((4,32),np.float32)
        for a,alpha,count in [(np.ones(31),.5,0),(np.full(32,np.nan),.5,0),(np.ones(32),1.5,0),(np.ones(32),.5,32)]:
            with self.assertRaises(ValueError):prepare(w,a,alpha,count)

if __name__=="__main__":unittest.main()
