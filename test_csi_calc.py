"""
Name: TEST_CSI_CALC
Date: 26/10/2022
Author: Vassil Savinov
Purpose: 
    Unit tests for CSI_CALC module
    Launch with `python test_csi_calc.py` or `python test_csi_calc.py -v`
"""

import unittest as ut

import numpy as np
import numpy.random as npr
import datetime as dt

import jax.config as jc
jc.update('jax_enable_x64', True)
import jax
import jax.numpy as jn
import jax.scipy as js
import jax.random as jr

import csi_calc as cc

##################

class TestCsiCalc(ut.TestCase):
    
    def test_compute_csi(self, bias_val=0.1, cat_count=10, draw_count=1000, almost_equal_sf=6):
        """
        test basic csi computation. do the computation locally using numpy, and using the routines in CSI_CALC
        NB! If one seeks to have equality beyond 6 significant figures, one will have to enable `jax_enable_x64`
        for JAX (to use float64)
        
        The histograms used for the purposes of comparison are randomly generated on the fly
        
        Arguments:
        ----------
            self: ...
            bias_val: value to add to each bin of the generated histogram (to avoid log(0))
            cat_count: number of categories (bins) in the multinomial distributions (histograms)
            draw_count: number of population members to draw for the histogram. This will be 
                the sum of elements in all bins (without the bias_val)
            almost_equal_sf: number of significant figures to compare the floating numbers to
                see unittest.assertAlmostEqual
        """
        
        assert bias_val > 1e-6
        assert cat_count > 1
        assert draw_count > 1
        assert almost_equal_sf >= 1
        
        # generate the histograms
        left_hist = npr.randint(low=0, high=cat_count, size=draw_count)+bias_val
        right_hist = npr.randint(low=0, high=cat_count, size=draw_count)+bias_val
        
        # normlazied historgams
        norm_left_hist = left_hist / np.sum(left_hist)
        norm_right_hist = right_hist / np.sum(right_hist)
        
        # compute csi locally
        calc_arr = (norm_left_hist-norm_right_hist) * np.log(norm_left_hist/norm_right_hist)
        local_csi = np.sum(calc_arr)
        
        # compute csi 
        ut_csi = cc.compute_csi(left_hist, right_hist, default_inc=0.)
        
        # compare
        self.assertAlmostEqual(local_csi, ut_csi, almost_equal_sf)
       
    ###############
    def test_csi_from_logits_trend(self, cat_count=10, shallow_draw_count=10000, deep_draw_count=100000):
        """
        Test CSI computation from logits. In partticular test that CSI decreases as the number of samples
        increases. Generate logits randomly, then use CSI_CALC routine to 
        draw random samples from multinomial distributions that correspond to these logits, and to 
        compute the CSI between histograms from these samples. Compute CSI twice. Once for few 
        draws (shallow) and once for much more samples (deep). We expect that CSI will be lower 
        in the deep case since the uncertainty is lower, i.e. two histograms drawn from the 
        same multinomial distribution will be more similar as the number of the samples grows.
        
        Use the same random key to draw shallow and deep samples in order to reduce uncertainty
        
        Arguments:
        -----------
            self: ...
            cat_count: cat_count: number of categories (bins) in the multinomial distributions (histograms), 
                see cc.compute_csi
            shallow_draw_count: number of samples to draw for the shallow CSI
            deep_draw_count: number of samples to draw for the deep CSI
        """
        
        # make show the shallow and deep csi defintions are intact
        assert cat_count > 1
        assert shallow_draw_count < deep_draw_count
        assert shallow_draw_count > 1
        
        # prepare random keys
        rnd_key = jr.PRNGKey(round(dt.datetime.now().timestamp()))
        rnd_key, cat_logit_key, csi_key= jr.split(rnd_key, num=1+2)
        
        # generate random logits
        cat_logits = jr.uniform(minval=-5, maxval=0, key=cat_logit_key, shape=[cat_count])
        
        # draw shallow and deep samples and compute csi
        #
        shallow_csi = cc.csi_from_logits(
            logits_left=cat_logits,
            logits_right=cat_logits,
            rnd_key=csi_key,
            count_left=shallow_draw_count,
            count_right=shallow_draw_count
        )
        #
        deep_csi = cc.csi_from_logits(
            logits_left=cat_logits,
            logits_right=cat_logits,
            rnd_key=csi_key,
            count_left=deep_draw_count,
            count_right=deep_draw_count
        )
        
        self.assertGreater(shallow_csi, deep_csi)
    
    #################
    
    def test_draw_csi_with_logit_variance_trend(
        self,
        cat_count=10,
        wide_logits_std=0.1,
        narrow_logits_std=0.05,
        csi_sample_count=10000,
        pop_count=10000,
        use_jit=False
    ):
        """
        Test CSI computation when left and right logits are different. Base logits are randomly
        generated, whilst left/right logits are same as base with added normal noise on top. Generate left/right logits
        and the histograms `csi_sample_count` times with `pop_count` in each histogram.
        Test for two different magnitudes of noise, `wide_logits_std`
        and `narrow_logits_std`. Test that the average CSI for the narrow_std is smaller.tainty
        
        Arguments:
        -----------
            self: ...
            cat_count: cat_count: number of categories (bins) in the multinomial distributions (histograms), 
                see cc.compute_csi
            wide_logits_std: large standard deviation for the normally distributed noise added to the right logits
            narrow_logits_std: small standard deviation for the normally distributed noise added to the right logits
            csi_sample_count: how many (pairs of) histograms to draw and how many csi-s to compute
            pop_count: how many elements to have in each histogram
            use_jit: whether to use JIT for drawing CSI-s, see `cc.draw_csi_with_logit_variance`
        """
        
        # sanity checks
        assert cat_count > 1
        assert wide_logits_std > narrow_logits_std
        assert narrow_logits_std > 1e-6
        assert csi_sample_count > 1
        assert pop_count > 1
        
        tick = dt.datetime.now()
        
        # prepare random keys
        rnd_key = jr.PRNGKey(round(dt.datetime.now().timestamp()))
        rnd_key, cat_logit_key = jr.split(rnd_key, num=1+1)
        
        # generate random logits
        cat_logits = jr.uniform(minval=-5, maxval=0, key=cat_logit_key, shape=[cat_count])
        
        rnd_key, narrow_key, wide_key = jr.split(rnd_key, num=1+2)
        
        print(f'Getting average CSI for narrow logits_std ({narrow_logits_std:.3f})')
        
        narrow_csi_samples = cc.draw_csi_with_logit_variance(
            rnd_key=narrow_key,
            base_logits=cat_logits,
            logits_std=narrow_logits_std,
            sample_count=csi_sample_count,
            count_left=pop_count,
            count_right=pop_count,
            use_jit=use_jit
        )
        
        print(f'Getting average CSI for wide logits_std ({wide_logits_std:.3f})')
        wide_csi_samples = cc.draw_csi_with_logit_variance(
            rnd_key=narrow_key,
            base_logits=cat_logits,
            logits_std=wide_logits_std,
            sample_count=csi_sample_count,
            count_left=pop_count,
            count_right=pop_count,
            use_jit=use_jit
        )
        
        narrow_csi = jn.mean(narrow_csi_samples)
        wide_csi = jn.mean(wide_csi_samples)
        
        tock = dt.datetime.now()
        time_in_sec = (tock-tick).seconds + (tock-tick).microseconds * 1e-6 
        print(f'Time taken {time_in_sec:.1f} sec')
        
        self.assertGreater(wide_csi, narrow_csi)
        
        
    ####
    
    def test_draw_csi_with_logit_variance_trend_jit(
        self,
        cat_count=10,
        wide_logits_std=0.1,
        narrow_logits_std=0.05,
        csi_sample_count=10000,
        pop_count=10000
    ):
        """
        Same as test_draw_csi_with_logit_variance_trend, but use_jit
        """

        self.test_draw_csi_with_logit_variance_trend(
            cat_count=cat_count,
            wide_logits_std=wide_logits_std,
            narrow_logits_std=narrow_logits_std,
            csi_sample_count=csi_sample_count,
            pop_count=pop_count,
            use_jit=True
        )
        
##################

if __name__ == '__main__':
    ut.main()