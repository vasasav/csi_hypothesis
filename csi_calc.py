"""
Name: CSI_CALC
Date: 26/10/2022
Author: Vassil Savinov
Purpose: 
    Routines to calculate CSI between histograms and build hypothesis testing for assessing whether the histograms are different.
    The working assumption is that the underlying process that generates the histograms is drawing from multinomial distribution. The logits
    for the multinomial dstribution are not precise variables instead, all logits are modelled as normally distributed random variables with
    the same variance. i.e. (using LaTeX notation, see https://quicklatex.com/)

    $$
    \begin{align*}
    \{X_1, \dots, X_n\} &\sim Multinomial(n, p_1, \dots p_k) \\
    p_i&=\sigma\left(l_i\right)/\sum_j\sigma\left(l_j\right)\,\forall i=1\dots k \\
    l_i &\sim N(\mu_i,\rho^2)
    \end{align*}
    $$

    Where $\sigma(x)=\left(1+\exp(-x)\right)^{-1}$ is the sigma-function, $\mu_i$ for $i=1\dots k$ are the mean logits and $\rho$ is the single 
    common standard deviation.

    Note that the module is designed to be run with JAX
"""

import jax.config as jc
#jc.update('jax_enable_x64', True)

import jax
import jax.numpy as jn
import jax.scipy as js
import jax.random as jr

import typing as tp

################################

@jax.jit
def compute_csi(
    left_hist: tp.List[float],
    right_hist: tp.List[float],
    default_inc: int=1
)->float:
    """
    Given two histograms compute the csi between them. Written for JAX
    
    Arguments:
    ----------
        left_hist: a 1-d array of the counts in the 'left histogram'
        right_hist: a 1-d array of counts in the 'right' histogram
        default_inc: a number to add to each bin in the histogram
            Used to ensure we don't end up taking log(0) for bins with no counts
            
    Returns:
    --------
        float64 a CSI between the two histograms
    """
    
    # make sure the shapes are equal and sane
    assert left_hist.shape==right_hist.shape
    assert len(left_hist.shape)==1
    assert left_hist.shape[0] >= 1
    
    # adjust the population by default increment
    mod_right_hist = right_hist + default_inc
    mod_left_hist = left_hist + default_inc
    
    # get relative population
    rel_right_hist = mod_right_hist/jn.sum(mod_right_hist)
    rel_left_hist = mod_left_hist/jn.sum(mod_left_hist)
    
    # compute and return CSI
    pre_csi_arr = ( rel_right_hist - rel_left_hist ) * jn.log(rel_right_hist/rel_left_hist)
    
    return jn.sum(pre_csi_arr)

#############################

def csi_from_logits(
    logits_left: tp.List[float],
    logits_right: tp.List[float],
    rnd_key: tp.List[float],
    count_left: int=1000,
    count_right: int=1000,
    csi_default_inc: int=1
)->float:
    """
    Given logits for the left and right categorical distirbutions, sample 'count_left' population members
    from the left distribution, and equivalent from the right. Convert them to histograms (i.e. count number of
    population members in each category). Then compute the CSI between these two histograms
    using compute_csi
    
    Arguments:
    ---------
        * logits_left: left distirbution logits, an array of floating point numbers. Does NOT need to be normalized
        * logits_right: right distribution logits, .. same size as logits_left. Does NOT need to be normalized
        * rnd_key: array with two integer numbers. used to initialize random number generators
        * count_left: number of population members to draw from the left categorical distribution
        * count_right: number of population members to draw from the right categorical distribution
        * csi_default_inc: number of population members to add to each bin in the histogram. Is used to 
            protect from computing log(0)
    
    Returns:
    --------
        csi between the two histograms
    """
    
    # santity checks
    assert logits_left.shape==logits_right.shape
    assert count_left>1
    assert count_right>1
    
    rnd_key, left_key, right_key = jr.split(rnd_key, num=1+2)
    
    #### left histogram
    left_hist = jn.histogram(
        jr.categorical(logits=logits_left, key=left_key, shape=[count_left]),
        bins=logits_left.shape[0]-1
    )[0]
    
    #### right histogram
    right_hist = jn.histogram(
        jr.categorical(logits=logits_right, key=right_key, shape=[count_right]),
        bins=logits_right.shape[0]-1
    )[0]
    
    #### csi
    csi_val = compute_csi(
        left_hist=left_hist,
        right_hist=right_hist,
        default_inc=csi_default_inc
    )
    
    return csi_val

####

def draw_csi_with_logit_variance(
    rnd_key: tp.Iterable[int],
    base_logits: tp.Iterable[float],
    logits_std: float,
    sample_count: int=10000,
    count_left: int=10000,
    count_right: int=10000,
    use_jit: bool=False
)->tp.Iterable[float]:
    """
    Given a base logits distribution and standard deviation, generate the left and right logits by adding
    normally distributed noise to the base (with specified standard deviation).
    
    Use left and right logits to generate two histograms and get the CSI between the two draws (see csi_from_logits).
    Repeat this for csi_sample_count times. Return results
    
    Argument:
    --------
        * rnd_key: array with two integers. used to initialize the random number generators. see jax.random.split
        * base_logits: array with floating point numbers. Used to generate the left and right logits
        * logits_std: a single standard deviation to apply to all logits to get the right logits from base logits
        * sample_count: number of times to compute csi
        * count_left: number of population members in the left histogram (see csi_from_logits)
        * count_right: ...
        * use_jit: compile the csi_from_logits for this step. Makes sense for large count_left/count_right
        
    Returns:
    --------
        array of floating point numbers - the CSI from the drawn histograms
    """
    
    # sanity checks
    assert len(base_logits.shape)==1
    assert base_logits.shape[0] > 1
    assert logits_std > 1e-6
    assert count_left > 1
    assert count_right > 1
    assert rnd_key.shape == (2,)
    
    # optional use to use jit compile. cannot compile the csi_from_logits in principle
    # due to dependends of shapes of generated arrays on count_left/count_right, however
    # once these are fixed, jit-compile is possible. it may however be unnecessarily slow
    # for low count_left/count_right, so use judgement
    if use_jit:
        csi_gen = jax.jit(lambda logits_left, logits_right, rnd_key: csi_from_logits(
                logits_left,
                logits_right,
                rnd_key,
                count_left=count_left,
                count_right=count_right
        ))
    else:
        csi_gen = lambda logits_left, logits_right, rnd_key: csi_from_logits(
                logits_left,
                logits_right,
                rnd_key,
                count_left=count_left,
                count_right=count_right
        )
    
    # right/left logits from a normal distribution with the center given by the
    # base_logits
    rnd_key, logits_left_key, logits_right_key = jr.split(rnd_key, num=1+2)
    logits_left_arr = base_logits + logits_std * jr.normal(key=logits_left_key, shape=[sample_count, base_logits.shape[0]])
    logits_right_arr = base_logits + logits_std * jr.normal(key=logits_right_key, shape=[sample_count, base_logits.shape[0]])
    
    # prepare random keys to generate different draws from the multinomial distirbution
    new_keys = jr.split(rnd_key, num=1+sample_count)
    rnd_key = new_keys[0, :]
    gen_keys = new_keys[1:, :]
    
    #
    # generate csi-s in in a vectorized opertation
    csi_samples = jax.vmap(
        csi_gen,
        in_axes=(0, 0, 0),
        out_axes=0
    )(logits_left_arr, logits_right_arr, gen_keys)
    
    return csi_samples

