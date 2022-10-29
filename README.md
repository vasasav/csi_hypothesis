# Hypothesis testing using CSI

The purpose of this project is to enable hypothesis-testing style analysis of CSI. CSI, i.e. Charateristic Stability Index is a known way of comparing sets of observations of a random variable, with a purpose of detecting distribution drift. Whilst computing this metric is easy, understanding what the different values mean for the case in question is difficult. This repo provides tools for extracting meaningful thresholds for CSI.

Files:
-----

* [csi_calc.py](csi_calc.py) contains the core tools for estimating the distribution of CSI metric

* [test_csi_calc.py](test_csi_calc.py) contains the unit-tests

* [extract_weather_patterns_v2.ipynb](extract_weather_patterns_v2.ipynb) is a worked example of extracting temperature distributions for different locations in UK, and then using CSI to decide whether these temperature distributions are sufficiently different

* [csi_hypothesis_writeup.ipynb](csi_hypothesis_writeup.ipynb) is the write up for the CSI and hypothesis testing. It will also be published as a separate blogpost
