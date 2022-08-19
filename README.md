# Are Dipolarization Fronts a Typical Feature of Magnetotail Plasma Jets Fronts?
[![GitHub license](https://img.shields.io/badge/license-Apache_2.0-blue.svg)](./LICENSE.txt) [![LASP](https://img.shields.io/badge/datasets-MMS_SDC-orange.svg)](https://lasp.colorado.edu/mms/sdc/)

Code for the paper Are Dipolarization Fronts a Typical Feature of Magnetotail Plasma Jets Fronts?

## Abstract

Plasma jets are ubiquitous in the Earth's magnetotail. Plasma jet fronts (JFs) are the seat of particle acceleration and energy conversion. JFs are commonly associated with dipolarization fronts (DFs) representing solitary sharp and strong increases in the northward component of the magnetic field. However, MHD and kinetic instabilities can develop at JFs and disturb the front structure which questions on the occurrence of DFs at the JFs. We investigate the structure of JFs using 5 years (2017-2021) of the Magnetospheric Multiscale observations in the CPS in the Earth's magnetotail. We compiled a database of 2394 CPS jets. We find that about half (42\%) of the JFs are associated with large amplitude changes in $B_z$. DFs constitute a quarter of these large-amplitude events, while the rest are associated with more complicated magnetic field structures. We conclude that the ``classical" picture of DFs at the JFs is not the most common situation.

## Reproducing our results
- Instructions for reproduction are given within each section folder, in 
  the associated README.md file.

## Requirements
- A [`requirements.txt`](./requirements.txt) file is available at the root 
  of this repository, specifying the required packages for our analysis. To install 
  the required packages run `pip install -r requirements.txt`

- Routines specific to this study [`FastFlows`](./FastFlows) is 
  pip-installable: from the [`FastFlows`](./FastFlows) folder run `pip 
  install .`


## Acknowledgement
We thank the entire MMS team and instrument PIs for data access and support.
All of the data used in this paper are publicly available from the MMS 
Science Data Center https://lasp.colorado.edu/mms/sdc/. Data analysis was 
performed using the pyrfu analysis package available at 
https://pypi.org/project/pyrfu/. This work is supported by the SNSA grant 
139/18.