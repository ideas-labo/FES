# Contexts Matter: An Empirical Study on Contextual Influence in Fairness Testing for Deep Learning Systems

<!-- ABOUT THE PROJECT -->

## About The Project

This online appendix is supplementary to the paper entitled "Contexts Matter: An Empirical Study on Contextual Influence in
Fairness Testing for Deep Learning Systems". It contains the raw results, code for the proposed approach, and Colab script to replicate our experiment's result analysis.This README file describes the structure of the provided files (Raw data, source code and results). as well as information on the content of this repository.

## Table of Content

<!-- TABLE OF CONTENTS -->

<details open="open">
<summary></summary>
<ol>
<li><a href="#about-the-project">About The Project</a></li>
<li><a href="#Table of Content">Table of Content</a></li>
<li><a href="#getting-started">Getting Started</a></li>
<li><a href="#Data">Data</a></li>
<li><a href="#Statistical Analysis">Statistical Analysis</a></li>
<li><a href="#Result">Result</a></li>
</ol>
</details>

## Getting Started

### Prerequisites

The codes have been tested with **Python 3.7** and **Tensorflow 2.x**

### Installation

```!git clone https://github.com/anonymoususr970416/Fair_EStudy ```

The easiest way to execute the codes for the proposed approach is to download the .ipynb script you need and run it in Google Colaboratory, where you can execute the Python code or R code in your broswer. Otherwise, you may encounter some configuration issues.
### Documents


**generator**:  

    contains three generator in the paper.
**model**:  

    contains utility functions to build DNN models.
**preprocessing**:  

    contains the function that preprocessing the data .
**evaluation**:  

    contains functions to evaluate adequacy metric and fairness metric.     
**result**:  

    contains the raw experiment results and supplementary document for all the research questions.
**statistical_analysis**:  

    contains functions to analysis the raw data.
**dataset**:  

    performance datasets of 12 fairness dataset as specified in the paper.

### Running
**Python IDE (e.g. Pycharm)**: Open the evlation file on the IDE, and choose the metric on the paper and run to create the raw data about rq1.
According the raw data run the statistical_analysis file to create the rq2,3,4 result data. Show in the <a href="#Result">Result</a>

## Data

### Experiment dataset

In data directory, you can find the raw data used in our experiments in Folder Datasets. These files will be loaded automatically in each model script.



## Statistical Analysis

Before performing Scott-Knott Analysis, please put the data into a .xlsx file ,and the rank as shown.

where elements in this excel table is metric value produced as the ouput of models. 
The data can be easily used in `statistical_analysis/statisticalanalysis.ipynb`, which is based on R, to perform Scott-Knott Analysis and Kruskal-Wallis test Analysis.


## Result

This folder contains all the experimental results to answer the research questions in our paper.The result files are the csv files that contain the metric value for 30 run for certain model in all experiments. 

We organize the results according to the RQ they answer:  

RQ1: This directory contains the 10 different metrics result in 12 datasets.  

- rq1:([https://github.com/anonymous970416/Fair_EStudy](https://github.com/anonymous970416/Fair_EStudy/blob/main/result/rq1_full/RQ1_fulltable.pdf))
  The supplementary document for RQ1 in the paper.

RQ2: This directory contains all hirmonic_mean in 3 contexts(HP,LB,SB) on adequacy and fairness metric.  
- rq2:([https://github.com/anonymous970416/Fair_EStudy](https://github.com/anonymous970416/Fair_EStudy/blob/main//result/rq2_full/rq2-1.pdf))
- rq2:([https://github.com/anonymous970416/Fair_EStudy](https://github.com/anonymous970416/Fair_EStudy/blob/main//result/rq2_full/rq2-2.pdf))
  The supplementary document for RQ2 in the paper.

RQ3: This directory contains all FDC and AC length in 3 contexts(HP,LB,SB).   
- rq3:([https://github.com/anonymous970416/Fair_EStudy](https://github.com/anonymous970416/Fair_EStudy/blob/main/result/rq3_full/rq3-1.pdf))
- rq3:([https://github.com/anonymous970416/Fair_EStudy](https://github.com/anonymous970416/Fair_EStudy/blob/main/result/rq3_full/rq3-2.pdf))
The supplementary document for RQ3 in the paper.

RQ4: The directory contains all metric dataset, and the correlation as shown.


