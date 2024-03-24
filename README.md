# DCACP

A Dyeing Clustering Algorithm based on Ant Colony Path-finding

## Background

Inspired by bionics, this study proposed a new clustering method called *DCACP*. Different from the traditional clustering method, this method refers to the phenomenon of food around the ant foraging in nature, and uses ant colony behavior, pheromone and other ideas for clustering. 

## Authors:
Shijie Zeng, Yuefei Wang, et al.

## Install

Installing **DCACP** package with pip command

```sh
pip install DCACP
```

## Usage

```python
import numpy as np
import DCACP
data=np.loadtxt("iris.csv",delimiter=",")
X=data[:,0:data.shape[1]-1]
round=2
niu=10
k=5
alpha=75
beta=10
ant_num=200

pheList,dataAnal,labels_pred =DCACP.antModel(X, round, niu, k, alpha, beta, ant_num)  # (data,round,niu,k,alpha,beta)
print(labels_pred)
```

## Parameter Details

- Input
  - data: data set.
  - round: The num of implemented rounds, every round has an ant crawling.
  - niu: The maximum number of repeated crawling at the same point.
  - k: Traversal neighbor range, k = 10, count only 10 nearest neighbors.
  - alpha: Factors controlling ant death. Is also the maximum value of normal distribution curve.
  - beta: Represents the base value to be added to the normal distribution curve. Because the first and last values of the normal distribution are the smallest, the ant will die meaninglessly.
  - ant_num: Number of ants.
- Output
  - pheList: A list of pheromones that store the final pheromones for each point.
  - dataAnal: A list of selected, storing the cumulative number of selected points.
  - typeList: The final cluster list stores the clusters corresponding to each point, which is also called *y_pred*  in some places.


## License

The MIT License (MIT)
Copyright © 2022 <copyright holders>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
