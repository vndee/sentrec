### Sentiment Graph for Product Recommendation System

#### Loader:

- [x] Load un-directed graph from [Amazon Fine Food Reviews Dataset](https://snap.stanford.edu/data/web-FineFoods.html).

#### Backbone:

- [x] GCN
- [x] SAGE
- [x] R-GCN  
- [x] SEAL

#### Learning strategy:

- [x] Single task: representation -> `[neg, neu, pos, nan]`
- [ ] Multiple task: representation -> `[nan, ext] + [neg, neu, pos]`

#### Mini-batches sampler:

- [x] Cluster GCN
- [ ] Graph SAINT
- [ ] Neighbor Sampler

#### References

- Zhang, M., & Chen, Y. (2018). **Link Prediction Based on Graph Neural Networks.** NeurIPS.
- Zhang, M., Li, P., Xia, Y., Wang, K., & Jin, L. (2020). **Revisiting Graph Neural Networks for Link Prediction.** ArXiv, abs/2010.16103.
- Schlichtkrull, M., Kipf, T., Bloem, P., Berg, R.V., Titov, I., & Welling, M. (2018). **Modeling Relational Data with Graph Convolutional Networks.** ESWC.
- Chiang, W., Liu, X., Si, S., Li, Y., Bengio, S., & Hsieh, C. (2019). **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks.** Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.
- Zeng, H., Zhou, H., Srivastava, A., Kannan, R., & Prasanna, V. (2020). **GraphSAINT: Graph Sampling Based Inductive Learning Method.** ArXiv, abs/1907.04931.
- Hamilton, W.L., Ying, Z., & Leskovec, J. (2017). **Inductive Representation Learning on Large Graphs.** NIPS.
