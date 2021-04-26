### Sentiment Graph for Product Recommendation System

#### Loader:

- Load un-directed graph from [Amazon Fine Food Reviews Dataset](https://snap.stanford.edu/data/web-FineFoods.html).

#### Baseline:

- [x] GCN
- [x] SAGE
- [x] R-GCN  
- [ ] GAE
- [ ] SEAL

#### Learning strategy:

- [ ] Single task: representation -> `[neg, neu, pos, nan]`
- [ ] Multiple task: representation -> `[nan, ext] + [neg, neu, pos]`

#### Mini-batches sampler:

- [x] Cluster GCN
- [ ] Graph SAINT
- [ ] Neighbor Sampler