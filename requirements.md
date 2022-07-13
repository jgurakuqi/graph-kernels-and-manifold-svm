Read this [article](https://www.dsi.unive.it/~atorsell/AI/graph/Unfolding.pdf) presenting a way to improve the disciminative power of graph kernels.

Choose one [graph kernel](https://www.dsi.unive.it/~atorsell/AI/graph/kernels.pdf) among

* Shortest-path Kernel
* Graphlet Kernel
* Random Walk Kernel
* Weisfeiler-Lehman Kernel
* Choose one manifold learning technique among

Isomap
Diffusion Maps
Laplacian Eigenmaps
Local Linear Embedding
Compare the performance of an SVM trained on the given kernel, with or without the manifold learning step, on the following datasets:

* [PPI]
* [Shock]

The zip files contain csv files representing the adjacecy matrices of the graphs and of the lavels. the files graphxxx.csv contain the adjaccency matrices, one per file, while the file labels.csv contains all the labels