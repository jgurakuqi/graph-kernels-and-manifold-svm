# graph-kernels-and-manifold-svm


## Table of Contents

- [Requirements](#Requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)


## Requirements

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

## Installation

In order to run this project it's required a Python 3 installation, the modules libraries: 

```bash
pip install numpy
pip install matplotlib
pip install imageio
pip install scikit-learn
```

Also download the required dataset as specified in the [Requirements](#Requirements).

OPTIONALY Intel Scikit to largely improve performance on x86-compatible CPU or Intel GPU:
```bash
pip install scikit-learn-intelex 
```

## Usage

In order to run the Jupyter Notebook a compatible IDE is required. 

## Contributing

In this implementation I chose to implement the Shortest Path Kernel and use the scikit provided manifold techniques: SpectralEmbedding, LocallyLinearEmbedding.
The provided project could be use to implement further kernels and manifold techniques, to compare their performance. Also, different datasets might be used,
allowing to gain a better view about the behaviour of these techniques.

```bash
git clone https://github.com/jgurakuqi/graph-kernels-and-manifold-svm
```

## License

MIT License

Copyright (c) 2022 Jurgen Gurakuqi

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



