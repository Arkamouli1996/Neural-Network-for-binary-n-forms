# machine-learning-math

## Project: Littlewood-Richardson Coefficients

Littlewood-Richardson coefficients (LR coefficients in short) [Reference](https://www.symmetricfunctions.com/littlewoodRichardson.htm) $c_{\lambda, \mu}^\nu$ are non-negative number indexes by a triple of partitions $(\lambda, \mu, \nu)$. They are the *structural constants* in the representation ring of the symmetric group $S_n$ (equivalently, the structural constants for *Schur symmetric functions*). That is, $s_\lambda s_\mu = \Sigma_{\nu}c_{\lambda, \mu}^\nu s_\nu.$

This project folder contains a sagemath notebook that generates triples of partitions whose Ferres diagrams are bounded by a $k\times n$-box and the associated LR coefficients. Various representations of the partitions have been implemented, including list, matrix, 0-1 lattice path, length, durfee square. The folder also includes jupyter notebooks of a simple feedforward neural network with 16 hidden dimensions, and a transformer model with one transformer layer. Both are able to achieve more than 90% of training and testing accuracies. 
