# pyugm

A package for building, doing inference with, and learning undirected graphical models.

I am interested in a package that is easy to use, and I want to specifically explore easy ways of defining
models without using mini languages like those used by Markov logic networks and relational Markov networks.
I also don't want to assume iid data - so a model must specify the complete network of observations.

So far, the following features are implemented for Markov networks:

- Specify a discrete model by listing factors. A cluster graph is then automatically constructed.
- Run the loopy belief update algorithm to calibrate the model. The flooding protocol and distribute-collect update
    orders are implemented, but custom orderings are easy to inject.
- Parameters are specified for each factor - allowing arbitrary parameter tying.
- Learn parameters by maximising the log-posterior with LM-BFGS (scipy) or some other optimizer. A gaussian prior's
    parameters can be specified.

## Example
See http://nbviewer.ipython.org/github/dirko/pyugm/blob/master/examples/pyugm%20models.ipynb for a
notebook showing how models can be specified.
