# Molecule Generation via SSL

This small project implements continuous molecule generation using a generative VAE
and Bayesian optimzation. The VAE architecture was initially introduced in
[Aspuru-guzik (2018)](https://pubs.acs.org/doi/10.1021/acscentsci.7b00572),
and retrained here on the [QM9 dataset](https://www.kaggle.com/code/zaharch/quantum-machine-9-qm9)
containing ~135k small organic molecules.

To optimize molecular properties across the learned latent space of the VAE,
we make use of the [Botorch](https://botorch.org/) framework,
which provides tools of training GP surrogate models, acquisition functions, 
and optimization tools built on top of PyTorch's autograd capabilities.

## Project Organization

The main classes and functions used for this project can be found in the `src/` directory.
Examples using these classes can be found in the Jupyter notebook `project.ipynb`.
VAEs were trained and persisted using the [MLflow](https://mlflow.org/docs/latest/index.html) library,
but are not included in the uploaded repository. 

