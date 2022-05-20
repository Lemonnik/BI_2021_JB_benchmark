## Benchmark creation for drug-target interaction (DTI) prediction task
Author: 
- Dmitrii Traktirov

Supervisor:
- Ellen Kartysheva


### Introduction
Drug-target interaction prediction (DTI) task plays an important role in the drug discovery process, which aims to identify new drugs for biological targets. Automation of prediction will speed up the process of creating new drugs. Now there are many machine learning models that solve this problem, however, due to the presence of a huge number of different datasets and testing protocols, it is difficult to compare different models with each other. And so one unified benchmark is needed.


### Aim and tasks
The **aim** of this project was to create a benchmark for drug-target interaction (DTI) prediction task. The following **tasks** were set in order to achive the goal:

0. Familiarize with the existing papers about DTI and benchmark creation;
1. Select suitable datasets to perform pipeline. Potential candidates are KiBA, Yamanishi_08, Davis, etc.;
2. Create an evaluation protocol;
3. Implement several most relevant models and test them using created evaluation protocol;
4. Create GUI/script


### Datasets
**BindingDB** and **Davis** are two databases of measured binding affinities, focusing chiefly on the interactions of proteins considered to be candidate drug-targets with ligands that are small, drug-like molecules. 
As of May 4, 2022, BindingDB contains 41,296 Entries, each with a DOI, containing 2,513,948 binding data for 8,839 protein targets and 1,077,922 small molecules.
Davis Kinase binding affinity dataset contains the interaction of 72 kinase inhibitors with 442 kinases covering >80% of the human catalytic protein kinome.


### Models
**DistMult** is a knowledge graph embedding (KGE) model that allows to learn the low-rank representations for all entities and relations using one embedding vector. In terms of current task entities are drugs and proteins, whereas relation is the existence of interaction between them.

**TriModel** is a KGE model based on tensor factorization that extends the DistMult and ComplEx models. It represents each entity and relation using three embedding vectors.

Neural Factorization Machine (NFM) combines the linearity of FM in modelling second-order feature interactions and the non-linearity of neural network in modelling higher-order feature interactions.
**KGE_NFM** model allows to train on of the KGE models and pass its embeddings into NFM model as features (along with drugs and targets features).



### Script usage

This project uses **Poetry** $-$ packaging and dependency management framework. It can be installed with the ```pip install``` command.

```
pip install poetry
```

All packages and version constraints that are required for this project are listed in *pyproject.toml* file. To install the defined dependencies for your project, just run the ```install``` command.

```
poetry install
```

Now you can simply **run the script** with the following command:

```
poetry shell python main.py
```

You can type ```exit``` to exit the shell.


### Conclusion and further plans

In further work we are going to:
1. Implement cold drug/protein start;
2. Implement more models and datasets;
3. Create a GUI


### Literature
Ye, Q., Hsieh, CY., Yang, Z. et al. A unified drug–target interaction prediction framework based on knowledge graph and recommendation system. Nat Commun 12, 6775 (2021). https://doi.org/10.1038/s41467-021-27137-3

Yang B. et al.  (2015) Embedding entities and relations for learning and inference in knowledge bases. In: ICLR.

Sameh K Mohamed, Vít Nováček, Aayah Nounu, Discovering protein drug targets using knowledge graph embeddings, Bioinformatics, Volume 36, Issue 2, 15 January 2020, Pages 603–610, https://doi.org/10.1093/bioinformatics/btz600

etc. (will be added)
