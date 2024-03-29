## Benchmark creation for drug-target interaction (DTI) prediction task
Authors: 
- Dmitrii Traktirov
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

### Results
0. Familiarized with the existing papers about DTI and benchmark creation;
1. Familiarized with a number of frameworks and used a number of frameworks useful for the project:
    - Poetry for packaging and dependency management;
    - Wandb to track hyperparameters, system metrics, and predictions;
    - Hydra to configure application via configuration file or command line;
2. Selected suitable datasets to perform pipeline – Davis, bindingDB;
3. Selected several relevant models – KGE models (DistMult, TriModel), KGE+NFM model. Models implemented using PyTorch;
4. Created a script that allows to train/test selected model on selected dataset and get result metrics.

### Script usage
#### Resolving dependencies

This project uses **Poetry v.1.1** $-$ packaging and dependency management framework. It can be installed with the ```pip install``` command.

```
pip install poetry
```

All packages and version constraints that are required for this project are listed in *pyproject.toml* file. To install the defined dependencies for the project, just run the ```install``` command.

```
poetry install
```

Now you can simply **run the script** (with default configuration) with the following command:

```
poetry shell python main.py
```

You can type ```exit``` to exit the shell.

### Script usage example
![](running_script_example.jpg)


#### Hyperparameters and script configuration

This project uses **Hydra** framework, that allows to configure script via configuration file or command line. Configuration parameters are stored at *config/config.yaml* file.
You can override values in the loaded config from the command line:

```
poetry shell python main.py cfg.run_args.dataset=BindingDB cfg.model.kge.epoch=25
```

This will change the dataset used to BindingDB and number of epochs for KGE model to 25.


### Conclusion and further plans
The final goal of my work was not to finish the project and create the fully working benchmark on my own. It is the task for a team of developers, and it requires a lot of time. But this work allowed me to learn more about benchmark creation, familiarize myself with datasets and algorithms typically used in DTI task.

There are a lot of features we are going to implement in the future and here are just a few of them:
1. cold drug/protein start;
2. more models and datasets;
3. more methods to encode drugs and targets into vectors;
4. Create a GUI


### Literature
Dunham, Brandan, and Madhavi K. Ganapathiraju. 2022. "Benchmark Evaluation of Protein–Protein Interaction Prediction Algorithms" Molecules 27, no. 1: 41. https://doi.org/10.3390/molecules27010041 

Kexin Huang, Cao Xiao, Lucas M Glass, Jimeng Sun, MolTrans: Molecular Interaction Transformer for drug–target interaction prediction, Bioinformatics, Volume 37, Issue 6, 15 March 2021, Pages 830–836, https://doi.org/10.1093/bioinformatics/btaa880

Sameh K Mohamed, Vít Nováček, Aayah Nounu, Discovering protein drug targets using knowledge graph embeddings, Bioinformatics, Volume 36, Issue 2, 15 January 2020, Pages 603–610, https://doi.org/10.1093/bioinformatics/btz600

Xiangnan He and Tat-Seng Chua. 2017. Neural Factorization Machines for Sparse Predictive Analytics. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '17). Association for Computing Machinery, New York, NY, USA, 355–364. https://doi.org/10.1145/3077136.3080777

Yang, Bishan & Yih, Wen-tau & He, Xiaodong & Gao, Jianfeng & Deng, li. (2014). Embedding Entities and Relations for Learning and Inference in Knowledge Bases. 

Ye, Q., Hsieh, CY., Yang, Z. et al. A unified drug–target interaction prediction framework based on knowledge graph and recommendation system. Nat Commun 12, 6775 (2021). https://doi.org/10.1038/s41467-021-27137-3
