# ALGE: Active Learning Based on Graph Entropy
The repository contains the codes and data for the "Precise influence evaluation in complex networks".
# Overview
For code, we provide the codes of our method ALGE and other algorithms for comparison of influence evaluation, which are stored in **influence_evaluation** folder. And for influence maximization problem(IMP), the codes of our ALGE-Greedy algorithm and analysis is provided in **influence_maximization** folder.

For data, we provide the networks for train and prediction in our work, using **csv** format with **node** and **edge** of netwokrs. The synthetic networks for train are generated by us and attached the influence simulation results for train.
# Data
Topology data of real networks is all from Tiago P. Peixoto, "The Netzschleuder network catalogue and repository", https://networks.skewed.de/ (2020). DOI 10.5281/zenodo.7839981. The data is compressed into the dataset's **real** file, and the reference to each network is listed in the dataset **ReadMe**. We also present simulation data for methods evaluating.
The synthetic topology and labels used for training are in the **synthetic** folder.
# Code
Before performing the calculations, please install the required packages based on the **requiements** file.
## Influence evaluation
## Influence maximization
