 # 🚀 ABOUT
 
 This repo contains Supporting Materials for the article **"Challenges and Opportunities for Machine Learning Potentials in Transition Path Sampling: Alanine Dipeptide and Azobenzene Studies"** by Nikita Fedik, Wei Li, Nicholas Lubbers, Benjamin Nebgen, Sergei Tretiak, Ying Wai Li   

 📄 [PREPRINT on ChemRxiv - v2 uploaded](https://chemrxiv.org/engage/chemrxiv/article-details/66af06f1c9c6a5c07af610b6)   
 📄 Final article on publisher site - will be added after publication

If you have any comments/questions/requests, feel free to **contact me**:    
✉️ nfedik@lanl.gov    
🌐 [nikitafedik.xyz](https://nikitafedik.xyz)

--- 
# <img src="nn.png" width="24" height="24" alt="Favicon"> ML interatomic potentials
 **HIP-NN-TS potential:**   
[hippynn - official github repo of HIP-NN-TS code](https://github.com/lanl/hippynn)    
[HIP-NN - original paper](https://pubs.aip.org/aip/jcp/article/148/24/241715/960039/Hierarchical-modeling-of-molecular-energies-using)    
[HIP-NN-TS [tensor-sensitive] - new paper](https://pubs.aip.org/aip/jcp/article/158/18/184108/2889493/Lightweight-and-effective-tensor-sensitivity-for)  

HIP-NN-TS trained to ANI-1x in TPS article   
*best model used for all results **except** active learning section*: `models/hipnnts_data_ani1x_only_before_al_seed533257`

*HIP-NN-TS trained to ANI-1x + 10800 structures from TPS trajectories in active learning section*   
path to model: `models/hipnnts_data_ani1x_tps_10800_dataseed0_modelseed<x>`   
path to training script: `models/hippnn_ani_al0.py`    
[docs for loading pretrained model](https://lanl.github.io/hippynn/examples/restarting.html)

**ANI-1X potential**  
[TorchAni - github.com/aiqm/torchani ](https://github.com/aiqm/torchani)  
[TorchAni - paper](https://pubs.acs.org/doi/10.1021/acs.jcim.0c00451)    
[ANI - original paper](https://pubs.rsc.org/en/content/articlelanding/2017/sc/c6sc05720a)  
> [!NOTE] 
> All ANI-1x calculations in the article were done by vanilla TorchANI with parameters defined in [https://github.com/aiqm/ani-model-zoo/tree/master/resources/ani-1ccx_8x](https://github.com/aiqm/ani-model-zoo/tree/master/resources/ani-1ccx_8x)    

---
# ⌨️ SCRIPTS

`scripts` contains all scripts mainly as `Jupyter notebooks` to reproduce the results in the article:
- run thermal MD for AD and AZ
- evaluate E and F using HIP-NN-TS (hippynn), ANI-1x (torchani), Amber 14 (through GAFF) and Sage by OpenFF
- get accuracy metrics, plot correlation plots and get coverage of phase space for both AD and AZ

Prerequisites:
- numpy
- torch
- natsorted
- torchani
- hippynn
- OpenMM
- OpenFF
- openbabel

It's best to create a dedicated conda environment for this project.    

---

# <img src="db.png" width="24" height="24" alt="Favicon"> DATA

 **ANI-1x database**   
[download ANI-1x dataset (5.21 GB)](https://springernature.figshare.com/articles/dataset/ANI-1x_Dataset_Release/10047041?file=18112775)   
[ANI-1x - "Less is more" paper](https://pubs.aip.org/aip/jcp/article/148/24/241733/963478/Less-is-more-Sampling-chemical-space-with-active)

> [!NOTE] 
> All data for the project is available at [Zenodo](https://zenodo.org/records/15047941). Just clone the repo and download the data in the root, e.g. `ml_tps_si`:       
> `git clone https://github.com/nikitafedik/ml_tps_si.git`    
> `cd ml_tps_si`    
> `wget https://zenodo.org/records/15047941/files/data.tar.zst`    
> `tar --use-compress-program=zstd -xf data.tar.zst`        


**alanine dipeptide (AD)**   
**10k snapshots from MD trajectories**  
`data/AD/thermal_MD_10k/DFT-logs` - DFT logs for 10k selected snapshots
`data/AD/thermal_MD_10k/` - npy arrays with E and F and xyz of important conformers

**12k points most visited configurations from TPS trajectories**

`data/AD/10800_seed0-<x>.npy`- train data     
`data/AD/1200_seed0-<x>.npy`- test data
where x:   
- E_formatioin_QM_kcal_mol = formation energies (full DFT E - E of all atoms) | kcal/mol
- G_QM_kcal_mol_A = gradients (not forces) | kcal/mol/A
- R = atomic positions | Angstrom
- Z = atomic numbers corresponsing to coordinates in R array

**azobenzene (AZ)**   
**10k snapshots from MD trajectories**  
`data/AZ/thermal_MD_10k/DFT-logs` - DFT logs for 10k selected snapshots
`data/AZ/thermal_MD_10k/` - npy arrays with E and F and xyz of important conformers

**full isomzerization path at DFT/UDFT levels**   
 path is a concatenation of (reopt of last + IRC to *trans-AD* + TS + IRC to *cis-AD* + reopt of last IRC point)     
see `data/AZ/<x>/DFT-logs` for more information or visualization   

 `data/AZ/cs-inversion/AD_cs-DFT_inversion_path.xyz` -  inversion path | closed-shell DFT  
 `data/AZ/os-rotationAD_os-DFT_rotation_path.xyz` - rotation path | open-shell DFT

# ⚙️ OTHER TOOLS:

[OpenPathSampling - Python package for TPS](http://openpathsampling.org/latest/)  
[TPS - comprehensive review](https://www.annualreviews.org/content/journals/10.1146/annurev.physchem.53.082301.113146)       
[ASE - Atomic Simulation Environment](https://wiki.fysik.dtu.dk/ase/)

