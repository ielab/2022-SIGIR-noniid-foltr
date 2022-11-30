# Is Non-IID Data a Threat in Federated Online Learning to Rank?
The official repository for [Is Non-IID Data a Threat in Federated Online Learning to Rank?](https://arxiv.org/pdf/2204.09272.pdf), *SIGIR2022*

Here are few steps to reproduce our experiments.

## Setup python environment
Create a conda environment for running this code using the code below.

````
conda create --name federated python=3.6
source activate federated
# assuming you want to checkout the repo in the current directory
git clone https://github.com/ielab/2022-SIGIR-noniid-foltr.git && cd 2022-SIGIR-noniid-foltr
pip install -r requirements.txt 
````

## Reproducing results
To reproduce our experiments result, set up corresponding parameters and run files: `./runs/run_fpdgd_non_iid_linear.py` `./runs/run_fpdgd_non_iid_neural.py`
```
python run_fpdgd_non_iid_linear.py
```
```
python run_fpdgd_non_iid_neural.py
```
or run files: `./scripts/run_fpdgd_noniid_linear.sh` `./scripts/run_fpdgd_noniid_neural.sh`
```
sh run_fpdgd_noniid_linear.sh
```
```
sh run_fpdgd_noniid_neural.sh
```

## Citation
If you find this code or the ideas in the paper useful in your research, please consider citing the paper:
```
@inproceedings{wang2022non,
author = {Wang, Shuyi and Zuccon, Guido},
title = {Is Non-IID Data a Threat in Federated Online Learning to Rank?},
year = {2022},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531709},
doi = {10.1145/3477495.3531709},
booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {2801–2813},
numpages = {13},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```
