# FairDen: Fair Density-based Clustering

This repository is the official implementation of [FairDen: Fair Density-Based Clustering](https://iclr.cc/virtual/2025/poster/29171), accepted at ICLR 2025.

---

## 🔄 Fork Information

This repository is a **fork and extension** of the original FairDen implementation by the authors. It has been used, extended, and re-experimented by a student team at **University of Information Technology (UIT), Vietnam National University Ho Chi Minh City**.

### Team Members
- **Phan Nhựt Anh**
- **Nguyễn Trung Quân**
- **Thái Hoàng Phúc**

### Our Contributions
We have made the following additions and modifications to the original codebase:

| File | Description |
|------|-------------|
| `src/experiments/compas_experiment.py` | Experiments on the COMPAS dataset (recidivism prediction) |
| `src/experiments/student_experiment.py` | Experiments on the Student Performance dataset |
| `src/experiments/compas_hyperparam_search.py` | Hyperparameter optimization for COMPAS dataset |
| `src/experiments/student_hyperparam_search.py` | Hyperparameter optimization for Student dataset |
| `config/realworld/compas.json` | Configuration for COMPAS dataset |
| `config/realworld/student_address.json` | Configuration for Student dataset with address as sensitive attribute |
| `scripts/visualize_balance.py` | Visualization script for Balance metric comparison |

---

## Three moons

![Three Moons](auxiliary/Plots/MotivationFairDen.png)
## Setup / Installation guide
Requirements: Python 3.9 
```bash
pip install -r requirements.txt
```
## Experiments
### Real World experiments
![Real World bar plot](auxiliary/Plots/rw_balance_recalc.svg)
![Legend](auxiliary/Plots/Legend_barplot.png)
adjust main to the following:
```python
from src.experiments.realworld_experiment import realworld_experiment
def main():
    realworld_experiment()
if __name__ == "__main__":
    main()
```
run 
```bash
python3 main.py
```


### k-line experiments
![Line plots](auxiliary/Plots/Lineplot_adult_both.svg)
![Legend](auxiliary/Plots/Legend.png)
adjust main to the following:
```python
from src.experiments.adult_experiment import adult_experiment
def main():
    adult_experiment()
if __name__ == "__main__":
    main()
```
run 
```bash
python3 main.py
```

### Categorical experiments

adjust main to the following:
```python
from src.experiments.categorical_experiments import categorical_experiments
def main():
    categorical_experiments()
if __name__ == "__main__":
    main()
```

run 
```bash
python3 main.py
```

### COMPAS Experiment (Our Addition)

Run experiments on the COMPAS recidivism dataset:
```python
from src.experiments.compas_experiment import compas_experiment
def main():
    compas_experiment()
if __name__ == "__main__":
    main()
```

### Student Performance Experiment (Our Addition)

Run experiments on the Student Performance dataset:
```python
from src.experiments.student_experiment import student_experiment
def main():
    student_experiment()
if __name__ == "__main__":
    main()
```

## Structure of the repository

```bash
.
├── auxiliary                       # auxiliary files for plotting, additional experiments, parameter optimization
│   ├── AuxExperiments              # Runtime and three moons experiment
│   ├── Parameters                  # parameter optimization results
│   └── Plots                       # plots
│
├── config  
│   ├── realworld                   # configuration files for realworld datasets
│   └── three_moons                 # configuration files for three moons dataset
│
├── data  
│   └── realworld                   # realworld datasets 
│
├── results                         # experiment results
│   ├── rw_experiment               # Real-world experiment results
│   ├── compas_experiment           # COMPAS experiment results (our addition)
│   └── student_experiment          # Student experiment results (our addition)
│              
├── src
│   ├── comparative_methods         # implementations for other methods
│   ├── dc_dist                     # dc_distance
│   ├── evaluation                  # evaluation: balance, dcsi, noise percentage
│   ├── experiments                 # experiments 
│   ├── utils                       # DataLoader, DataEncoder
│   └── FairDen.py                  # our method
│
├── scripts                         # utility scripts (our addition)
│   └── visualize_balance.py        # Balance visualization script
│ 
├── .gitignore                      # ignore files that cannot commit to Git
├── LICENSE                         # license file  
├── main.py                         # main to call experiments  
├── README.md                       # project description   
└── requirements.txt                # dependencies  
```
## Citation
If you use our method or code from this repository, please cite our paper:
Lena Krieger*, Anna Beer*, Pernille Matthews, Anneka Myrup Thiesson, Ira Assent, (2025, April). FairDen: Fair Density-based Clustering. Accepted for publication at the *Thirteenth International Conference on Learning Representations (ICLR)*.
```
@unpublished{kriegerbeer2025fairden,
  title        =    {FairDen: Fair Density-based Clustering},
  author       =    {Krieger*, Lena and Beer*, Anna and Matthews, Pernille and Thiesson, Anneka Myrup and Assent, Ira},
  url          =    {https://openreview.net/forum?id=aPHHhnZktB},
  year         =    {2025},
  note         =    {Accepted for publication at The Thirteenth International Conference on Learning Representations,
                    (ICLR) 2025}
}
```
## License

### Code
This work is licensed under the [Apache 2.0 License](LICENSE). This license is valid for all code files implemented by us.

### Data
The following datasets are taken from [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) and licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0) license](https://creativecommons.org/licenses/by/4.0/legalcode):
- Bank dataset 
- Communities and Crime
- Diabetic Dataset
- UCI Census 
- COMPAS (ProPublica)
- Student Performance

Data that we generated with [DENSIRED](https://github.com/PhilJahn/DENSIRED) or our motivational three moons dataset are licensed under [Creative Commons Zero (CC0) license](https://creativecommons.org/public-domain/cc0/).




<!-- MARKDOWN LINKS & IMAGES -->

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
