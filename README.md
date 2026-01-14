# FairDen: Phân cụm công bằng dựa trên mật độ 

Repository này là phiên bản mở rộng và phát triển dựa trên mã nguồn chính thức của bài báo [FairDen: Fair Density-Based Clustering](https://iclr.cc/virtual/2025/poster/29171), (ICLR 2025). Với mục đích kiểm tra tính đúng đắn và mở rộng thực nghiệm trên những tập dữ liệu mới.


### Thành viên nhóm
- **Phan Nhựt Anh** - MSSV: 23127023
- **Nguyễn Hoàng Quân** - MSSV: 23127106
- **Thái Hoàng Phúc** - MSSV: 23127458

### Các đóng góp bổ sung của nhóm:

| File | Mô tả | Thực nghiệm | 
|------|-------|-----------|
| `src/experiments/compas_experiment.py` | Thực nghiệm trên tập dữ liệu COMPAS | Phân cụm công bằng trên dữ liệu thực tế
| `src/experiments/student_experiment.py` | Thực nghiệm trên tập dữ liệu Student Performance | Phân cụm công bằng trên dữ liệu thực tế, Phân cụm với dữ liệu nhiều biến phân loại
| `src/experiments/compas_hyperparam_search.py` | Tối ưu hóa siêu tham số cho tập COMPAS | Tìm kiếm tham số tối ưu cho thuật toán FairDen
| `src/experiments/student_hyperparam_search.py` | Tối ưu hóa siêu tham số cho tập Student | Tìm kiếm tham số tối ưu cho thuật toán FairDen
| `src/experiments/census_experiment.py` | Thực nghiệm trên tập dữ liệu Census | Phân cụm công bằng trên dữ liệu có nhiều thuộc tính nhạy cảm
| `src/experiments/census_hyperparam_search.py` | Tối ưu hóa siêu tham số cho tập Census | Tìm kiếm tham số tối ưu cho thuật toán FairDen
---

## Tập dữ liệu Three moons (Ba trăng khuyết)

![Three Moons](auxiliary/Plots/MotivationFairDen.png)

## Hướng dẫn cài đặt
Yêu cầu: Python 3.9 
```bash
pip install -r requirements.txt
```

## Các thực nghiệm
### I. Tìm kiếm siêu tham số

Nhóm đã bổ sung việc tối ưu hóa siêu tham số DBSCAN cho các tập dữ liệu COMPAS và Student Performance dựa trên phương pháp của tác giả

#### Cách chạy thực nghiệm
```python
from src.experiments.compas_hyperparam_search import compas_hyperparam_search
from src.experiments.student_hyperparam_search import student_hyperparam_search

def main():
    compas_hyperparam_search()   # Cho tập COMPAS
    student_hyperparam_search()  # Cho tập Student
    census_hyperparam_search()   # Cho tập Census
if __name__ == "__main__":
    main()
```

### Thực nghiệm Real World
#### Kết quả của tác giả trong bài báo gốc:
![Real World bar plot](auxiliary/Plots/rw_balance_recalc.svg)
![Legend](auxiliary/Plots/Legend_barplot.png)
#### Kết quả của nhóm:
![Real World bar plot](visualization/balance_comparison.png)
#### Cách chạy thực nghiệm
```python
from src.experiments.realworld_experiment import realworld_experiment
def main():
    realworld_experiment()
if __name__ == "__main__":
    main()
```

### II. Thực nghiệm về độ ổn định đối với số lượng cụm ($k$)
#### Kết quả của tác giả trong bài báo gốc:
![Line plots](auxiliary/Plots/Lineplot_adult_both.svg)
![Legend](auxiliary/Plots/Legend.png)
#### Kết quả của nhóm:
![Line plots](visualization/kline_adult_comparison.png)
#### Kết quả trên tập dữ liệu COMPAS:
![Line plots](visualization/kline_compas_comparison.png)
#### Cách chạy thực nghiệm
```python
from src.experiments.adult_experiment import adult_experiment
from src.experiments.compas_experiment import compas_experiment
def main():
    adult_experiment() # Thực nghiệm trên tập dữ liệu Adult (của tác giả)
    compas_experiment() # Thực nghiệm trên tập dữ liệu COMPAS (bổ sung của nhóm)
if __name__ == "__main__":
    main()
```

### III. Thực nghiệm với thuộc tính phân loại:

```python
from src.experiments.categorical_experiments import categorical_experiments
def main():
    categorical_experiments()
if __name__ == "__main__":
    main()
```
### IV. Thực nghiệm với nhiều thuộc tính nhạy cảm:
```python
from src.experiments.adult_experiment import adult_experiment
from src.experiments.census_experiment import census_experiment
def main():
    adult_experiment()
    census_experiment()
if __name__ == "__main__":
    main()
```

---

## Cấu trúc thư mục

> **Lưu ý:** Các file/thư mục không được mô tả chức năng là các file gốc hoặc kiểm thử của tác giả mà nhóm không sử dụng đến.

```bash
.
├── auxiliary                       # File phụ trợ của tác giả
│   ├── AuxExperiments              
│   ├── Parameters             
│   └── Plots                       # Biểu đồ gốc của tác giả
│
├── config  
│   ├── realworld/                  # File cấu hình cho tập dữ liệu thực tế
│   │   ├── adult.json              # Adult (race)
│   │   ├── adult4.json             # Adult (gender)
│   │   ├── bank.json               # Bank (marital) với categorical
│   │   ├── bank3.json              # Bank (marital) không categorical
│   │   ├── compas.json             # COMPAS (race) - bổ sung của nhóm
│   │   ├── student.json            # Student (sex) - bổ sung của nhóm
│   │   ├── student_address.json    # Student (address) - bổ sung của nhóm
│   │   ├── cens_*.json             # Census configs (7 files) - bổ sung của nhóm
│   │   └── ...                     
│   └── three_moons/                
│
├── data/realworld/                 # Các tập dữ liệu thực tế
│   ├── bank-full.csv               # Bank Marketing Dataset
│   ├── communities.data            # Communities and Crime Dataset
│   ├── diabetic_data.csv           # Diabetes Readmission Dataset
│   ├── compas-scores-two-years.csv # COMPAS Recidivism Dataset
│   ├── student_performance.csv     # Student Performance Dataset
│   └── uci_census.csv              # UCI Census Income Dataset
│
├── results/                        # Kết quả thực nghiệm
│   ├── rw_experiment/              # Thực nghiệm Real-World (gom cụm công bằng)
│   ├── k_line_experiment/          # Thực nghiệm K-line (thay đổi số cụm k) của tác giả
│   ├── compas_experiment/          # Kết quả cho thực nghiệm K-line trên tập COMPAS
│   ├── adult_multi_exp/            # Kết quả cho thực nghiệm Multiple sensitive attribute trên tập Adult
│   ├── multi_attr/                 # Kết quả cho thực nghiệm Multiple sensitive attribute trên tập Census
│   ├── compas_hyperparam/          # Tối ưu tham số cho COMPAS
│   ├── student_hyperparam/         # Tối ưu tham số cho Student
│   ├── census_hyperparam/          # Tối ưu tham số cho Census (7 configs)
│   └──               
│
├── src/
│   ├── comparative_methods/        # Triển khai các phương pháp so sánh
│   ├── dc_dist/                    # Khoảng cách dc_distance
│   ├── evaluation/                 # Đánh giá: balance, dcsi, tỷ lệ noise
│   ├── experiments/                # Các thực nghiệm
│   │   ├── realworld_experiment.py # Thực nghiệm Real-World
│   │   ├── categorical_experiments.py # Thực nghiệm Categorical
│   │   ├── compas_experiment.py    # Thực nghiệm COMPAS (bổ sung)
│   │   ├── census_experiment.py    # Thực nghiệm Census (bổ sung)
│   │   ├── *_hyperparam_search.py  # Tối ưu siêu tham số (bổ sung)
│   │   └── ...
│   ├── utils/                      # DataLoader, DataEncoder
│   └── FairDen.py                  # Phương pháp FairDen
│
├── scripts/                        # Script tiện ích (bổ sung của nhóm)
│   ├── visualize_balance.py        # Trực quan hóa Balance
│   ├── visualize_kline.py          # Trực quan hóa K-line
│   └── visualize_census.py         # Trực quan hóa Census
│
├── visualization/                  # Thư mục lưu biểu đồ (bổ sung của nhóm)
│   ├── balance_comparison.png      # So sánh Balance Real-World
│   ├── kline_adult_comparison.png  # K-line cho Adult
│   ├── kline_compas_comparison.png # K-line cho COMPAS
│   └── ...
│ 
├── Report/                         # Báo cáo LaTeX (bổ sung của nhóm)
│   ├── content/                    # Nội dung các section
│   └── main.tex                    # File LaTeX chính
│
├── .gitignore                      
├── LICENSE                         
├── main.py                         # File chính để gọi các thực nghiệm  
├── README.md                         
└── requirements.txt                
```

## Trích dẫn
Nếu sử dụng phương pháp hoặc mã nguồn từ repository này, vui lòng trích dẫn bài báo của tác giả:

Lena Krieger*, Anna Beer*, Pernille Matthews, Anneka Myrup Thiesson, Ira Assent, (2025, April). FairDen: Fair Density-based Clustering. Accepted for publication at the *Thirteenth International Conference on Learning Representations (ICLR)*.

```bibtex
@unpublished{kriegerbeer2025fairden,
  title        =    {FairDen: Fair Density-based Clustering},
  author       =    {Krieger*, Lena and Beer*, Anna and Matthews, Pernille and Thiesson, Anneka Myrup and Assent, Ira},
  url          =    {https://openreview.net/forum?id=aPHHhnZktB},
  year         =    {2025},
  note         =    {Accepted for publication at The Thirteenth International Conference on Learning Representations,
                    (ICLR) 2025}
}
```

### Trích dẫn Dataset

**UCI Census Income Dataset:**
```bibtex
@misc{kohavi1996,
  author       = {Ron Kohavi},
  title        = {Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid},
  booktitle    = {Proceedings of KDD-96},
  year         = {1996},
  publisher    = {AAAI Press}
}
```

## Giấy phép

### Mã nguồn
Công trình này được cấp phép theo [Apache 2.0 License](LICENSE). Giấy phép này áp dụng cho tất cả các file mã nguồn do tác giả triển khai.

### Dữ liệu
Các tập dữ liệu sau được lấy từ [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/) và được cấp phép theo [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode):
- Tập Bank Marketing
- Tập Communities and Crime
- Tập Diabetes Readmission
- Tập UCI Census Income
- Tập COMPAS (ProPublica)
- Tập Student Performance

Dữ liệu do tác giả tạo bằng [DENSIRED](https://github.com/PhilJahn/DENSIRED) hoặc tập three moons được cấp phép theo [Creative Commons Zero (CC0)](https://creativecommons.org/public-domain/cc0/).



<!-- MARKDOWN LINKS & IMAGES -->

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
