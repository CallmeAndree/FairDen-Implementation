# Copyright 2025 Forschungszentrum Juelich GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File author: Lena Krieger

from src.experiments.categorical_experiments import categorical_experiments
from src.experiments.k_line_experiment import k_line_multi
from src.experiments.realworld_experiment import realworld_experiment
from src.experiments.adult_experiment import adult_experiment
from src.experiments.compas_experiment import compas_experiment
from src.experiments.student_experiment import student_experiment

# perform all experiments
def main():
    # Multiple sensitive attributes
    # adult_experiment()  # DONE - completed adult_gmr + adult_g, adult_m, adult_r, adult_gm, adult_gr, adult_mr
    # Categorical experiments
    # categorical_experiments()  # DONE - completed bank, adult2, adult5 + bank3, adult, adult4
    # K-Experiments
    # k_line_multi()  # DONE - completed adult2 + adult5
    # Real-World experiments
    # realworld_experiment()  # INCOMPLETE - adult4 done, diabetes interrupted (Fairlet MCF), communities not started
    # COMPAS experiments
    compas_experiment()  # DONE
    # Student Performance experiments
    # student_experiment()



if __name__ == "__main__":
    main()