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
from src.experiments.compas_hyperparam_search import compas_hyperparam_search
from src.experiments.student_hyperparam_search import student_hyperparam_search

# perform all experiments
def main():
    # UNCOMMENT TO RUN
    # adult_experiment() # Multiple sensitive attributes
    # categorical_experiments() # Categorical experiments
    # k_line_multi()  # K-Experiments
    # realworld_experiment() # Real-World experiments
    # compas_experiment()  # COMPAS experiments
    # student_experiment() # Student Performance experiments
    # compas_hyperparam_search()  # Search optimal DBSCAN params for COMPAS
    # student_hyperparam_search()  # Search optimal DBSCAN params for Student
    pass



if __name__ == "__main__":
    main()