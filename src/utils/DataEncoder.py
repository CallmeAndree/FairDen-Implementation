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

import os

import numpy as np
import category_encoders as ce

from mpire import WorkerPool
from tqdm.auto import tqdm


def ordinal_encode(data):
    """
        Parameters:
            data (np.array): data to be encoded.

        Returns:
            ordinal encoded (array): Returns the data in ordinal encoding.

    """
    data = data.copy()
    encoder = ce.OrdinalEncoder(data)
    data = encoder.fit_transform(data)
    return data


def Frequency_table(data):
    """
        Parameters:
            data (np.array): data to be encoded.

        Returns:
            Frequency_table (array): Returns the frequency table for the categories and possibilities.

    """
    # dictionary category : number of possibilities
    cat_dict = {str(i): len(np.unique(data[category])) for
                i, category in enumerate(data.columns.tolist())}
    s = data.shape[1]
    # for each category determine frequency
    with WorkerPool(n_jobs=os.cpu_count() - 2, shared_objects=(data, max(cat_dict.values()))) as pool:
        result = pool.map(counting, range(s), progress_bar=True)
    print('Counting Done')
    del cat_dict
    return np.array(result) / data.shape[0]


def counting(shared, i):
    data, maximum_cat = shared
    unique, counts = np.unique(data.iloc[:, i], return_counts=True)
    counts = np.pad(counts, (0, maximum_cat - len(counts)), 'constant')
    return np.expand_dims(counts, axis=0)


def map_goodall(shared, i):
    data, cats = shared
    scores = []
    r = data.shape[0]
    d = data.shape[1]
    sample_i = np.array(data.iloc[i, :])
    for j in range(i + 1, r):
        sum = 0
        # indices of the same features
        for index in np.equal(sample_i, np.array(data.iloc[j, :])).nonzero()[0]:
            logic = np.flatnonzero(cats[index, :] < cats[index][sample_i[index] - 1])
            for l in logic:
                sum = sum + ((cats[index, l] * (cats[index, l] - 1)) / (r * (r - 1))) ** 2
        sum = 1 - sum * (1 / d)
        scores.append(sum)
    del cats
    return scores


def Goodall1(data):
    """
        Parameters:
            data (np.array): data to be encoded.

        Returns:
            Goodall_encoding (array): Returns the goodall1 (https://epubs.siam.org/doi/pdf/10.1137/1.9781611972788.22) encoding.

    """
    # ordinal encoding
    data = ordinal_encode(data)
    r = data.shape[0]
    # frequency table
    freq_table = Frequency_table(data)
    # for each datapoint calculate goodall1 similarity to all others (triangular matrix)
    with WorkerPool(n_jobs=os.cpu_count() - 2, shared_objects=(data, freq_table)) as pool:
        result = pool.map(map_goodall, range(r), progress_bar=True)
    del freq_table
    print('Mapping Goodall done')
    good1 = []
    for i, res in enumerate(tqdm(result)):
        good1.append([0.0] * (i + 1) + res)
    del result
    # combine triangular matrices
    return np.add(np.array(good1), np.transpose(good1))
