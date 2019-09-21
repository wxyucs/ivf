// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "inverted_list.h"
#include "clustering.h"


void
InvertedList::train(const std::vector<VectorPtr> &vectors) {
    Clustering clustering(vectors, nlist_, dim_);
    clustering.init();

    while (clustering.clustering());
    centroids_ = clustering.centroids();

    id_lists_.resize(nlist_);
    lists_.resize(nlist_);
}

void
InvertedList::add(const std::vector<Id> &ids, const std::vector<VectorPtr> &vectors) {
    std::mutex mutex;

#pragma omp parallel for
    for (uint64_t i = 0; i < vectors.size(); ++i) {
        auto &vector = vectors[i];

        auto pair = nearest(vector, centroids_, dim_);
        auto min_idx = pair.first;

        {
            std::lock_guard<std::mutex> lock(mutex);
            id_lists_[min_idx].push_back(ids[i]);
            lists_[min_idx].push_back(vector);
        }
    }
}

std::pair<Id, float>
InvertedList::search(const VectorPtr &vector, uint64_t nprobe, uint64_t k) {
    auto idx = nearest(vector, centroids_, dim_).first;

    return nearest(vector, lists_[idx], dim_);
}
