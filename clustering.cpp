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

#include "clustering.h"
#include "distance.h"


Clustering::Clustering(std::vector<VectorPtr> vectors,
                       uint64_t nlist,
                       uint64_t dim)
    : dim_(dim),
      nlist_(nlist),
      vectors_(std::move(vectors)) {}

void
Clustering::init() {
    centroids_ = std::move(generate(nlist_, dim_));
}

bool
Clustering::clustering() {
    vector_lists_.clear();
    vector_lists_.resize(nlist_);

    std::mutex mutex;
#pragma omp parallel for
    for (uint64_t i = 0; i < vectors_.size(); ++i) {
        auto &vector = vectors_[i];

        auto pair = nearest(vector, centroids_, dim_);
        auto min_idx = pair.first;

        {
            std::lock_guard<std::mutex> lock(mutex);
            vector_lists_[min_idx].push_back(vector);
        }
    }

    bool change = false;
#pragma omp parallel for
    for (uint64_t i = 0; i < vector_lists_.size(); ++i) {
        auto vector_list = vector_lists_[i];
        auto centroid = std::make_shared<Vector>(dim_);

        for (uint64_t d = 0; d < dim_; ++d) {
            float sum = 0;
            for (auto &vector : vector_list) {
                sum += vector->data()[d];
            }
            centroid->data()[d] = sum / vector_list.size();
        }

        if (distance(centroid, centroids_[i], dim_) > 0.1) {
            change = true;
        }
        centroids_[i] = centroid;
    }

    return change;
}
