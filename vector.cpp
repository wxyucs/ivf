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

#include <random>

#include "vector.h"
#include "distance.h"


std::vector<VectorPtr>
generate(uint64_t num, uint64_t dim) {
    std::vector<VectorPtr> ret;
    ret.resize(num);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

#pragma omp parallel for
    for (uint64_t i = 0; i < num; ++i) {
        auto vector = std::make_shared<Vector>(dim);
        for (uint64_t j = 0; j < dim; ++j) {
            vector->data()[j] = dis(gen);
        }
        ret[i] = vector;
    }
    return ret;
}


std::pair<uint64_t, float>
nearest(const VectorPtr &vector, const std::vector<VectorPtr> &lists, uint64_t dim) {

    float min_dist = std::numeric_limits<float>::max();
    uint64_t min_idx = 0;

    for (uint64_t idx = 0; idx < lists.size(); ++idx) {
        auto dist = distance(vector, lists[idx], dim);

        if (dist < min_dist) {
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = idx;
            }
        }
    }
    return std::make_pair(min_idx, min_dist);
}
