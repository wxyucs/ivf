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
#pragma once

#include <string>
#include <vector>
#include <list>
#include <queue>
#include <deque>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>

#include "id.h"
#include "vector.h"


class Clustering {
public:
    Clustering(std::vector<VectorPtr> vectors,
               uint64_t nlist,
               uint64_t dim);

    void
    init();

    bool
    clustering();

    std::vector<VectorPtr> &
    centroids() {
        return centroids_;
    }

private:
    uint64_t dim_ = 0;
    uint64_t nlist_ = 0;
    std::vector<VectorPtr> vectors_;

    std::vector<VectorPtr> centroids_;
    std::vector<VectorList> vector_lists_;
};
