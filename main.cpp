#include <iostream>
#include "vector.h"
#include "distance.h"
#include "inverted_list.h"


int main() {
    std::cout << "Hello, World!" << std::endl;

    const uint64_t DIM = 2;

    auto v1 = generate(10, DIM);
    auto v2 = generate(100, DIM);
    auto v3 = generate(1, DIM);

    InvertedList iv(3, DIM);
    iv.train(v1);

    std::vector<Id> ids;
    ids.resize(100);
    for (uint64_t i = 0 ; i < 100; ++i) {
        ids[i] = i;
    }
    iv.add(ids, v2);

    auto res = iv.search(v3[0], 1, 1);

    std::cout << "id: " << res.first;
    std::cout << ", dist: " << res.second;
    std::cout << std::endl;

    return 0;
}