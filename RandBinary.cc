#include <vector>
#include <iostream>
#include <random>
#include "RandBinary.h"

using std::vector;
using std::ostream;


RandBinary::RandBinary(const int& L) {
    L_ = L;
    static std::mt19937 gen(std::random_device{}());
    static std::bernoulli_distribution dist(0.5);
    sequence_.reserve(L);
    for (int i = 0; i < L; i++) {
        sequence_.push_back(dist(gen) ? 1 : -1);
    }
}

RandBinary::RandBinary(const vector<int>& seq) {
    L_ = seq.size();
    sequence_ = seq;
}


const vector<int>& RandBinary::getSequence() const {
    return sequence_;
}


ostream& operator<<(ostream& os, const RandBinary& bin) {
    for(int i = 0; i < bin.L_; i++) {
        os << bin.sequence_[i];
    }
    return os;
}
