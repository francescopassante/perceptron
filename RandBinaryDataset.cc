#include "RandBinaryDataset.h"
#include "RandBinary.h"
#include "Perceptron.h"
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>

RandBinaryDataset::RandBinaryDataset(const int& P, const int& L, const Perceptron& teacher) {
    P_ = P;
    L_ = L;

    for (int i = 0; i < P; i++) {
        data_.push_back(RandBinary(L));
    }
    trueLabels_ = calcTrueLabels(teacher);
}

RandBinaryDataset::RandBinaryDataset(const int& P, const int& L) {
    P_ = P;
    L_ = L;

    for (int i = 0; i < P; i++) {
        data_.push_back(RandBinary(L));
    }
    trueLabels_ = calcTrueLabels(Perceptron(L));
}

const vector<RandBinary>& RandBinaryDataset::getData() const {
    return data_;
}

int RandBinaryDataset::size() const {
    return P_;
}

void RandBinaryDataset::print() const {
    for(int i = 0; i < P_; i++) {
        std::cout << data_[i] << "\t" << trueLabels_[i] << std::endl;
    }
}



vector<int> RandBinaryDataset::calcTrueLabels(const Perceptron& teacher) {
    vector<int> labels;
    labels.reserve(P_);
    for (const RandBinary& i : data_) {
        double v = teacher.apply(i);
        labels.push_back(v < 0 ? -1 : 1);
    }
    return labels;
}

int RandBinaryDataset::countErrors(const vector<int>& predictions) const {
    int err = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        if (predictions[i] != trueLabels_[i]) {
            err++;
        }
    }
    return err;
}

const vector<int>& RandBinaryDataset::getLabels() const {
    return trueLabels_;
}

void RandBinaryDataset::shuffle() {
    vector<int> p;
    for(int i = 0; i < P_; i++) {
        p.push_back(i);
    }

    static std::mt19937 rng(std::random_device{}());
    std::shuffle(p.begin(), p.end(), rng);

    vector<RandBinary> new_data;
    new_data.reserve(P_);
    vector<int> new_labels;
    new_labels.reserve(P_);
    for(int i = 0; i < P_; i++) {
        new_data.push_back(data_[p[i]]);
        new_labels.push_back(trueLabels_[p[i]]);
    }

    data_ = new_data;
    trueLabels_ = new_labels;
}
