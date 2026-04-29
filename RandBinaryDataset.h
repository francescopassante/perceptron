#ifndef RandBinaryDataset_h
#define RandBinaryDataset_h

#include <vector>
#include <string>
#include "Perceptron.h"
#include "RandBinary.h"

using std::vector;

class RandBinaryDataset {
private:
    int L_;
    int P_;
    vector<RandBinary> data_;
    vector<int> trueLabels_;

public:
    RandBinaryDataset(const int& P, const int& L, const Perceptron& teacher);
    RandBinaryDataset(const int& P, const int& L);
    const vector<RandBinary>& getData() const;
    const vector<int>& getLabels() const;
    vector<int> calcTrueLabels(const Perceptron& teacher);
    int countErrors(const vector<int>& predictions) const;
    void shuffle();
    void print() const;
    int size() const;
};
#endif
