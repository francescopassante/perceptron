#ifndef Perceptron_h
#define Perceptron_h

#include <vector>
#include <string>
#include "RandBinary.h"
// #include "RandBinaryDataset.h"
class RandBinaryDataset;

using std::vector;
using std::string;



class Perceptron {
private:
    vector<double> weights_;
    int L_;

public:
    Perceptron(const int& L);
    Perceptron(const vector<double>& weights);
    double apply(const RandBinary& bin) const;
    int testOnDataset(const RandBinaryDataset& ds) const;
    vector<double> getWeights() const;
    Perceptron& trainOnDataset(RandBinaryDataset& ds, const string& rule, double variance = 50.0);
    Perceptron& trainOnDataset(const int& P, const string& rule, double variance = 50.0);
    double calcCost(const RandBinaryDataset& ds) const;
};
#endif
