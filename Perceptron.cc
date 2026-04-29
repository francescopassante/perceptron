#include "Perceptron.h"
#include "RandBinary.h"
#include "RandBinaryDataset.h"
#include <random>
#include <vector>
#include <string>
#include <cmath>

using std::vector;
using std::string;


double randn(double mean=0.0, double stddev=1.0) {
    static std::mt19937 gen(std::random_device{}());
    std::normal_distribution<> dist(mean, stddev);
    return dist(gen);
}



Perceptron::Perceptron(const int& L) {
    for(int i = 0; i < L; i++) {
        weights_.push_back(randn());
    }
    L_ = L;
}


Perceptron::Perceptron(const vector<double>& weights) {
    weights_ = weights;
    L_ = weights.size();
}

vector<double> Perceptron::getWeights() const {
    return weights_;
}



double Perceptron::apply(const RandBinary& bin) const {
    double res = 0;
    for(int i = 0; i < L_; i++) {
        res += bin.getSequence()[i]*weights_[i];
    }
    return res;
}

double Perceptron::calcCost(const RandBinaryDataset& ds) const {
    const vector<RandBinary>& data = ds.getData();
    const vector<int>& labels = ds.getLabels();
    double cost = 0;
    for(int mu = 0; mu < ds.size(); mu++) {
        cost += 0.5*pow(apply(data[mu]) - labels[mu], 2);
    }
    return cost/ds.size();
}


int Perceptron::testOnDataset(const RandBinaryDataset& ds) const {
    vector<int> predictions;
    predictions.reserve(ds.size());
    for(const RandBinary& i: ds.getData()) {
        predictions.push_back(apply(i) < 0 ? -1 : 1);
    }
    return ds.countErrors(predictions);
}


Perceptron& Perceptron::trainOnDataset(RandBinaryDataset& ds, const string& rule, double variance) {
    int N = L_;
    int P = ds.size();

    int errors = -1;
    double prediction;
    const vector<RandBinary>& data = ds.getData();
    const vector<int>& labels = ds.getLabels();

    if (rule == "Hebb") {

        for (int mu = 0; mu < P; mu++) {
            for(int i = 0; i < N; i++) {
                weights_[i] = weights_[i] + ((double)1/((double)sqrt(N))) * labels[mu] * data[mu].getSequence()[i];
            }
        }
    }


    if (rule == "Perceptron") {
        while (errors != 0) {
            for(int mu = 0; mu < P; mu++) {
                int prediction = apply(data[mu]) < 0 ? -1 : 1;
                if (prediction != labels[mu]) {
                    for(int i = 0; i < N; i++) {
                        weights_[i] = weights_[i] + ((double)1/((double)sqrt(N))) * labels[mu] * data[mu].getSequence()[i];
                    }
                }
            }
            errors = testOnDataset(ds);
            ds.shuffle();
        }
    }

    if (rule == "RandomPerceptron") {
        while (errors != 0) {
            for(int mu = 0; mu < P; mu++) {
                int prediction = apply(data[mu]) < 0 ? -1 : 1;
                if (prediction != labels[mu]) {
                    for(int i = 0; i < N; i++) {
                        weights_[i] = weights_[i] + ((double)1/((double)sqrt(N))) * labels[mu] * data[mu].getSequence()[i] * (1+randn(0, sqrt(variance)));
                    }
                }
            }
            errors = testOnDataset(ds);
            ds.shuffle();
        }

    }

    if (rule == "Adaline") {
        // Initialize J as sum of all the examples
        for(int i = 0; i < N; i++) {
            weights_[i] = 0;
        }

        for(int mu = 0; mu < P; mu++) {
            for(int i = 0; i < N; i++) {
                weights_[i] = weights_[i] + labels[mu]*data[mu].getSequence()[i];
            }
        }

        // Normalize J
        double norm = 0;
        for(double w: weights_) {
            norm += pow(w,2);
        }

        for(int i = 0; i < N; i++) {
            weights_[i] = weights_[i]/sqrt(norm);
        }


        // Apply correction
        double gamma = 0.05;
        double newCost = calcCost(ds);
        double oldCost = 0;
        while (fabs(newCost - oldCost) > pow(10,-8)) {
            vector<double> delta(N,0.0);
            for(int mu = 0; mu < P; mu++) {
                double pred = apply(data[mu]);
                for(int i = 0; i < N; i++) {
                    delta[i] += -gamma * ((double)1/P) * data[mu].getSequence()[i]
                                      * (pred - labels[mu]);
                }
            }
            for(int i = 0; i < N; i++) {
                weights_[i] += delta[i];
            }
            oldCost = newCost;
            newCost = calcCost(ds);
        }
    }

    return *this;
}



Perceptron& Perceptron::trainOnDataset(const int& P, const string& rule, double variance) {
    RandBinaryDataset ds(P, L_);
    return trainOnDataset(ds, rule, variance);
}
