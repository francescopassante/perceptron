#ifndef RandBinary_h
#define RandBinary_h

#include <vector>

using std::vector;
using std::ostream;

class RandBinary {
public:
    RandBinary(const int& L);
    RandBinary(const vector<int>& seq);
    const vector<int>& getSequence() const;
    friend ostream& operator<<(ostream& os, const RandBinary& other);
private:
    int L_;
    vector<int> sequence_;
};
#endif
