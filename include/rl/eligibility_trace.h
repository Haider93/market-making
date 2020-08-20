#ifndef ELIGIBILITY_TRACE_H
#define ELIGIBILITY_TRACE_H

#include "../rl/state.h"

#include <iterator>

namespace rl {

#define MAX_NONZERO_TRACES 10

class EligibilityTrace
{
    private:
        const int N_ACTIONS;

        float tolerance;

        int* nonzero_traces_inverse;

    public:
        float* eligibility;

        int n_nonzero_traces;
        int nonzero_traces[MAX_NONZERO_TRACES];

    public:
		EligibilityTrace(int n_actions);
        ~EligibilityTrace();

        void decay(float rate);
        void update(State& state, int action);

        int* begin();
        int* end();

        float get(int action);
        void set(int action, float value);

        void clear(int action);
        void clearExisting(int action, int loc);

        void increaseTolerance();
};

}

#endif
