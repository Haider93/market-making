#ifndef RL_STATE_H
#define RL_STATE_H

#include <vector>

#include "../environment/base.h"
#include <armadillo>
using namespace arma;

namespace rl {

class State
{
    private:
        // Pseudo-constant:
        long MEMORY_SIZE;
        int N_TILINGS;
        int N_ACTIONS;
        // ---

        std::vector<float> state_vars;
        std::vector<std::vector<int>> features;

        double potential;
		float normalized_mean_val;
		vec v_mvd;
		//contains updated means of state variables
		rowvec mean_vector;
		//contains covariances of state variables
		mat covar_mat;
		//accumulate values of state variables
		mat variable_values;
		

    public:
        State(long n_states, int n_actions, int n_tilings);
        State(Config &c);

        void initialise();

        void newState(environment::Base& env);
        void newState(std::vector<float>& vars, double potential = 0.0);

        void populateFeatures();
        std::vector<int>& getFeatures(int action = 0);

        double getPotential();

		float getNormalizedMeanVal();
		vector<float> getNormalizedVars();
		vec get_vars_for_mvd();

        std::vector<float>& toVector();
        void printState();
		//calculates mean of all state variables
		double normalize_mean(vector<float> vars);

		//normalize (-1 to 1) all state variables
		void normalized_vars(vector<float>& vars);

		//calculate mean of subsets of state variables
		vector<float> normalized_vars_subset(vector<float> vars);

		//normalize state variables and return arma vector
		vec vars_multi(vector<float> vars);

		//calculate expectation of state variable values vector
		double expectation_val(colvec v);

		//calsulate covariance between two variables
		double calc_cov(colvec a, colvec b);

		//update mean vector and covriances
		void update_mean_covariances(rowvec vars);

		//return mean vector
		rowvec get_mean_vector_state_values();

		//return covrainces
		mat get_covariances_state_values();

		//check if matrix is azero matrix
		bool isZero(mat m);

		
};

}

#endif
