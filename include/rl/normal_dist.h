#pragma once
#include <armadillo>
using namespace arma;
namespace rl{
class Distribution {
private:
	double mean;
	double deviation;
	int observations;
	double weight;
	uword dimension;
	vec means;
	mat covariance;
public:
	Distribution(){}
	Distribution(double m, double sd, int n_observation, double w):mean(m), deviation(sd), observations(n_observation),
	weight(w){}
	Distribution(uword dimen, vec m, mat c, int n_observation, double w) :dimension(dimen), means(m), covariance(c),
		observations(n_observation), weight(w) {}
	~Distribution();
	double getMean();
	double getDeviation();
	double pdf(double x);
	void setMean(double m);
	void setDeviation(double sd);
	void setObservation(int x);
	int getObservation();
	void setWeight(double w);
	double getWeight();
	//multivariate
	double pdf_multi(vec X);
	vec getMeanVec();
	mat getCovMat();
	void setMeanVec(vec m);
	void setCovMat(mat c);
	uword getDimension();

};
}