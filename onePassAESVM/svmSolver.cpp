/*
 * svmSolver.cpp

 *
 *  Created on: Jul 24, 2013
 *      Author: mn
 */
#include <iostream>
#include <stdlib.h>
#include <math.h>
#include "svmSolver.h"
#include <algorithm>

using namespace std;

#define INF HUGE_VAL


static inline double getNorm(const feat_T *F, UINT numFeats) {
    double dotProduct = 0;
	for(UINT fI = 0; fI < numFeats; fI++) { 	//feature index
		double fVal = F[fI].fVal;
		dotProduct += fVal*fVal;
	}
    return dotProduct;
}

void svmSolver(vector<dataVect_T> &X, double *w,
		double C, UINT maxF) {

	UINT numVects = X.size();
	int maxIterNum = 1000;

	// PG: projected gradient, for shrinking and stopping
	unsigned active_size = numVects;
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	double *alpha = new double[numVects];
	UINT *index = new UINT[numVects];
	double *vectNorm = new double[numVects];
	for(UINT vI=0; vI<numVects; vI++) {
		alpha[vI] = 0;
		index[vI] = vI;
		vectNorm[vI] = getNorm(X[vI].F, X[vI].numFeats);
	}

	for(UINT fI = 0; fI < maxF; fI++)
		w[fI] = 0;

	int iterNum = 0;
	while (iterNum < maxIterNum) {
		PGmax_new = -INF;
		PGmin_new = INF;

		for (UINT i=0; i<active_size; i++) {
			UINT j = i + rand()%(active_size - i);
			swap(index[i], index[j]);
		}

		for (UINT s=0; s<active_size; s++)
		{
			UINT vI = index[s];	// vector index

			double Gradient = 0;
			feat_T *F = X[vI].F;
			int y = (int)X[vI].label;
			for(UINT fI = 0; fI < X[vI].numFeats; fI++) 	//feature index
				Gradient += w[F[fI].fNum]*F[fI].fVal;
			Gradient = Gradient*y - 1;

			PG = 0;
			if (alpha[vI] == 0) {
				if (Gradient > PGmax_old) {
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (Gradient < 0)
					PG = Gradient;
			}
			else if (alpha[vI] == C) {
				if (Gradient < PGmin_old) {
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (Gradient > 0)
					PG = Gradient;
			}
			else
				PG = Gradient;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12) {
				double alpha_old = alpha[vI];
				alpha[vI] = min(max(alpha[vI] - Gradient/vectNorm[vI], 0.0), C);		// box constraints

				double delta = (alpha[vI] - alpha_old)*y;
				for(UINT fI = 0; fI < X[vI].numFeats; fI++) 	//feature index
					w[F[fI].fNum] += delta*F[fI].fVal;
			}
		}

		iterNum++;
		if(iterNum % 10 == 0)
			cout<< ".";

		if(PGmax_new - PGmin_new <= 0.1) {
			if(active_size == numVects)
				break;
			else {
				active_size = numVects;
				cout << "*";
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	delete [] vectNorm;
	delete [] alpha;
	delete [] index;
}


void svmSolverRp(dataVect_T* X, double *w,
		 double *B, double C, UINT numVects, UINT maxF) {

	int maxIterNum = 1000;

	// PG: projected gradient, for shrinking and stopping
	unsigned active_size = numVects;
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	double *alpha = new double[numVects];
	UINT *index = new UINT[numVects];
	double *vectNorm = new double[numVects];
	for(UINT vI=0; vI<numVects; vI++) {
		alpha[vI] = 0;
		index[vI] = vI;
		vectNorm[vI] = getNorm(X[vI].F, X[vI].numFeats);
	}

	for(UINT fI = 0; fI < maxF; fI++)
		w[fI] = 0;

	int iterNum = 0;
	while (iterNum < maxIterNum) {
		PGmax_new = -INF;
		PGmin_new = INF;

		for (UINT i=0; i<active_size; i++) {
			UINT j = i + rand()%(active_size - i);
			swap(index[i], index[j]);
		}

		for (UINT s=0; s<active_size; s++)
		{
			UINT vI = index[s];	// vector index

			double Gradient = 0;
			double Cnow = C*B[vI];
			feat_T *F = X[vI].F;
			int y = (int)X[vI].label;
			for(UINT fI = 0; fI < X[vI].numFeats; fI++) 	//feature index
				Gradient += w[F[fI].fNum]*F[fI].fVal;
			Gradient = Gradient*y - 1;

			PG = 0;
			if (alpha[vI] == 0) {
				if (Gradient > PGmax_old) {
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (Gradient < 0)
					PG = Gradient;
			}
			else if (alpha[vI] == Cnow) {
				if (Gradient < PGmin_old) {
					active_size--;
					swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (Gradient > 0)
					PG = Gradient;
			}
			else
				PG = Gradient;

			PGmax_new = max(PGmax_new, PG);
			PGmin_new = min(PGmin_new, PG);

			if(fabs(PG) > 1.0e-12) {
				double alpha_old = alpha[vI];
				alpha[vI] = min(max(alpha[vI] - Gradient/vectNorm[vI], 0.0), Cnow);		// box constraints

				double delta = (alpha[vI] - alpha_old)*y;
				for(UINT fI = 0; fI < X[vI].numFeats; fI++) 	//feature index
					w[F[fI].fNum] += delta*F[fI].fVal;
			}
		}

		iterNum++;
		if(iterNum % 10 == 0)
			cout<< ".";

		if(PGmax_new - PGmin_new <= 0.1) {
			if(active_size == numVects)
				break;
			else {
				active_size = numVects;
				cout << "*";
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	delete [] vectNorm;
	delete [] alpha;
	delete [] index;
}
