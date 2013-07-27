/*
 * getRP.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: mn
 */

#include "onePdRS.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int tempInd;
#define MIN(x,y) ((x<y)?x:y)
#define MAX(x,y) ((x>y)?x:y)
#define SWAP_I(i,j) {tempInd = index[i]; index[i] = index[j]; index[j] = tempInd;}
#define INF HUGE_VAL
#define TAU 1e-12
#define LAMBDA_MAX 1.0
#define LAMBDA_MIN 0


class DRS_Solver {
public:
    DRS_Solver() {
        Cp = 1;
        eps = 0.001;
    }
    virtual ~DRS_Solver() {};

protected:
    double *G;		// gradient of objective function
    enum { LOWER_BOUND, UPPER_BOUND, FREE };
    char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
    double *alpha;
    double eps;
    double Cp;
    double *p;
    double ** rpCache;
    int l;


    void update_alpha_status(int i) {
        if(alpha[i] >= LAMBDA_MAX)
            alpha_status[i] = UPPER_BOUND;
        else if(alpha[i] <= LAMBDA_MIN)
            alpha_status[i] = LOWER_BOUND;
        else alpha_status[i] = FREE;
    }

    bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
    bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
    bool is_free(int i) { return alpha_status[i] == FREE; }
};


class Solver_DERIVE_AE : public DRS_Solver {
public:
    Solver_DERIVE_AE() {Cp = 1; eps = 0.001;}
    void Solve(int l, double *xTz, double *alpha, double ** rpCache);

private:
    int select_working_set(int &i, int &j);
};

// return 1 if already optimal, return 0 otherwise
int Solver_DERIVE_AE::select_working_set(int &out_i, int &out_j) {
    double Gmax = -INF;
    double Gmax2 = -INF;
    int Gmax_idx = -1;
    int Gmin_idx = -1;
    double obj_diff_min = INF;

    for(int t=0;t<l;t++)
        if(!is_upper_bound(t))
            if(-G[t] >= Gmax) {
        Gmax = -G[t];
        Gmax_idx = t;
            }

    int i = Gmax_idx;

    for(int j=0;j<l;j++) {
        if (!is_lower_bound(j)) {
            double grad_diff=Gmax+G[j];
            if (G[j] >= Gmax2)
                Gmax2 = G[j];
            if (grad_diff > 0) {
                double obj_diff;
                double quad_coef = rpCache[i][i] + rpCache[j][j] - 2*rpCache[MIN(i, j)][MAX(i, j)];
                obj_diff = -(grad_diff*grad_diff)/quad_coef;
                if (obj_diff <= obj_diff_min) {
                    Gmin_idx=j;
                    obj_diff_min = obj_diff;
                }
            }
        }
    }

    if(Gmax+Gmax2 < eps)
        return 1;

    out_i = Gmax_idx;
    out_j = Gmin_idx;
    return 0;
}

void Solver_DERIVE_AE::Solve(int l, double *xTz, double *alpha, double ** rpCache) {
    this->l = l;
    this->Cp = 1.0;
    this->eps = 0.001;
    this->p = xTz;
    this->alpha = alpha;
    this->rpCache = rpCache;
    // initialize alpha_status
    {
        double alphaInitVal = (double)1.0/l;
        alpha_status = new char[l];
        for(int i=0;i<l;i++) {
            alpha[i] = alphaInitVal;
            update_alpha_status(i);
        }
    }

    // initialize active set (for shrinking)
    // initialize gradient
    {
        G = new double[l];
        for(int i=0;i<l;i++) {
            G[i] = -p[i];
        }
        for(int i=0;i<l;i++) {
            if(!is_lower_bound(i)) {
                for(int j=0;j<l;j++)
                    G[j] += alpha[i]*rpCache[MIN(i, j)][MAX(i, j)];
            }
        }
    }
    // optimization step
    int iter = 0;
    while(iter < 1000) {
        int i, j;
        if(select_working_set(i, j)!=0)
            break;
        ++iter;
        // update alpha[i] and alpha[j], handle bounds carefully
        double old_alpha_i = alpha[i];
        double old_alpha_j = alpha[j];
        double quad_coef = rpCache[i][i] + rpCache[j][j] - 2*rpCache[MIN(i, j)][MAX(i, j)];
        if (quad_coef <= 0)
            quad_coef = TAU;
        double delta = (G[i]-G[j])/quad_coef;
        double sum = alpha[i] + alpha[j];
        alpha[i] -= delta;
        alpha[j] += delta;
        if(sum > LAMBDA_MAX)  {
            if(alpha[i] > LAMBDA_MAX) {
                alpha[i] = LAMBDA_MAX;
                alpha[j] = sum - LAMBDA_MAX;
            }
        }
        else {
            if(alpha[j] < LAMBDA_MIN) {
                alpha[j] = LAMBDA_MIN;
                alpha[i] = sum - LAMBDA_MIN;
            }
        }
        if(sum > LAMBDA_MAX) {
            if(alpha[j] > LAMBDA_MAX) {
                alpha[j] = LAMBDA_MAX;
                alpha[i] = sum - LAMBDA_MAX;
            }
        }
        else {
            if(alpha[i] < LAMBDA_MIN) {
                alpha[i] = LAMBDA_MIN;
                alpha[j] = sum - LAMBDA_MIN;
            }
        }
        // update G
        double delta_alpha_i = alpha[i] - old_alpha_i;
        double delta_alpha_j = alpha[j] - old_alpha_j;
        for(int k=0;k<l;k++)
            G[k] += (rpCache[MIN(i, k)][MAX(i, k)]*delta_alpha_i + rpCache[MIN(j, k)][MAX(j, k)]*delta_alpha_j);
        update_alpha_status(i);
        update_alpha_status(j);
    }

    delete[] alpha_status;
    delete[] G;
}

void updateLambda(int rpSize, double ** rpCache, double *lambdaArr, double *xTz) {
    Solver_DERIVE_AE s;
    s.Solve(rpSize, xTz, lambdaArr, rpCache);
}

double getRepErr(int rpSize, double ** rpCache, double *lambdaArr, double *xTz, double xNorm) {
    Solver_DERIVE_AE s;
    s.Solve(rpSize, xTz, lambdaArr, rpCache);

    double v1 = 0, v2 = 0, err;
	for(int i=0; i<rpSize; i++) {
		for(int j=0; j<rpSize; j++)
			v1 += lambdaArr[i]*lambdaArr[j]*rpCache[MIN(i,j)][MAX(i,j)];
		v2 += lambdaArr[i]*xTz[i];
	}
	err = xNorm + v1 - 2*v2;
	return err;
}


