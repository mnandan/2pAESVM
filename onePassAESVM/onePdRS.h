/*
 * onePdRS.h
 *
 *  Created on: Jul 24, 2013
 *      Author: mn
 */

#ifndef ONEPDRS_H_
#define ONEPDRS_H_

#include "fileInt.h"
#include <vector>

typedef struct
{
    double dist;
    double repErr;
    UINT ind;
}distDatType;

void onePassDRS(std::vector<dataVect_T> &X, double *w, int runNum, double numRp, std::ofstream & outRp, UINT &totBatchRpNum);
double getRepErr(int rpSize, double ** rpCache, double *lambda, double *xTz, double xNorm);
void updateLambda(int rpSize, double ** rpCache, double *lambda, double *xTz);

#endif /* ONEPDRS_H_ */
