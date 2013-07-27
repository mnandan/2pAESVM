/*
 * svmSolver.h
 *
 *  Created on: Jul 24, 2013
 *      Author: mn
 */

#ifndef SVMSOLVER_H_
#define SVMSOLVER_H_

#include "fileInt.h"
#include <vector>

void svmSolver(std::vector<dataVect_T> &X, double *w, double C, UINT maxF);
void svmSolverRp(dataVect_T *X, double *w, double * B, double C, UINT totRpNum, UINT maxF);

#endif /* SVMSOLVER_H_ */
