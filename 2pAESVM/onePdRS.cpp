/*
 * onePdRS.cpp
 *
 *  Created on: Jul 24, 2013
 *      Author: mn
 */
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include "onePdRS.h"
#include <algorithm>
#include <vector>
#include "fileInt.h"
#include <fstream>

using namespace std;

#define INF HUGE_VAL
#define M_VAL (UINT)10

static void segregateDataFLS2(UINT anchorInd, UINT startIndex, UINT endIndex,
		vector<dataVect_T> &X, dataVect_T* X2, UINT *cint, UINT subSize,
		distDatType *distVals);
static void onePassDRSsub(vector<dataVect_T> &X, UINT startInd, UINT endInd,
		UINT subSize, ofstream & outRp, UINT &totBatchRpNum);
static distDatType quickSelectDist(distDatType *arr, UINT n, UINT k);

static inline double getNorm(const feat_T *F, UINT numFeats) {
	double dotProduct = 0;
	for (UINT fI = 0; fI < numFeats; fI++) { 	//feature index
		double fVal = F[fI].fVal;
		dotProduct += fVal * fVal;
	}
	return dotProduct;
}

static inline double getDotProduct(dataVect_T x1, dataVect_T x2) {
	double dotProduct = 0;
	UINT fn1 = 0, fn2 = 0;
	while (fn1 < x1.numFeats && fn2 < x2.numFeats) {
		if (x1.F[fn1].fNum == x2.F[fn2].fNum) {
			dotProduct += x1.F[fn1].fVal * x2.F[fn2].fVal;
			++fn2;
			++fn1;
		} else {
			if (x1.F[fn1].fNum > x2.F[fn2].fNum)
				++fn2;
			else
				++fn1;
		}
	}
	return dotProduct;
}

static inline void updateCache(double ** rpCache, vector<dataVect_T> &X,
		UINT startIndex, UINT indToAdd, distDatType * distVals) {
	rpCache[indToAdd][indToAdd] = distVals[indToAdd].dist;
	for (UINT ind = indToAdd; ind > 0; ind--)
		rpCache[ind - 1][indToAdd] = getDotProduct(X[indToAdd + startIndex],
				X[ind - 1 + startIndex]);
}

void onePassDRS(vector<dataVect_T> &X, double *w, int runNum, double numRp,
		ofstream & outRp, UINT &totBatchRpNum) {

	double uBoundErr = 1.2;
	double lBoundErr = -0.1;
	UINT numVects = (UINT) X.size();
	if (runNum > 1) {
		uBoundErr = 1.1;
		lBoundErr = -0.1;
	}

	UINT currSwapIndPos = 0;
	UINT currSwapIndNeg = numVects - 1;
	UINT vI = 0;

	while (vI <= currSwapIndNeg) {
		int label = (int) X[vI].label;
		double predErr = 0;
		for (UINT fI = 0; fI < X[vI].numFeats; fI++)
			predErr += w[X[vI].F[fI].fNum] * X[vI].F[fI].fVal;
		predErr *= label;
		if (predErr <= uBoundErr && predErr >= lBoundErr) {
			if (label > 0) {
				swap(X[vI], X[currSwapIndPos]);
				currSwapIndPos++;
			} else {
				swap(X[vI], X[currSwapIndNeg]);
				currSwapIndNeg--;
				vI--;	//check vector at vI again
			}
		}
		vI++;
	}
	currSwapIndNeg++;

	if (numVects - currSwapIndNeg == 0) {
		fprintf(stderr, "No negative vectors in block! Cannot continue\n");
		exit(1);
	} else if (currSwapIndPos == 0) {
		fprintf(stderr, "No positive vectors in block! Cannot continue\n");
		exit(1);
	}
	UINT numPvects = currSwapIndPos;
	UINT numNvects = numVects - currSwapIndNeg;
	double numSubs = numRp / (double) M_VAL;
	double numPsubs = numPvects * numSubs / (numPvects + numNvects);
	numPsubs = pow(2, floor(log(numPsubs) / log(2)));
	UINT subSize = max(M_VAL, (UINT) ceil(numPvects / numPsubs));
	//cout << currSwapIndPos << "\t" << numPsubs << "\t" << subSize << "\t" << numVects << endl;
	onePassDRSsub(X, 0, currSwapIndPos, subSize, outRp, totBatchRpNum);

	double numNsubs = numNvects * numSubs / (numPvects + numNvects);
	numNsubs = pow(2, floor(log(numNsubs) / log(2)));
	subSize = max(M_VAL, (UINT) ceil(numNvects / numNsubs));
	//cout << numNvects << "\t" << numNsubs << "\t" << subSize << "\t" << numVects << endl;
	onePassDRSsub(X, currSwapIndNeg, numVects, subSize, outRp, totBatchRpNum);
	return;
}

static void onePassDRSsub(vector<dataVect_T> &X, UINT startInd, UINT endInd,
		UINT subSize, ofstream & outRp, UINT & totBatchRpNum) {

	UINT numVects = endInd - startInd;
	dataVect_T * X2 = new dataVect_T[numVects];
	distDatType *distVals = new distDatType[numVects];
	UINT * clustEndInd = new UINT[endInd];
	double ** lambdaArr = new double*[subSize];		// size VxM
	double ** rpCache = new double*[M_VAL ];				// size MxM
	for (UINT ind = 0; ind < M_VAL ; ind++) {
		lambdaArr[ind] = new double[M_VAL ];
		rpCache[ind] = new double[M_VAL ];
	}
	for (UINT ind = M_VAL; ind < subSize; ind++)
		lambdaArr[ind] = new double[M_VAL ];
	double *xTz = new double[M_VAL ];

	segregateDataFLS2(startInd, startInd, endInd, X, X2, clustEndInd, subSize,
			distVals);

	UINT prevCSI = 0;
	UINT prevCS = 0;
	UINT clustStartIndex = startInd;
	UINT clustEndIndex;
	while (clustStartIndex < endInd) {
		clustEndIndex = clustEndInd[clustStartIndex];
		UINT clustSize = clustEndIndex - clustStartIndex;
		//printf("clustSize = %i\t, clustStartIndex = %i, clustEndIndex = %i, startIndex = %i, endIndex = %i\n", clustSize, clustStartIndex, clustEndIndex, startIndex, endIndex);
		for (UINT ind = 0; ind < M_VAL ; ind++) {
			for (UINT rpInd = 0; rpInd < M_VAL ; rpInd++)
				lambdaArr[ind][rpInd] = 0;
			lambdaArr[ind][ind] = 1;
		}

		if (clustSize > M_VAL) {
			UINT rpInd = 0;
			// find max norm vect: x1
			UINT maxNormInd = 0;
			double maxDistVal = -INF;
			for (UINT vI = clustStartIndex; vI < clustEndIndex; vI++) {
				double curr_dist = getNorm(X[vI].F, X[vI].numFeats);
				distVals[vI - clustStartIndex].dist = curr_dist;

				if (prevCS != 0) {
					for (UINT rpInd2 = 0; rpInd2 < prevCS; rpInd2++)
						xTz[rpInd2] = getDotProduct(X[vI], X[rpInd2 + prevCSI]);

					distVals[vI - clustStartIndex].repErr = getRepErr(prevCS,
							rpCache, lambdaArr[subSize - 1], xTz, curr_dist);
					if (curr_dist + distVals[vI - clustStartIndex].repErr
							> maxDistVal) {
						maxNormInd = vI;
						maxDistVal = curr_dist;
					}
				} else {
					distVals[vI - clustStartIndex].repErr = 0;
					if (curr_dist > maxDistVal) {
						maxNormInd = vI;
						maxDistVal = curr_dist;
					}
				}
			}
			swap(X[clustStartIndex + rpInd], X[maxNormInd]);
			swap(distVals[rpInd], distVals[maxNormInd - clustStartIndex]);
			rpInd++;

			// find max norm from x1: x2
			UINT maxNormInd2 = 0;
			double maxDistVal2 = -INF;
			for (UINT ind = clustStartIndex + rpInd; ind < clustEndIndex;
					ind++) {
				double curr_dist = distVals[ind - clustStartIndex].dist
						+ maxDistVal
						- 2 * getDotProduct(X[clustStartIndex], X[ind]);
				if (curr_dist + distVals[ind - clustStartIndex].repErr
						> maxDistVal2) {
					maxNormInd2 = ind;
					maxDistVal2 = curr_dist;
				}
			}
			swap(X[clustStartIndex + rpInd], X[maxNormInd2]);
			swap(distVals[rpInd], distVals[maxNormInd2 - clustStartIndex]);
			rpCache[0][0] = maxDistVal;
			rpCache[1][1] = distVals[rpInd].dist;
			rpCache[0][1] = (maxDistVal + distVals[rpInd].dist - maxDistVal2)
					/ 2.0;
			rpInd++;
			// find x_i that gives max f value: x3 ... xM
			for (rpInd = 2; rpInd < M_VAL ; rpInd++) {
				maxNormInd = 0;
				maxDistVal = -INF;
				for (UINT ind = clustStartIndex + rpInd; ind < clustEndIndex;
						ind++) {
					//currEndIndex <- startIndex + rpInd
					for (UINT rpInd2 = 0; rpInd2 < rpInd; rpInd2++)
						xTz[rpInd2] = getDotProduct(X[ind],
								X[rpInd2 + clustStartIndex]);
					double rep_err = getRepErr(rpInd, rpCache,
							lambdaArr[subSize - 1], xTz,
							distVals[ind - clustStartIndex].dist);

					if (rep_err + distVals[ind - clustStartIndex].repErr
							> maxDistVal) {
						maxNormInd = ind;
						maxDistVal = rep_err;
					}
				}
				swap(X[clustStartIndex + rpInd], X[maxNormInd]);
				swap(distVals[rpInd], distVals[maxNormInd - clustStartIndex]);
				updateCache(rpCache, X, clustStartIndex, rpInd, distVals);
			}
			// derive lambda for other vectors
			for (UINT ind = clustStartIndex + rpInd; ind < clustEndIndex;
					ind++) {
				for (UINT rpInd2 = 0; rpInd2 < M_VAL ; rpInd2++)
					xTz[rpInd2] = getDotProduct(X[ind],
							X[rpInd2 + clustStartIndex]);
				updateLambda(M_VAL, rpCache, lambdaArr[ind - clustStartIndex],
						xTz);
			}
		}

		outRp.precision(8);

		for (UINT ind = 0; ind < min(clustSize, (UINT) M_VAL); ind++) {
			UINT origInd = ind + clustStartIndex;

			int label = (int) X[origInd].label;
			double lambdaSum = 0;
			for (UINT ind2 = 0; ind2 < clustEndIndex - clustStartIndex; ind2++)
				lambdaSum += lambdaArr[ind2][ind];
			outRp << label << " " << lambdaSum;

			for (UINT fI = 0; fI < X[origInd].numFeats; fI++)
				outRp << " " << X[origInd].F[fI].fNum << ":"
						<< X[origInd].F[fI].fVal;

			outRp << endl;
			totBatchRpNum++;
		}

		prevCSI = clustStartIndex;
		prevCS = min(clustSize, (UINT) M_VAL);
		clustStartIndex = clustEndIndex;
	}

	for (UINT ind = 0; ind < M_VAL ; ind++) {
		delete[] rpCache[ind];
		delete[] lambdaArr[ind];
	}
	for (UINT ind = M_VAL; ind < subSize; ind++)
		delete[] lambdaArr[ind];
	delete[] rpCache;
	delete[] lambdaArr;
	delete[] xTz;
	delete[] X2;
	delete[] distVals;
	delete[] clustEndInd;
}

// C code for the quickselect algorithm (Hoare's selection algorithm), based on code in Numerical Recipes in C
// selects element in array (arr) that is kth in largest among the first n elements
// quickselect has an expected complexity of O(N) and worst case complexity of O(N^2), though in practice it is fast
distDatType temp;
#define SWAP_D(a, b) {temp.dist=(a.dist);(a.dist)=(b.dist);(b.dist)=temp.dist;temp.ind=(a.ind);(a.ind)=(b.ind);(b.ind)=temp.ind;}

static distDatType quickSelectDist(distDatType *inpArr, UINT n, UINT k) {
	UINT beforeK, afterK, mid;
	UINT i, j;
	distDatType a, temp;	// a is the partition element

	beforeK = 0;
	afterK = n - 1;
	while (1) {
		if (afterK <= beforeK + 1) {//Active partition contains 1 or 2 elements
			if (afterK == beforeK + 1
					&& inpArr[afterK].dist < inpArr[beforeK].dist)//Case of 2 elements
				SWAP_D(inpArr[beforeK], inpArr[afterK]);
			return inpArr[k];
		} else {
			mid = (beforeK + afterK) / 2;
			SWAP_D(inpArr[mid], inpArr[beforeK + 1]);
// rearrange inpArr so that inpArr[afterK] >= inpArr[beforeK + 1] >= inpArr[beforeK]
			if (inpArr[beforeK].dist > inpArr[afterK].dist)
				SWAP_D(inpArr[beforeK], inpArr[afterK]);
			if (inpArr[beforeK + 1].dist > inpArr[afterK].dist)
				SWAP_D(inpArr[beforeK + 1], inpArr[afterK]);
			if (inpArr[beforeK].dist > inpArr[beforeK + 1].dist)
				SWAP_D(inpArr[beforeK], inpArr[beforeK + 1]);
// inpArr[beforeK + 1] is selected as the partion element, a, with sentinels, inpArr[afterK] and inpArr[beforeK]
			a = inpArr[beforeK + 1];
// find all {(i,j): j>= i and inpArr[i] < a < inpArr[j]} and swap
// at the end of the while loop, we have inpArr[beforeK]....inpArr[i] <= a <= inpArr[j]....inpArr[afterK]
			i = beforeK + 1;
			j = afterK;
			while (1) {
				do
					i++;
				while (inpArr[i].dist < a.dist);
				do
					j--;
				while (inpArr[j].dist > a.dist);
				if (j < i)
					break;
				SWAP_D(inpArr[i], inpArr[j]);
			}
// put partition element, a, in correct position in rearranged array
			inpArr[beforeK + 1] = inpArr[j];
			inpArr[j] = a;
// if the position of a, j, is ahead of k, search only till j - 1 in the next iteration
			if (j >= k)
				afterK = j - 1;
// if the position of a, j, is before k, search only from i
			if (j <= k)
				beforeK = i;
		}
	}
}

// Implements FLS2 of DeriveRS. Data is divided into sets of size less than P
static void segregateDataFLS2(UINT anchorInd, UINT startIndex, UINT endIndex,
		vector<dataVect_T> &X, dataVect_T *X2, UINT * clustEndInd, UINT V_VAL,
		distDatType *dVals) {

	if (endIndex - startIndex >= 2 * V_VAL) {	//steps 1,2,3, and 5
		UINT p = 0;
		UINT k;
		UINT midVal;
		for (k = startIndex; k < endIndex; k++) { 	        // step 1
			dVals[p].dist = getDotProduct(X[k], X[anchorInd]);
			dVals[p++].ind = k;
		}
		midVal = p / 2;
		quickSelectDist(dVals, p, midVal);				// step 2
// steps 3 and 5
		for (k = 0; k < p; k++)		//mass swap
			X2[k] = X[dVals[k].ind];
		for (k = 0; k < p; k++)
			X[k + startIndex] = X2[k];
		midVal += startIndex;
		segregateDataFLS2(anchorInd, startIndex, midVal, X, X2, clustEndInd,
				V_VAL, dVals);
		segregateDataFLS2(midVal, midVal, endIndex, X, X2, clustEndInd, V_VAL,
				dVals);
	} else if (endIndex - startIndex > V_VAL) {	//steps 1,2,3, and 4
		UINT p = 0;
		UINT k;
		UINT midVal;
		for (k = startIndex; k < endIndex; k++) {
			dVals[p].dist = getDotProduct(X[k], X[anchorInd]);
			dVals[p++].ind = k;
		}
		midVal = p / 2;
		quickSelectDist(dVals, p, midVal);
		for (k = 0; k < p; k++)		//mass swap
			X2[k] = X[dVals[k].ind];
		for (k = 0; k < p; k++)
			X[k + startIndex] = X2[k];

		for (k = 0; k < midVal; k++)
			clustEndInd[k + startIndex] = midVal + startIndex;
		midVal += startIndex;
		for (k = midVal; k < endIndex; k++)	//step 4
			clustEndInd[k] = endIndex;
	} else {	//step 4
		for (UINT p = startIndex; p < endIndex; p++)
			clustEndInd[p] = endIndex;
	}
}

