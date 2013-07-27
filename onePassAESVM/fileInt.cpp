#include <iostream>
#include <cstdlib>
#include <cstring>
#include "fileInt.h"
#include <vector>
#include <queue>
#include <algorithm>
#include <fstream>
#include "svmSolver.h"
#include "onePdRS.h"

using namespace std;

double getFileSize(char *fName) {
	ifstream inpF(fName);
	inpF.seekg(0, inpF.end);
	double fSize = (double) inpF.tellg();
	inpF.close();
	return fSize;
}

int getBlockAE(ifstream & inpF, ofstream & outRp, double G, double C,
		double fSize, UINT & totRpNum) {
	if (inpF.bad())
		return -1;

	vector<dataVect_T> X;
	UINT maxF = 0;			// largest index of features among all vectors
	int lineNum = 0;
	double memUsed = 0;
	double maxNumFeats = 0;		//maximum number of features in a vector
	while (1) {
		string line;
		streampos FposRevert = inpF.tellg();
		if (getline(inpF, line)) {
			char * lineC = new char[line.length() + 1];
			strcpy(lineC, line.c_str());
			char * labelC = strtok(lineC, " \t\n");
			if (labelC == NULL) {
				cout << "Error in input file read\n";
				return -1;
			}
			char *endPtr, *idx, *val;
			char label = (char) strtol(labelC, &endPtr, 10);
			queue<feat_T> tempQ;
			UINT numFeats = 0, fNum = 0;
			double fVal = 0;
			if (endPtr == labelC || *endPtr != '\0') {
				cout << "Error in input file read\n";
				return -1;
			}

			while (1) {
				idx = strtok(NULL, ":");
				val = strtok(NULL, " \t");

				if (val == NULL)
					break;

				fNum = (UINT) strtoll(idx, &endPtr, 10);
				if (endPtr == idx || *endPtr != '\0') {
					cout << "Error in input file read\n";
					return -1;
				}

				fVal = strtod(val, &endPtr);
				if (endPtr == val || (*endPtr != '\0' && !isspace(*endPtr))) {
					cout << "Error in input file read\n";
					return -1;
				}
				feat_T tempF;
				tempF.fNum = fNum;
				tempF.fVal = fVal;
				tempQ.push(tempF);
				numFeats++;
				maxF = max(maxF, fNum);
			}
			delete[] lineC;
			maxNumFeats = max(maxNumFeats, (double)numFeats);

			double memReq = (double) sizeof(dataVect_T) + sizeof(feat_T) * numFeats;
			if (memUsed + memReq >= G) {
				if (numFeats > 0) {
					for (UINT ind = 0; ind < numFeats; ind++)
						tempQ.pop();
					queue<feat_T> empty;
					swap(tempQ, empty);
				}
				inpF.seekg(FposRevert);
				break;
			}
			dataVect_T Xtemp;
			Xtemp.numFeats = numFeats;
			Xtemp.label = label;
			if (numFeats > 0) {
				Xtemp.F = new feat_T[numFeats];
				for (UINT ind = 0; ind < numFeats; ind++) {
					Xtemp.F[ind] = tempQ.front();
					tempQ.pop();
				}
				queue<feat_T> empty;
				swap(tempQ, empty);
			} else
				Xtemp.F = NULL;

			X.push_back(Xtemp);
			memUsed += memReq;
		} else {
			break;
		}
		lineNum++;
	}
	maxF++;
	double *w = new double[maxF];
	// solve SVM for selected vectors
	svmSolver(X, w, C, maxF);
	// calc number of rp vects.
	double numRp = G / ((double) sizeof(feat_T) * maxNumFeats);
	numRp = numRp * G / fSize;
	if (numRp < 1) {
		cerr << "Not enough memory to store representative set.\n";
		exit(1);
	}
	// comp AE vects
	onePassDRS(X, w, 1, numRp, outRp, totRpNum);

	for (UINT ind = 0; ind < X.size(); ind++) {
		if (X[ind].F != NULL)
			delete[] X[ind].F;
	}
	vector<dataVect_T>().swap(X);
	delete[] w;
	if (inpF.eof())
		return -1;

	return 0;
}

int getAESVMsol(std::ifstream & inpRp, std::ofstream & outMdl,
		double C, UINT totRpNum) {
	if (inpRp.bad())
		return -1;

	dataVect_T * X = new dataVect_T[totRpNum];
	double *B = new double[totRpNum];
	UINT maxF = 0;
	UINT lineNum = 0;

	while (1) {
		string line;
		if (getline(inpRp, line)) {
			char * lineC = new char[line.length() + 1];
			strcpy(lineC, line.c_str());
			char * labelC = strtok(lineC, " \t\n");
			if (labelC == NULL) {
				cout << "Error in input file read\n";
				return -1;
			}
			char *endPtr, *idx, *val;
			char label = (char) strtol(labelC, &endPtr, 10);
			queue<feat_T> tempQ;
			UINT numFeats = 0, fNum = 0;
			double fVal = 0;
			if (endPtr == labelC || *endPtr != '\0') {
				cout << "Error in input file read\n";
				return -1;
			}

			val = strtok(NULL, " \t");
			fVal = strtod(val, &endPtr);
			if (endPtr == val || (*endPtr != '\0' && !isspace(*endPtr))) {
				cout << "Error in input file read\n";
				return -1;
			}
			B[lineNum] = fVal;

			while (1) {
				idx = strtok(NULL, ":");
				val = strtok(NULL, " \t");

				if (val == NULL)
					break;

				fNum = (UINT) strtoll(idx, &endPtr, 10);
				if (endPtr == idx || *endPtr != '\0') {
					cout << "Error in input file read\n";
					return -1;
				}

				fVal = strtod(val, &endPtr);
				if (endPtr == val || (*endPtr != '\0' && !isspace(*endPtr))) {
					cout << "Error in input file read\n";
					return -1;
				}
				feat_T tempF;
				tempF.fNum = fNum;
				tempF.fVal = fVal;
				tempQ.push(tempF);
				numFeats++;
				maxF = max(maxF, fNum);
			}

			X[lineNum].numFeats = numFeats;
			X[lineNum].label = label;
			if (numFeats > 0) {
				X[lineNum].F = new feat_T[numFeats];
				for (UINT ind = 0; ind < numFeats; ind++) {
					X[lineNum].F[ind] = tempQ.front();
					tempQ.pop();
				}
				queue<feat_T> empty;
				swap(tempQ, empty);
			} else
				X[lineNum].F = NULL;

			delete[] lineC;
		} else {
			break;
		}
		lineNum++;
	}
	maxF++;
	double *w = new double[maxF];
	if(lineNum == totRpNum) {
		// solve AESVM
		svmSolverRp(X, w, B, C, totRpNum, maxF);
	}
	else
		cerr<<"Could not read representative set file"<<endl;
// write AESVM model to file
	if(outMdl.is_open() && outMdl.good()) {
		outMdl<< "solver_type L2R_L1LOSS_SVC_DUAL\n"
				<< "nr_class 2\nlabel 1 -1\nnr_feature "<<maxF<<"\n"
				<< "bias -1\nw"<<endl;
		for(UINT fI = 1; fI < maxF; fI++)
			outMdl<< w[fI]<<endl;
	}
	else
		cerr<<"Cannot write to model file\n";

	for (UINT ind = 0; ind < totRpNum; ind++) {
		if (X[ind].F != NULL)
			delete[] X[ind].F;
	}
	delete[] X;
	delete[] w;
	delete [] B;

	if(lineNum != totRpNum)
		return -1;

	return 0;
}
