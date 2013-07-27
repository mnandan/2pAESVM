#ifndef _FILEINT_H
#define _FILEINT_H

#include <fstream>

typedef unsigned int UINT;

struct feat_T {
  UINT fNum;       // feature number
  double fVal;
};

struct dataVect_T {
  UINT numFeats;
  char label;
  feat_T * F;
};


double getFileSize(char *fName);
int getBlockAE(std::ifstream & inpF, std::ofstream & outF, double G, double C, double fSize, UINT &totRpNum);
int getAESVMsol(std::ifstream & inpRp, std::ofstream & outMdl, double C, UINT totRpNum);

#endif
