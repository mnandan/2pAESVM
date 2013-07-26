/*
 * onePAE.cpp
 *
 *  Created on: Jul 23, 2013
 *      Author: mn
 */
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "fileInt.h"

using namespace std;

// /media/Storage/MN/linear/data/epsilon_normalized

// /media/Storage/MN/AESVM/data/Mnist/MnistTest/newFLS1.dat

// /media/Storage/MN/AESVM/data/w8a/w8aTest/newFLS1.dat

int main() {
	char inpBuff[1024];
	char inpFname[1024];
	char outFname[1024];
	char modelFname[1024];

	cout << "Enter available memory size in MB: ";
	cin.getline(inpBuff, 1024);
	double G = atof(inpBuff);
	G = G*1024*1024;  //converting megabytes to bytes

	cout << "Enter name of input data file: ";
	cin.getline(inpFname, 1024);
	double fSize = getFileSize(inpFname);

	cout << "Enter name of file to store representative set: ";
	cin.getline(outFname, 1024);

	cout << "Enter name of file to store solution model: ";
	cin.getline(modelFname, 1024);

	cout << "Enter penalty hyper-parameter C: ";
	cin.getline(inpBuff, 1024);
	double C = atof(inpBuff);

	ifstream inpF(inpFname);
	ofstream outRp(outFname);
	int blockNum = 1;
	UINT totRpNum = 0;
	while (getBlockAE(inpF, outRp, G , C, fSize, totRpNum) != -1) {
		cout << "  Analyzed block " << blockNum << endl;
		blockNum++;
		//cin.getline(inpBuff, 1024);
		//break;
	}
	inpF.close();
	outRp.close();
	if(totRpNum == 0)
		return -1;
	cout << "  Analyzed block " << blockNum << endl;
	cout << "Size of representative set = "<<totRpNum<<endl;


	ifstream inpRp(outFname);
	ofstream outMdl(modelFname);
	if(getAESVMsol(inpRp, outMdl, C, totRpNum) != -1)
		cout << "  AESVM solution computed\n";

	inpRp.close();
	outMdl.close();

	return 0;
}
