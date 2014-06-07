#ifndef __CONFIG_H__
#define __CONFIG_H__
#include <string>
#include <vector>

using std::string;
using std::vector;

class CConfig
{
public:
	static int		nEpoch;
	static int    nHidden;
	static int    nEmbeddingSize;
	static int 		nThreads;

	static int    nBatchSize;
	static int    nLeft;
	static int 		nRight;
	static int    nBurnIn;
	static int		nCutoff;
	static int 		nSaveFreq;
	static int		nBlockSize;
	
	static int		nRBMVbsEpoch;			// from which rbm verbose
	static int		nTrainVEpoch;			// from which training verbose

	static double fPretrainRate;
	static double fBiasRate;
	static double fInitMom;					// initial momentum
	static double fFinalMom;				// final momentum
	static double fL1;
	static double fL2;

	static string strCorpus;
	static string strPrefix;	
	static string strTempDir;
	static string strDBNSubName;
	static string strLogDir;
	static string strModelDir;

	
	static string strOldCorpus;
	static string strRBMPath;				// 	for re-training an existing 
																	//	model on a different corpus

	static vector<int>  vHiddenSizes;
public:
	static bool ReadConfig(const char *pszPath);
	static bool SaveConfig(const char *pszPath);
	static bool LoadConfig(const char *pszPath);
	static string BuildPath();
};


#endif  /*__CONFIG_H__*/
