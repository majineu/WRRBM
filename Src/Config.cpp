#include <cstring>
#include "Config.h"
using std::pair;
/* static members of class config */
int		CConfig::nEpoch		 = -1;
int   CConfig::nHidden   = -1;
int 	CConfig::nBatchSize = 100;
int		CConfig::nLeft = 2;
int		CConfig::nRight = 2;
int		CConfig::nCutoff = 2;
int   CConfig::nBurnIn = 10;
int 	CConfig::nEmbeddingSize = 50;
int		CConfig::nThreads = 1;
int		CConfig::nSaveFreq = 3;
int 	CConfig::nBlockSize = 100000;

int		CConfig::nRBMVbsEpoch = 100000;
int		CConfig::nTrainVEpoch = 100000;


double  CConfig::fBiasRate = 0.1;
double  CConfig::fL1 = 0.0;
double  CConfig::fL2 = 0.0;
double  CConfig::fPretrainRate = 0.1;
double  CConfig::fInitMom = 0.5;
double  CConfig::fFinalMom = 0.5;

string	CConfig::strCorpus("NULL_STR");
string	CConfig::strOldCorpus("NULL_STR");
string  CConfig::strPrefix("NULL_STR");
string	CConfig::strDBNSubName("NULL_STR");
string  CConfig::strTempDir("NULL_STR");
string  CConfig::strLogDir("NULL_STR");
string  CConfig::strModelDir("NULL_STR");
string	CConfig::strRBMPath("NULL_STR");
string  CConfig::strTempDataPrefix("NULL_STR");
bool		CConfig::bDBNBySample = true;

vector<int> CConfig::vHiddenSizes;

void removeNewLine(char *pBuf)
{
	for (int i = 0; i < (int)strlen(pBuf); ++i)
	{
		if (pBuf[i] == '\r' || pBuf[i] == '\n')
		{
			pBuf[i] = 0;
			break;
		}
	}
}

bool CConfig::SaveConfig(const char *pszPath)
{
	FILE *fp = fopen(pszPath, "w");
//	CHK_OPEN_FAILED(fp, pszPath);
	fclose(fp);
	return true;
}


bool CConfig::LoadConfig(const char *pszPath)
{
	FILE *fp = fopen(pszPath, "r");
//	CHK_OPEN_FAILED(fp, pszPath);
	fclose(fp);
	return true;
}


bool CConfig::
ReadConfig(const char * pszPath)
{
	FILE *fpIn = fopen(pszPath, "r");
	if (fpIn == NULL)
	{
		fprintf(stderr, "Error: Open %s failed\n", pszPath);
		return false;
	}

	const int BUF_LEN = 256;
	char buf[BUF_LEN];
	fprintf (stderr, "-------------------Config-----------------\n");
	while (fgets(buf, BUF_LEN, fpIn) != NULL)
	{
		if (buf [0] == ':' && buf [1] == ':')
			continue;
		
		removeNewLine(buf);
		if (strlen(buf) == 0)
			continue;


		char *pKey = strtok(buf, " \r\t\n");
		if (pKey == NULL)
		{
			fprintf(stderr, "Error: config file invalid format %s\n", buf);
			return false;
		}
		
		char *pVal = strtok(NULL, " \r\t\n");
		if (pVal == NULL)
		{
			fprintf(stderr, "Error: config file invalid format %s\n", buf);
			return false;
		}

		if (strcmp(pKey, "nBatchSize") == 0)
		{
			CConfig::nBatchSize = atoi(pVal);
			fprintf(stderr, "nBatchSize %d\n", CConfig::nBatchSize);
		}
		else if (strcmp(pKey, "bDBNBySample") == 0)
		{
			CConfig::bDBNBySample = string(pVal) == "true";
			fprintf(stderr, "using Sample as training data : %d\n",
					CConfig::bDBNBySample);
		}

		else if (strcmp(pKey, "strTempDataPrefix") == 0)
		{
			CConfig::strTempDataPrefix = pVal;
			fprintf(stderr, "temp data file prefix %s\n",
					CConfig::strTempDataPrefix.c_str());
		}
		
		else if (strcmp(pKey, "nRight") == 0)
		{
			CConfig::nRight = atoi(pVal);
			fprintf(stderr, "nRight: %d\n", CConfig::nRight);
		}

		else if (strcmp(pKey, "nThreads") == 0)
		{
			CConfig::nThreads = atoi(pVal);
			fprintf(stderr, "thread number %d\n", CConfig::nThreads);
		}

		else if (strcmp(pKey, "nHidden") == 0)
		{
			CConfig::nHidden = atoi(pVal);
			fprintf(stderr, "hidden layer size: %d\n", CConfig::nHidden);
		}

		else if (strcmp(pKey, "nLeft") == 0)
		{
			CConfig::nLeft = atoi(pVal);
			fprintf(stderr, "nLeft: %d\n", CConfig::nLeft);
		}

		else if (strcmp(pKey, "nEpoch") == 0)
		{
			CConfig::nEpoch = atoi(pVal);
			fprintf(stderr, "nEpoch %d\n", CConfig::nEpoch);
		}
		
		else if (strcmp(pKey, "strOldCorpus") == 0)
		{
			CConfig::strOldCorpus = pVal;
			fprintf(stderr, "collect word count from old corpus %s\n", CConfig::strOldCorpus.c_str());
		}

		else if (strcmp(pKey, "strCorpus") == 0)
		{
			CConfig::strCorpus = pVal;
			fprintf(stderr, "training corpus %s\n", CConfig::strCorpus.c_str());
		}

		else if (strcmp(pKey, "nRBMVEpoch") == 0)
		{
			CConfig::nRBMVbsEpoch = atoi(pVal);
			fprintf(stderr, "rbm verbose epoch %d\n", CConfig::nRBMVbsEpoch);
		}

		else if (strcmp(pKey, "nTrainVEpoch") == 0)
		{
			CConfig::nTrainVEpoch = atoi(pVal);
			fprintf(stderr, "training verbose epoch %d\n", CConfig::nTrainVEpoch);
		}

		else if (strcmp(pKey, "nBurnIn") == 0)
		{
			CConfig::nBurnIn = atoi(pVal);
			fprintf(stderr, "Burn-In %d\n", CConfig::nBurnIn); 
		}

		else if (strcmp(pKey, "strLogDir") == 0)
		{
			CConfig::strLogDir = pVal;
			fprintf(stderr, "Log directory %s\n", CConfig::strLogDir.c_str());
		}

		else if (strcmp(pKey, "strModelDir") == 0)
		{
			CConfig::strModelDir = pVal;
			fprintf(stderr, "Model directory %s\n", CConfig::strModelDir.c_str());
		}

		else if (strcmp(pKey, "strTempDir") == 0)
		{
			CConfig::strTempDir = pVal;
			fprintf(stderr, "Temp data directory %s\n", CConfig::strTempDir.c_str());
		}

		else if (strcmp(pKey, "nCutoff") == 0)
		{
			CConfig::nCutoff = atoi(pVal);
			fprintf(stderr, "cutoff %d\n", CConfig::nCutoff);
		}

		else if (strcmp(pKey, "nEmbeddingSize") == 0)
		{
			CConfig::nEmbeddingSize = atoi(pVal);
			fprintf(stderr, "embedding size %d\n", CConfig::nEmbeddingSize);
		}

		else if (strcmp(pKey, "fL1") == 0)
		{
			CConfig::fL1 = atof(pVal);
			fprintf(stderr, "l1 regularization %.2f\n", CConfig::fL1);
		}

		else if (strcmp(pKey, "fL2") == 0)
		{
			CConfig::fL2 = atof(pVal);
			fprintf(stderr, "l2 regularization %.2f\n", CConfig::fL2);
		}
		
		else if (strcmp(pKey, "fBiasRate") == 0)
		{
			CConfig::fBiasRate = atof(pVal);
			fprintf(stderr, "bias learning rate:%.5f\n", CConfig::fBiasRate);
		}

		else if (strcmp(pKey, "fPretrainRate") == 0)
		{
			CConfig::fPretrainRate = atof(pVal);
			fprintf(stderr, "pretrain learning rate:%.5f\n", CConfig::fPretrainRate);
		}

		else if (strcmp(pKey, "fInitMom") == 0)
		{
			CConfig::fInitMom = atof(pVal);
			fprintf(stderr, "Initial momentum:%.3f\n", CConfig::fInitMom);
		}

		else if (strcmp(pKey, "fFinalMom") == 0)
		{
			CConfig::fFinalMom = atof(pVal);
			fprintf(stderr, "final momentum:%.3f\n", CConfig::fFinalMom);
		}
		
		else if (strcmp(pKey, "strRBMPath") == 0)
		{
			CConfig::strRBMPath = pVal;
			fprintf(stderr, "retrain model %s\n", 
					CConfig::strRBMPath.c_str());
		}

		else if (strcmp(pKey, "prefix") == 0)
		{
			CConfig::strPrefix = pVal;
			fprintf(stderr, "prefix %s\n", CConfig::strPrefix.c_str());
		}

		else if (strcmp(pKey, "nFreq") == 0)
		{
			CConfig::nSaveFreq = atoi(pVal);
			fprintf(stderr, "save frequency %d\n", CConfig::nSaveFreq);
		}

		else if (strcmp(pKey, "strDBNSubName") == 0)
		{
			CConfig::strDBNSubName = pVal;
			fprintf(stderr, "DBN model sub name %s\n", CConfig::strDBNSubName.c_str());
		}
		
		else if (strcmp(pKey, "HiddenSize") == 0)
		{
			char *pSizes = pVal; 
			vHiddenSizes.clear();
			while (pSizes != NULL)
			{
				vHiddenSizes.push_back(atoi(pSizes));
				pSizes = strtok(NULL, " \t\r\n");
			}
		}
	}

	fclose(fpIn);
	fprintf(stderr, "\n-------------------------------------\n");
	return nEpoch > 0;
}

string CConfig::BuildPath()
{
	char buf[65536];
	sprintf(buf, "mom%.2f_%.2fbatch%d_l%d_r%d_hidden%d_rb%.3f_rW%.3f_embeding%d_burnIn%d_cutoff%d_l1%.3f_l2%.3f",
					CConfig::fInitMom, 		CConfig::fFinalMom,		CConfig::nBatchSize, 			CConfig::nLeft, 		
					CConfig::nRight, 			CConfig::nHidden, 	CConfig::fBiasRate, 		CConfig::fPretrainRate,
					CConfig::nEmbeddingSize, 	CConfig::nBurnIn,   CConfig::nCutoff, 		CConfig::fL1, 		CConfig::fL2);

	string path = CConfig::strPrefix + buf;
	fprintf(stderr, "path %s\n", path.c_str());
	return path;
}


//--------------------------------------------------------------------------------------------
