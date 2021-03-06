#include <ctime>
#include <cstdio>
#include <cstdlib>

#include "WRDBNTrainer.h"
#include "CDLearner.h"
#include "Config.h"
#include "Pool.h"
#include "WRRBM.h"
#include "WRDBN.h"

bool ReadData(FILE *fp, 	vector<vector<int> > &rData, 	size_t batchSize,  
							int nLeft, 	int nRight, 	CWordIDMap &widMap);

string CreateTempPath(int LayerId)
{
	string path(CConfig::strTempDir);
	char buf[65536];
	sprintf(buf, "batch%d_l%dr%d_embd%d_layer%d_burnIn%d.temp",
			CConfig::nBatchSize,  CConfig::nLeft,  CConfig::nRight,
			CConfig::nEmbeddingSize,  LayerId,  CConfig::nBurnIn);

	path += buf;
	return path;
}

string CreateModelPath(SWRDBN *pDBN)
{
	char buf[65536];
	sprintf(buf, "ngram%d_embed%d_", pDBN->m_pInLayer->m_nGram, pDBN->m_pInLayer->m_wDim);
	string path(CConfig::strDBNSubName);
	path += buf;
	path += "hSize_";

	for (size_t i = 0; i < pDBN->m_vRBMLayers.size(); ++i)
	{
		sprintf(buf, "%d_", pDBN->m_vRBMLayers[i]->m_vNum);
		path += buf;
	}

	sprintf(buf, "%d", pDBN->m_vRBMLayers[pDBN->m_vRBMLayers.size() - 1]->m_hNum);
	path += buf;
	fprintf(stderr, "Model path %s\n", path.c_str());
	return path;
}


void GeneratingHiddenData
(const char *pszCorpus, const char *pszOut,
 SWRRBM *pRBM, int left, int right, size_t blockSize)
{
	FILE *fpLayer1 = fopen(pszOut, "wb");
	if (fpLayer1 == nullptr)
	{
		fprintf(stderr, "Error: %s open failed\n",  pszOut);
		exit(0);
	}
	FILE *fpCorpus = fopen(pszCorpus, "r");
			
	int nBlock = 0;
	vector<double> hProbs, hSamples;
	time_t start, now;
	time(&start);
	while (!feof(fpCorpus))
	{
		++ nBlock;
		vector<vector<int> > inData;
		ReadData(fpCorpus,  inData,  blockSize, left, right, *pRBM->m_pMap);

		for (size_t i = 0; i < inData.size(); ++i)
		{
			if ((i + 1) % 10000 == 0)
				fprintf(stderr, "Generating data block %-3d, item %lu, dim %d\r",
								nBlock, i + 1, pRBM->m_hNum);

			pRBM->H_Given_V(inData[i], hProbs, hSamples);
//			disVec(&hProbs[0], hProbs.size(), stderr);
			
			fwrite(&hProbs[0], sizeof(double), hProbs.size(), fpLayer1);
//			fwrite(&hSamples[0], sizeof(double), hSamples.size(), fpLayer1);
		}
	}

	time(&now);
	int secs = difftime(now, start);
	fprintf(stderr, "\nData generation done %d secs\n", secs);
	fclose(fpLayer1);
	fclose(fpCorpus);
}


bool ReadHiddenData(FILE *fp,   vector<double *> &rData, 
										int nDim,   int batchSize,   CPool &rPool)
{
	rPool.Recycle();
	rData.clear();
	while (rData.size() < (size_t)batchSize)
	{
		double *pData = (double *)rPool.Allocate(sizeof(double) * nDim);
		if (fread(pData, sizeof(double), nDim, fp) != (size_t) nDim)
		{
			if (feof(fp))
				return true;

			fprintf(stderr, "Error: Reading hidden data failed\n");
			exit(0);
		}
		rData.push_back(pData);
	}

	return feof(fp);
}



void LearningDBN(const char *pszCfgPath)
{
	if (CConfig::ReadConfig(pszCfgPath) == false)
	{
		fprintf(stderr, "Error: reading config file failed\n");
		exit(0);
	}

	SWRRBM *pInLayer = SWRRBM::RBMFromFile(CConfig::strPrefix.c_str());


	// generating data for higher level rbms
	const int blockSize = CConfig::nBlockSize;
	GeneratingHiddenData(CConfig::strCorpus.c_str(),  CreateTempPath(0).c_str(),
											 pInLayer,   CConfig::nLeft,   CConfig::nRight,   blockSize);

	vector<SRBM *> vRBMLayers;
	CPool dataPool;
	vector<double *> inData;
	for (size_t layer = 0; layer < CConfig::vHiddenSizes.size(); ++layer)
	{
		int inDim = layer == 0? pInLayer->m_hNum:
								CConfig::vHiddenSizes[layer - 1];
		SRBM *pRBM = new SRBM(inDim, CConfig::vHiddenSizes[layer]);
		vRBMLayers.push_back(pRBM);

		FILE *fpIn = fopen(CreateTempPath(layer).c_str(), "rb"), *fpOut = nullptr;
		if (layer != CConfig::vHiddenSizes.size() - 1)
			fpOut = fopen(CreateTempPath(layer + 1).c_str(), "wb");
		
		CCDLearner learner(pRBM);
		bool endOfFile = false;
		double error = 0;
		int nBatch = 0, dataSize = 0;
		fprintf(stderr, "\nStart training hidden layer %lu, %d, %d\n", 
						layer,  pRBM->m_vNum,  pRBM->m_hNum);
		for (int epoch = 0; epoch < CConfig::nEpoch; epoch += endOfFile, ++nBatch)
		{
			double mom = epoch > 5 ? CConfig::fFinalMom: CConfig::fInitMom;
			endOfFile = ReadHiddenData(fpIn,  inData,  pRBM->m_vNum,  
														CConfig::nBatchSize,  dataPool);	
			
			if ((int)inData.size() > CConfig::nBatchSize/10)
			{
				dataSize += (int)inData.size();
				error += learner.MiniBatchUpdate(inData.begin(),  
						inData.end(),  CConfig::fPretrainRate,  mom);
				
				fprintf(stderr, "layer %lu, batch %d, error %.4e\r",  
						layer,  nBatch,  error);
			}

			for (size_t i = 0; i < inData.size(); ++i)
			{
				if (i % 5000 == 0)
				{
					vector<double> hProbs, hSamples, v1, p1;
					pRBM->Sampling(inData[i],  pRBM->m_vNum, hProbs, hSamples, SRBM::H_GIVEN_V);
					pRBM->Sampling(&hProbs[0], pRBM->m_hNum, p1,     v1, SRBM::V_GIVEN_H);
					fprintf(stdout, "\n\nlayer %lu, epoch %d, input %d:\n", layer,  epoch, (int)i);
					disVec(inData[i], 0.2, 	0.8, 	5, 	pRBM->m_vNum/5, stdout);
					fprintf(stdout, "recon:\n");
					disVec(&p1[0], 		0.2, 	0.8, 	5, 	pRBM->m_vNum/5, stdout);
				}
			}


			// in the last epoch, generate higher level data
			if (epoch == CConfig::nEpoch - 1 && fpOut != nullptr)
			{
				vector<double> hProbs, hSamples;
				for (size_t i = 0; i < inData.size(); ++i)
				{
					pRBM->Sampling(inData[i], pRBM->m_vNum, hProbs, hSamples, SRBM::H_GIVEN_V);
					fwrite(&hProbs[0], sizeof(double), hProbs.size(), fpOut);
				}
			}

			if (endOfFile)
			{
				fprintf(stderr, "layer %lu, epoch %d, data size %d, error %.4e, avg %.4e\n", 
						layer,  epoch,  dataSize,  error,  error/dataSize);
				error = 0;
				nBatch = dataSize = 0;
				fseek(fpIn, SEEK_SET, 0);
			}
		}

		fclose(fpIn);
		if (fpOut != nullptr)
			fclose(fpOut);
	}

	
	// saving
	SWRDBN dbn(pInLayer, vRBMLayers);
	string modelPath = CreateModelPath(&dbn);
	modelPath = "dbnModels/" + modelPath;
	fprintf(stderr, "Saving to path %s\n", modelPath.c_str());
	dbn.SaveWRDBN(modelPath.c_str());


	// save load checking
	SWRDBN dbnChk;
	dbnChk.LoadWRDBN(modelPath.c_str());

	FILE *fpCorpus = fopen(CConfig::strCorpus.c_str(), "r");
	vector<vector<int> > ids;
	ReadData(fpCorpus,  ids,  1000,  CConfig::nLeft,  
						CConfig::nRight,  *pInLayer->m_pMap);
	for (size_t i = 0; i < ids.size(); ++i)
	{
		vector<vector<double> > p, h, pChk, hChk;
		dbn.Inference(ids[i],  		p,  	h);
		dbnChk.Inference(ids[i], 	pChk, hChk);

		for (size_t layer = 0; layer < p.size(); ++layer)
		{
			for (size_t hid = 0; hid < p[layer].size(); ++hid)
			{
				fprintf(stderr, "layer %lu, size %lu\r", layer, hid);
				if (p[layer][hid] != pChk[layer][hid])
				{
					fprintf(stderr, "inconsistent layer[%lu][%lu] %.4e vs %.4e\n", layer, hid,
									p[layer][hid], pChk[layer][hid]);
				}
			}
		}
	}

	fclose(fpCorpus);
	// free memory
	for (size_t i = 0; i < vRBMLayers.size(); ++i)
		delete vRBMLayers[i];
	delete pInLayer->m_pGen;
	delete pInLayer->m_pMap;
	delete pInLayer;
}

