#include <cstdio>
#include <ctime>

#include "WRRBM.h"
#include "Config.h"
#include "WordIDMap.h"
#include "WRRBMTrainer.h"
#include "WRDBNTrainer.h"
#include "WRDBN.h"
bool ReadData(FILE *fp, 	vector<vector<int> > &rData, 	
							size_t batchSize,  int nLeft, 	
							int nRight, 	CWordIDMap &widMap);

void CollectStatics(const char *pszPath, CWordIDMap &rMap)
{
	FILE *fp = fopen(pszPath, "r");
	assert(fp != NULL);
	char buf[65536];
	bool insertNewWordMode = false;
	int lineId = 0;
	while (fgets(buf, 65536, fp) != NULL)
	{
		if (++lineId % 10000 == 0)
			fprintf(stderr, "processing %d line\r", lineId);
		vector<char *> words = Split(buf, " \r\t\n");
		for (size_t i = 0; i < words.size(); ++i)
			rMap.Inc(words[i], insertNewWordMode);
	}
	fprintf(stderr, "processing %d line\n", lineId);
	fclose(fp);
}


void ReTrain(const char *pszConfig)
{
	if (CConfig::ReadConfig(pszConfig) == false)
	{
		fprintf(stderr, "Reading config file %s failed\n", pszConfig);
		return;
	}

	SWRRBM *pRBM = SWRRBM::RBMFromFile(CConfig::strRBMPath.c_str());
	fprintf(stderr, "Expanding dict with corpus %s\n",
					CConfig::strCorpus.c_str());
	CWordIDMap &widMap = *pRBM->m_pMap;
	CollectStatics(CConfig::strOldCorpus.c_str(), widMap);
	widMap.ExtractDictFromText(CConfig::strCorpus.c_str());
	widMap.Filter(CConfig::nCutoff);
	
	// compute the multinomial distribution of the vcb
	vector<double> vcbDis(widMap.GetCount().size());
	std::copy(widMap.GetCount().begin(), widMap.GetCount().end(), vcbDis.begin());
	double total = std::accumulate(vcbDis.begin(), vcbDis.end(), 0);
	fprintf(stderr, "total number of words %.2f\n", total);
	for (size_t i = 0; i < vcbDis.size(); ++i)
		vcbDis[i] /= total;


	// update random generator
	CRandomGen rGen;
	if (rGen.Init(vcbDis) == false)
	{
		fprintf(stderr, "Initializing alias table failed\n");
		exit(0);
	}
	pRBM->m_pGen = &rGen;
	pRBM->ReAllocate();

	// initializing rbm and trainer
	int nGram = pRBM->m_nGram;
	int batchSize = CConfig::nBatchSize;
	bool trainingVerbose = false;
	CWRRBMTrainer trainer(pRBM, CConfig::nThreads, trainingVerbose);

	
	// training
	CConfig::nHidden = pRBM->m_hNum;
	CConfig::nLeft   = pRBM->m_nLeft;
	CConfig::nRight  = pRBM->m_nRight;
	CConfig::nBurnIn = pRBM->m_nBurnIn;
	string path = CConfig::BuildPath();
	string logPath = "logs/" + path + ".log";
	FILE *fpLog = fopen(logPath.c_str(), "w");
	bool fileEnd = false;
	FILE *fpCorpus = fopen(CConfig::strCorpus.c_str(), "r");
	assert(fpCorpus);
	vector<vector<int> > vData;
	double error = 0, hit = 0;
	size_t dataSize = 0, nBatch = 0;
	time_t  timer, now;
	time(&timer);
	fprintf(stderr, "Training ....\n");
	for (int epoch = 0; epoch < CConfig::nEpoch; epoch += fileEnd)
	{
		double momentum = epoch > 5 ? CConfig::fFinalMom  :  CConfig::fInitMom;
		fileEnd = ReadData(fpCorpus, 	vData, 	batchSize, 	CConfig::nLeft, 	CConfig::nRight, 	widMap);

		if (epoch >= CConfig::nRBMVbsEpoch)
			pRBM->m_verbose = true;
		
		if (epoch >= CConfig::nTrainVEpoch)
			trainer.SetVerbose(true);
	
		if (1.0 * vData.size() / batchSize > 0.5)
		{
			dataSize += vData.size();
			std::pair<double, double> error_hit;
			error_hit = trainer.MiniBatchUpdateMt(vData.begin(),  vData.end(),  
														CConfig::fPretrainRate, 	CConfig::fBiasRate,  
														momentum,  	CConfig::fL1,		CConfig::fL2);

			error += error_hit.first;
			hit += error_hit.second;
			
			time(&now);
			fprintf(stderr, "epoch %-3d processing  batch %lu, size %lu, secs %d\r", 
							epoch, ++nBatch, vData.size(), (int)difftime(now, timer));
		}

		if (fileEnd == true)
		{
			time(&now);
			int secs = difftime(now, timer);
			timer = now;
			fprintf(stderr, "epoch %-3d error %-7.3e hit %-7.3e dataSize %lu  wer %.2f%% hitRate %.2f%%  mins %d and %d sec\n", 	
					epoch, 	error, hit,	dataSize,  
					100 * error/nGram/dataSize,
					100 * hit/nGram/dataSize,
					secs/60, ((int)secs) % 60);
			
			fprintf(fpLog, "epoch %-3d error %-7.3e hit %-7.3e dataSize %lu  wer %.2f%% hitRate %.2f%%  mins %d and %d sec\n", 	
					epoch, 	error, hit,	dataSize,  
					100 * error/nGram/dataSize,
					100 * hit/nGram/dataSize,
					secs/60, ((int)secs) % 60);
			
			fflush(fpLog);
			fseek(fpCorpus, SEEK_SET, 0);
			error = hit = 0;
			dataSize = nBatch = 0;
			
			char buf[65535];
			
			if ((epoch + 1) % CConfig::nSaveFreq == 0)
			{
				sprintf(buf, "%s_epoch%d", path.c_str(), epoch);
				string modelPath("models/");
				modelPath += buf;
				fprintf(stderr, "Saving model %s\n", modelPath.c_str()); 
				pRBM->Save((modelPath + ".model").c_str());
				rGen.SaveGen((modelPath + ".random").c_str());
				widMap.SaveDict((modelPath + ".dict").c_str());

				pRBM->Display(stdout);
			}
		}
	}

	string modelPath = "models/" + string("retrain") + path;
	pRBM->Save((modelPath + ".model").c_str());
	rGen.SaveGen((modelPath + ".random").c_str());
	widMap.SaveDict((modelPath + ".dict").c_str());
	

	fclose(fpLog);
	fclose(fpCorpus);
}

