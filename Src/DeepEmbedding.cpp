#include "WordIDMap.h"
#include "Config.h"
#include "WRRBM.h"
#include "WRRBMTrainer.h"

// when batch size is set -1, only initialize 
// word id map for fp
bool ReadDepData(FILE *fp, vector<vector<int> > &rData,
								 int batchSize,  CWordIDMap & widMap,
								 int nOrder)
{
	const int MAX_BUF_LEN = 65536;
	char buf[MAX_BUF_LEN];
	char pRoot[] = "</s>";
	bool verbose = false;

	rData.clear();
	while (!feof(fp) && 
					(batchSize == -1 || (int)rData.size() < batchSize))
	{
		vector<int> vWids, vHids;
		widMap.Inc(pRoot);
		if (verbose == true)
			fprintf(stdout, "\n");

		while (fgets(buf, MAX_BUF_LEN, fp) != NULL)
		{
			if (verbose == true)
			{
				fprintf(stdout, buf);
			}

			vector<char *> items = Split(buf, " \t\r\n");
			if (items.size() == 0)
				break;

			vWids.push_back(widMap.Inc(items[0]));
			vHids.push_back(atoi(items[2]));
		}

		
		if (batchSize != -1)
		{
			for (size_t i = 0; i < vWids.size(); ++i)
			{
				vector<int> sample(1, vWids[i]);
				int headIdx = vHids[i];
				for (int h = 0; h < nOrder; ++h)
				{
					if (headIdx == 0)
						sample.push_back(widMap.GetID(pRoot));
					else
					{
						sample.push_back(vWids[headIdx - 1]);
						headIdx = vHids[headIdx - 1];
					}
				}

				if (verbose == true)
				{
					for (size_t i = 0; i < sample.size(); ++i)
						fprintf(stdout, "%s ", widMap.GetWord(sample[i]).c_str());
					fprintf(stdout, "\n");
				}

				rData.push_back(sample);
			}
		}		// if (batchSize != -1)
	}

	return feof(fp);
}


void TrainDepEmbedding(const char *pszPath)
{
	if (CConfig::ReadConfig(pszPath) == false)
	{
		fprintf(stderr, "Reading config file %s failed\n", pszPath);
		return;
	}
	
	// initialize word id map
	CWordIDMap widMap;
	FILE *fpCps = fopen(CConfig::strCorpus.c_str(), "r");
	vector<vector<int> > data;
	ReadDepData(fpCps, data, -1, widMap, 0);
	widMap.Filter(CConfig::nCutoff);
	fseek(fpCps, SEEK_SET, 0);
	fclose(fpCps);

	
	// compute the multinomial distribution of the vcb
	vector<double> vcbDis(widMap.GetCount().size());
	std::copy(widMap.GetCount().begin(), widMap.GetCount().end(), vcbDis.begin());
	double total = std::accumulate(vcbDis.begin(), vcbDis.end(), 0);
	fprintf(stderr, "total number of words %.2f\n", total);
	for (size_t i = 0; i < vcbDis.size(); ++i)
		vcbDis[i] /= total;


	// initializing random generator
	CRandomGen rGen;
	if (rGen.Init(vcbDis) == false)
	{
		fprintf(stderr, "Initializing alias table failed\n");
		exit(0);
	}


	// initializing rbm and trainer
	int wDim  = CConfig::nEmbeddingSize,  batchSize = CConfig::nBatchSize;
	
	// left is always 0
	// right: order
	int hNum  = CConfig::nHidden;
	int left  = CConfig::nLeft,  right = CConfig::nRight;
	int order = CConfig::nRight - 1;
	bool rbmVerbose = false;
	bool trainingVerbose = false;
	SWRRBM wrRBM(	left, 		right, 		hNum, 		wDim, 
								CConfig::nBurnIn, &widMap, &rGen, rbmVerbose);  
	CWRRBMTrainer trainer(&wrRBM, CConfig::nThreads, trainingVerbose);

	
	// training
	string path = CConfig::BuildPath();
	string logPath = CConfig::strLogDir + "/" + path + ".log";
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
		double momentum = CConfig::fInitMom + epoch * 0.02;
		if (momentum > 0.9)
			momentum = 0.9;

		fileEnd = ReadDepData(fpCorpus, 	vData, 	batchSize, 	
													widMap,		order);
	
		if (epoch >= CConfig::nRBMVbsEpoch)
			wrRBM.m_verbose = true;
		
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
					100 * error/right/dataSize,
					100 * hit/right/dataSize,
					secs/60, ((int)secs) % 60);
			
			fprintf(fpLog, "epoch %-3d error %-7.3e hit %-7.3e dataSize %lu  wer %.2f%% hitRate %.2f%%  mins %d and %d sec\n", 	
					epoch, 	error, hit,	dataSize,  
					100 * error/right/dataSize,
					100 * hit/right/dataSize,
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
				wrRBM.Save((modelPath + ".model").c_str());
				rGen.SaveGen((modelPath + ".random").c_str());
				widMap.SaveDict((modelPath + ".dict").c_str());

				wrRBM.Display(stdout);
			}
		}
	}

	string modelPath = CConfig::strModelDir + "/" + path;
	wrRBM.Save((modelPath + ".model").c_str());
	rGen.SaveGen((modelPath + ".random").c_str());
	widMap.SaveDict((modelPath + ".dict").c_str());
	

	fclose(fpLog);
	fclose(fpCorpus);
}

