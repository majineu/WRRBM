#include <cstdio>
#include <ctime>

#include "WRRBM.h"
#include "Config.h"
#include "WordIDMap.h"
#include "WRRBMTrainer.h"
#include "WRDBNTrainer.h"
#include "WRDBN.h"


void TrainDepEmbedding(const char *pszPath);
void ReTrain(const char *pszConfig);

bool ReadData(FILE *fp, 	vector<vector<int> > &rData, 	size_t batchSize,  
							int nLeft, 	int nRight, 	CWordIDMap &widMap)
{
	const int MAX_BUF_LEN = 65536;
	char buf[MAX_BUF_LEN];
	char pStart[] = "<s>";
	char pEnd[] = "</s>";
	bool verbose = false;

	int ngram = nLeft + nRight;
	rData.clear();
	CPool pool;
	while (rData.size() < batchSize && fgets(buf, MAX_BUF_LEN, fp) != NULL)
	{
		vector<char *> words = Split(buf, " \t\r\n");
		
		// remove the original start and end symbol
		while (words.size() > 0 && strcmp(words[0], pStart) == 0)
			words.erase(words.begin());
		
		while (words.size() > 0 && strcmp(words.back(), pEnd) == 0)
			words.pop_back();


		size_t nWords = words.size();
		
		// re-insert start and end symbol
		if (nLeft > 0)
			words.insert(words.begin(), nLeft, pStart);
		
		if (nRight > 0)
			words.insert(words.end(), nRight - 1, pEnd);

		
		if (verbose == true)
		{
			fprintf(stderr, "\nsentence :");
			for (size_t i = 0; i < words.size(); ++i)
				fprintf(stderr, "%s ", words[i]);
			fprintf(stderr, "\n");
		}
		
		int *pSenIds = (int *)pool.Allocate(sizeof(int) * words.size());
		for (size_t widx = 0; widx < words.size(); ++widx)
			pSenIds [widx] = widMap.GetID(words[widx]);

		if (verbose == true)
		{
			fprintf(stderr, "\n-----------ids-------------\n");
			for (size_t widx = 0; widx < words.size(); ++widx)
				fprintf(stderr, "%d %s|||", pSenIds[widx], widMap.GetWord(pSenIds[widx]).c_str());
			fprintf(stderr, "\n");
		}
	
		for (size_t i = 0; i < nWords; ++i)
		{
			vector<int> ids(ngram);
			std::copy(pSenIds + i, pSenIds + i + ngram, ids.begin());
			rData.push_back(ids);
			if (verbose == true)
			{
				for (size_t k = 0; k < rData.back().size(); ++k)
					fprintf(stderr, "%-6s ", widMap.GetWord(rData.back()[k]).c_str());
				fprintf(stderr, "\n");
			}
		}
	}

	return feof(fp);
}


void Learn(const char *pszConfig)
{
	if (CConfig::ReadConfig(pszConfig) == false)
	{
		fprintf(stderr, "Reading config file %s failed\n", pszConfig);
		return;
	}
	
	CWordIDMap widMap;
	widMap.ExtractDictFromText(CConfig::strCorpus.c_str());
	widMap.Filter(CConfig::nCutoff);

	
	// compute the multinomial distribution of the vcb
	vector<double> vcbDis(widMap.GetCount().size());
	std::copy(widMap.GetCount().begin(), widMap.GetCount().end(), vcbDis.begin());
	double total = std::accumulate(vcbDis.begin(), vcbDis.end(), 0);
	fprintf(stderr, "total number of words %.2f\n", total);
	
	for (size_t i = 0; i < vcbDis.size(); ++i)
	{
		vcbDis[i] /= total;
	}


	// initializing random generator
	CRandomGen rGen;
	if (rGen.Init(vcbDis) == false)
	{
		fprintf(stderr, "Initializing alias table failed\n");
		exit(0);
	}

#if 0
	long long nSample = 0;
	while (true)
	{
		if (++nSample % 10000000 == 0)
			fprintf(stderr, "sampling %lld\r", nSample);
		if (rGen.Sample() == -1)
			fprintf(stderr, "\nGot, bad smaple %lld\n", nSample); 
	}
#endif


	// initializing rbm and trainer
	int nGram = CConfig::nLeft + CConfig::nRight;
	int hNum  = CConfig::nHidden, wDim  = CConfig::nEmbeddingSize,  batchSize = CConfig::nBatchSize;
	int left  = CConfig::nLeft,  right = CConfig::nRight;	
	bool rbmVerbose = false;
	bool trainingVerbose = false;
	SWRRBM wrRBM(left, right, hNum, wDim, CConfig::nBurnIn, &widMap, &rGen, rbmVerbose);  
	CWRRBMTrainer trainer(&wrRBM, CConfig::nThreads, trainingVerbose);

	
	// training
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
				wrRBM.Save((modelPath + ".model").c_str());
				rGen.SaveGen((modelPath + ".random").c_str());
				widMap.SaveDict((modelPath + ".dict").c_str());

				wrRBM.Display(stdout);
			}
		}
	}

	string modelPath = "models/" + path;
	wrRBM.Save((modelPath + ".model").c_str());
	rGen.SaveGen((modelPath + ".random").c_str());
	widMap.SaveDict((modelPath + ".dict").c_str());
	

	fclose(fpLog);
	fclose(fpCorpus);
}

void Examine(const char *prefix, const char *pszData)
{
	fprintf(stderr, "Testing model....\n");
	FILE *fpCorpus = fopen(pszData, "r");
	assert(fpCorpus);

	CWordIDMap widMap;
	widMap.LoadDict((string(prefix) + string(".dict")).c_str());
	
	// initializing random generator
	CRandomGen rGen;
	rGen.LoadGen((string(prefix) + string(".random")).c_str());
	
	SWRRBM wrRBM;
	wrRBM.m_verbose = true;
	if(wrRBM.Load((string(prefix) + string(".model")).c_str()) == false)
		fprintf(stderr, "loading model failed\n");

	wrRBM.m_pGen = &rGen;
	wrRBM.m_pMap = &widMap;
	vector<vector<int> > vData;
	int batchSize = 1000;
	int left = 3, right = 2;
	bool EndOfFile = false;
	int error = 0, total = 0, batchNum = 0;
	while (!EndOfFile)
	{
		++batchNum;
		EndOfFile = ReadData(fpCorpus, 	vData, 	batchSize, 	left, 	right, 	widMap);
		total += vData.size() * wrRBM.m_nGram;
		for (size_t i = 0; i < vData.size(); ++i)
		{
			vector<int> &vV0 = vData[i], vV1;
			vector<double> vH0, vQ0, vH1, vQ1, vP1;

			// Sampling
			wrRBM.H_Given_V(vV0, vQ0, vH0);
			wrRBM.V_Given_H(vH0, vV0, vP1, vV1);		// sampling visuable using binary hidden state 
			
			for (size_t idx = 0; idx < vV0.size(); ++idx)
				error += vV0[idx] != vV1[idx];
			fprintf(stderr, "batch %-4d, size %-6lu, total %-6d, error %-6d\r", 
					batchNum, 	vData.size(), 	total,	error); 
		}
	}

	fprintf(stderr, "accuracy total %-6d, correct %-6d, accuracy %.2f%%\n",
						total, total - error,  100.0*(total - error)/total);
					
	fclose(fpCorpus);
}

void Loading(const char *pszPath)
{
	SWRRBM rbm;
	rbm.Load(pszPath);
	double l, u;
	rbm.GetEmbeddingBound(l, u);
	fprintf(stderr, "lower %.2f, upper %.2f\n", 	l,  u);
}

void RepairWRRBM(const char *pszPath, int left, int right)
{
	SWRRBM rbm;
	rbm.Load(pszPath);
	rbm.m_nLeft = left, rbm.m_nRight = right;
	rbm.Save(pszPath);
}

void RepairWRDBN(const char *pszPath, int left, int right)
{
	SWRDBN dbn;
	dbn.LoadWRDBN(pszPath);
	dbn.m_pInLayer->m_nLeft  = left;
	dbn.m_pInLayer->m_nRight = right;
	dbn.SaveWRDBN((string(pszPath) + string(".res")).c_str());
}

void QuantEmbedding(int n, const char *pszPath, const string &pszOut)
{
	SWRRBM *pRBM = SWRRBM::RBMFromFile(pszPath);
	double maxV, minV;
	pRBM->GetEmbeddingBound(minV, maxV, true);
	pRBM->QuantizationEmbedding(n, minV, maxV);

	pRBM->m_pMap->SaveDict((pszOut + ".dict").c_str());
	pRBM->m_pGen->SaveGen((pszOut + ".random").c_str());
	pRBM->Save((pszOut + ".model").c_str());

	delete pRBM->m_pGen;
	delete pRBM->m_pMap;
	delete pRBM;
}

void usage(const char *name)
{
	fprintf(stderr, "usage: %s Training <configFile>\n", name);
	fprintf(stderr, "       %s ReTrain <configFile>\n", name);
	fprintf(stderr, "       %s Testing  <modelPrefix>  <testFile>\n", name);
	fprintf(stderr, "       %s TrainDBN <configFile>\n", name);
	fprintf(stderr, "       %s LoadWRRBM <WRRBM>\n", name);
	fprintf(stderr, "       %s RepairWRRBM <WRRBM> nLeft nRight\n", name);
	fprintf(stderr, "       %s QuantEmbedding n <WRRBM>  <ResRBM> \n", name);
	fprintf(stderr, "       %s RepairWRDBN <WRDBN> nLeft nRight\n", name);
	exit(0);
}

int main(int argc, char ** argv)
{
	if (argc < 2)
		usage(*argv);

	if (strcmp(argv[1], "Training") == 0 && argc == 3)
		Learn(argv[2]);
	else if (strcmp(argv[1], "ReTrain") == 0 && argc == 3)
		ReTrain(argv[2]);
	else if (strcmp(argv[1], "Testing") == 0 && argc == 4)
		Examine(argv[2], argv[3]);
	else if (strcmp(argv[1], "TrainDBN") == 0 && argc == 3)
		LearningDBN(argv[2]);		
	else if (strcmp(argv[1], "LoadWRRBM") == 0 && argc == 3)
		Loading(argv[2]);
	else if (strcmp(argv[1], "RepairWRRBM") == 0 && argc == 5)
		RepairWRRBM(argv[2],  atoi(argv[3]),  atoi(argv[4]));
	else if (strcmp(argv[1], "QuantEmbedding") == 0 && argc == 5)
		QuantEmbedding(atoi(argv[2]), argv[3], argv[4]);
	else if (strcmp(argv[1], "RepairWRDBN") == 0 && argc == 5)
		RepairWRDBN(argv[2],  atoi(argv[3]),  atoi(argv[4]));
	else if (strcmp(argv[1], "TrainDepEmbedding") == 0 && argc == 3)
		TrainDepEmbedding(argv[2]);
	else
	{
		usage(argv[0]);
	}
	return 0;
}


