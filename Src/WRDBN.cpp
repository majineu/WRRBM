#include "WRDBN.h"


SWRDBN::
SWRDBN(SWRRBM *pInLayer,  vector<SRBM *> &vHiddens)
{
	m_pInLayer = pInLayer;
	m_vRBMLayers.resize(vHiddens.size(), nullptr);
	std::copy(vHiddens.begin(),  vHiddens.end(),  m_vRBMLayers.begin());
	m_nLayer = 1 + (int)vHiddens.size();
}



void SWRDBN::
Inference(vector<int> &ids,  
					vector<vector<double> > & vhProbs,
					vector<vector<double> > & vhSamples)
{
	vhProbs.clear();
	vhSamples.clear();
	vector<double> hProbs, hSamples;

	m_pInLayer->H_Given_V(ids, hProbs, hSamples);
	vhProbs.push_back(hProbs);
	vhSamples.push_back(hSamples);

	for (size_t i = 0; i < m_vRBMLayers.size(); ++i)
	{
		vector<double> & vIn = vhProbs[i];
		m_vRBMLayers[i]->Sampling(&vIn[0],  vIn.size(),  hProbs, 
															 hSamples,  SRBM::H_GIVEN_V);

		vhProbs.push_back(hProbs);
		vhSamples.push_back(hSamples);
	}
}


void SWRDBN::
SaveWRDBN(const char *pszPath)
{
	FILE *fp = fopen(pszPath, "wb");
	if (fp == nullptr)
	{
		fprintf(stderr, "Error: file %s open failed\n", pszPath);
		exit(0);
	}

	
	// writing mate data
	int nLayer = (int)m_vRBMLayers.size();
	fwrite(&nLayer, sizeof(nLayer), 1, fp);
	vector<int> sizes;
	for (size_t i = 0; i < m_vRBMLayers.size(); ++i)
		sizes.push_back(m_vRBMLayers[i]->m_vNum);

	// for the last layer, 
	sizes.push_back(m_vRBMLayers[m_vRBMLayers.size() - 1]->m_hNum);
	fwrite(&sizes[0], sizeof(int), sizes.size(), fp);
	
	fprintf(stderr, "Hidden sizes:");
	for (size_t i = 0; i < sizes.size(); ++i)
		fprintf(stderr, "%d ", sizes[i]);
	fprintf(stderr, "\n");

	// writing rbm layers	
	for (size_t i = 0; i < m_vRBMLayers.size(); ++i)
		m_vRBMLayers[i]->SaveRBM(fp);
	
	m_pInLayer->m_pGen->SaveGen(fp);
	m_pInLayer->Save(fp);

	string strPath (pszPath);
	m_pInLayer->m_pMap->SaveDict((strPath + ".dict").c_str());

	fclose(fp);
}



void SWRDBN::
LoadWRDBN(const char *pszPath)
{
	FILE *fp = fopen(pszPath, "rb");
	if (fp == nullptr)
	{
		fprintf(stderr, "Error: file %s open failed\n", pszPath);
		exit(0);
	}

	int nHiddenLayer = -1;
	fread(&nHiddenLayer, sizeof(nHiddenLayer), 1, fp);
	fprintf(stderr, "%d hidden layers\n", nHiddenLayer);


	// reading sizes of each layer
	vector<int> sizes(nHiddenLayer + 1);
	fread(&sizes[0],  sizeof(int),  sizes.size(),  fp);


	// reading rbm layer
	for (size_t i = 0; i < sizes.size() - 1; ++i)
	{
		int inDim = sizes[i], outDim = sizes[i + 1];
		SRBM *pRBM = new SRBM(inDim, outDim);
		pRBM->LoadRBM(fp);
		m_vRBMLayers.push_back(pRBM);
	}


	// reading wrrbm related 
	m_pInLayer = new SWRRBM();
	m_pInLayer->m_pGen = new CRandomGen();
	m_pInLayer->m_pGen->LoadGen(fp);
	m_pInLayer->Load(fp);

	m_pInLayer->m_pMap = new CWordIDMap();
	string strPath (pszPath);
	m_pInLayer->m_pMap->LoadDict((strPath + ".dict").c_str());
	fclose(fp);
}



