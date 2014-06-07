#ifndef __W_R_D_B_N_H__
#define __W_R_D_B_N_H__

#include <vector>
#include "RBM.h"
#include "WRRBM.h"
using std::vector;

struct SWRDBN
{
	SWRDBN(SWRRBM *pInLayer, vector<SRBM *> &vHiddens);
	SWRDBN():m_pInLayer(nullptr), m_nLayer(-1){}
	SRBM *GetRBM(int layer) 		{return m_vRBMLayers[layer];}
	~SWRDBN() {}


	void Inference(vector<string> &nGram,  
								 vector<vector<double> > &hProbs,
								 vector<vector<double> > &hSamples)
	{
		vector<int> ids;
		if ((int)nGram.size() != m_pInLayer->m_nGram)
		{
			fprintf(stderr, "Error: input length %lu vs ngram %d\n",
					nGram.size(),   m_pInLayer->m_nGram);
			exit(0);
		}
		
		for (size_t i = 0; i < nGram.size(); ++i)
			ids.push_back(m_pInLayer->m_pMap->GetID(nGram[i]));

		Inference(ids, hProbs, hSamples);
	}
	
	void Inference(vector<int> & ids, 
								 vector<vector<double> >& hProbs, 
								 vector<vector<double> >& hSamples);
	
	
	void SaveWRDBN(const char *pszPath);
	void LoadWRDBN(const char *pszPath);


	// 	for the data members
	SWRRBM 					*m_pInLayer;
	vector<SRBM *> 	 m_vRBMLayers;
	int							 m_nLayer;
};


#endif  /*__W_R_D_B_N_H__*/
