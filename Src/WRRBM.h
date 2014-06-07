#ifndef __S_W_R_R_B_M_H__
#define __S_W_R_R_B_M_H__
#include <unordered_map>
#include <vector>
#include <string>
#include <cstring>
#include "RandomGen.h"
#include "Pool.h"
#include "WordIDMap.h"

using std::vector;
using std::string;
using std::unordered_map;

struct SWRRBM
{
public:
	//SWRRBM(int ngram, int hSize, int wDim, int m_nBurnIn, DICT_TYPE *pDict, CRandomGen *pGen, verbose = true);
	SWRRBM(int left, int right,  int hSize,  int wDim, 
				 int m_nBurnIn,   CWordIDMap *pMap, 
				 CRandomGen *pGen,  bool verbose = true);
	SWRRBM();
	
	void H_Given_V(vector<const char *> &m_nGram, 
								 vector<double> &hProbs, 
								 vector<double> &hSamples);
	
	void H_Given_V(vector<int> & ids, 
								 vector<double> &hProbs, 
								 vector<double> &hSamples);
	
	void V_Given_H(vector<double> &hSamples, 	vector<int> &vOld, 
								 vector<double> &vProbs,  	vector<int> &vNew,
								 vector<bool> *pVHit = NULL);			
	
	double ComputeActivate(int wid, int position, 
												 vector<double> &hSamples);
	
	static SWRRBM* RBMFromFile(const char *pszPath);
	void 	 GetEmbeddingBound(double &rMin, double &rMax, 
													 bool normalized = true);

	void   QuantizationEmbedding(int n, double lower, double upper);
	void 	 Display(FILE *fp = stderr);
	bool 	 Save(FILE *fp);
	bool 	 Load(FILE *fp);	
	void 	 CheckNan();
	int  	 GetWDim()									{return m_wDim;}

	double * GetEmbedding(const char *pszWord)
	{
		int wid = m_pMap->GetID(pszWord);
		return m_D + wid * m_wDim;
	}


	bool Save(const char *pszPath)  
	{
		FILE *fp = fopen(pszPath, "wb");
		if (fp == NULL)
		{
			fprintf(stderr, "Error: File %s open failed\n", pszPath);
			exit(0);
		}
		
		Save(fp);
		fclose(fp);
		return true;
	}

	bool Load(const char *pszPath)
	{
		FILE *fp = fopen(pszPath, "rb");
		if (fp == nullptr) 
		{
			fprintf(stderr, "Error: File %s open failed\n", pszPath);
			exit(0);
		}

		Load(fp);
		fclose(fp);
		return true;
	}

	void ReAllocate();

	

	// data members
	int				m_nLeft, m_nRight;
	int 		  m_nGram;					// number of ngrams
	int 			m_hNum;						// hidden layer size
	int				m_wDim;						// dimension of each word
	int 			m_nBurnIn;				// how many samples should be discarded	
	CWordIDMap 	*m_pMap;				// vocab dictionary
	CRandomGen 	*m_pGen;				// generator of the proposal distribution
	int 			m_vcbSize;
	double 	 *m_pHB;
	double   *m_pVB;						// all position share the same visuable bias
	double 	 *m_D; 							// n * d projection layer
	vector<double *> m_vW;			// m_pW[i] is a H*D matrix
	CPool   m_pool;							// no longer share with other SWRRBM object

	// always for debugging
	bool 		m_verbose;
};


#endif  /*__S_W_R_R_B_M_H__*/
