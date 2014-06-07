#ifndef __W_R_B_M_TRAINER_H__
#define __W_R_B_M_TRAINER_H__

#include <set>
#include <thread>
#include <mutex>
#include "WRRBM.h"

class CWRRBMTrainer
{
public:
	CWRRBMTrainer(SWRRBM *pRBM, int nThread, bool verbose = false);
	~CWRRBMTrainer();
	double MiniBatchUpdate(vector<vector<int> >::iterator beg,
												 vector<vector<int> >::iterator end,
												 double rate, double biasRate, double momentum, 
												 double l1Reg, double l2Reg);					// regularizer, currently not supported
	
	std::pair<double, double> MiniBatchUpdateMt(vector<vector<int> >::iterator beg,
												 vector<vector<int> >::iterator end,
												 double rate, double biasRate, double momentum, 
												 double l1Reg, double l2Reg);					// regularizer, currently not supported


	void SetVerbose(bool v) {m_verbose = v;}
	void threadFunc(vector<vector<int> >::iterator beg, 
									vector<vector<int> >::iterator end, 
									double *pError,    double *pHit);
private:
	void updateParameter(double rate, double biasRate, int batchSize, double momentum, double l1Reg, double l2Reg);
private:
	
	std::mutex		m_biasMtx;
	std::mutex		m_DMtx;
	std::mutex		m_WMtx;
	std::mutex		m_RBMMtx;
	SWRRBM 	 			*m_pRBM;
	int 					m_nThread;
	bool 					m_verbose;
	double   			*m_gVB, 	*m_incVB;
	double   			*m_gHB, 	*m_incHB;
	double   			*m_gD,  	*m_incD;
	vector<double *> m_gW, 	 m_incW;
	vector<bool>	 m_sparseVec;								// m_sparseVec[w] == true: the w-th row of D has been updated
};


#endif  /*__W_R_B_M_TRAINER_H__*/
