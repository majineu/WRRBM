#ifndef __C_D_LEARNER_H__
#define __C_D_LEARNER_H__

#include "RBM.h"
#include "Pool.h"

#include <vector>
using std::vector;

class CCDLearner
{
public:
	CCDLearner(SRBM *pRBM, bool verbose = false);
	~CCDLearner()
	{
		m_pRBM = NULL;
		delete[] m_pVBiasInc;
		delete[] m_pHBiasInc;
		delete[] m_pWInc;
		delete[] m_gVB;
		delete[] m_gHB;
		delete[] m_gW;
	}

	double UpdateCD1(double *pInputVec, int inLen, double rate, double momentum = 0.5);
	double MiniBatchUpdate(vector<vector<double> >::iterator beg,
												 vector<vector<double> >::iterator end,
												double rate, double momentum);

	double MiniBatchUpdate(vector<double* >:: iterator iterBeg,
								vector<double* >:: iterator iterEnd,
								double rate, 		double momentum);
	void Learning(vector<double *> &rData, double rate, int maxIter);

private:
	SRBM * m_pRBM;
	bool   m_verbose;


	double *m_gVB;
	double *m_gHB;
	double *m_gW;
	double *m_pVBiasInc;				// mimic hinton's code
	double *m_pHBiasInc;				// mini  hinton's code to use momentum
	double *m_pWInc;
};


#endif  /*__C_D_LEARNER_H__*/
