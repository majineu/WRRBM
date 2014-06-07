#include <ctime>
#include <cassert>

#include "Config.h"
#include "CDLearner.h"
#include "util.h"

CCDLearner::
CCDLearner(SRBM *pRBM, bool verbose):
	m_pRBM(pRBM),  m_verbose(verbose)
{
	m_pVBiasInc = new double[pRBM->m_vNum];
	m_pHBiasInc = new double[pRBM->m_hNum];
	m_pWInc			= new double[pRBM->m_vNum * pRBM->m_hNum];

	assert(m_pVBiasInc != NULL && m_pHBiasInc != NULL && m_pWInc != NULL);
	
	memset(m_pVBiasInc, 0, sizeof(double) * pRBM->m_vNum);
	memset(m_pHBiasInc, 0, sizeof(double) * pRBM->m_hNum);
	memset(m_pWInc, 0, sizeof(double) * pRBM->m_hNum * pRBM->m_vNum);
	
	m_gVB = new double[pRBM->m_vNum];
	m_gHB = new double[pRBM->m_hNum];
	m_gW	= new double[pRBM->m_vNum * pRBM->m_hNum];

	assert(m_gVB != NULL && m_gHB != NULL && m_gW != NULL);
	
	memset(m_gVB, 0, sizeof(double) * pRBM->m_vNum);
	memset(m_gHB, 0, sizeof(double) * pRBM->m_hNum);
	memset(m_gW, 0, sizeof(double) * pRBM->m_hNum * pRBM->m_vNum);
}


//CPool CCDLearner::s_pool;
#if 0
void CCDLearner::
Learning(vector<double *> &data,  double rate, int maxIter)
{
	fprintf(stderr, "CDLearning start:");
	const int LOG_PATH_LEN = 256;
	char buf[LOG_PATH_LEN];
	sprintf(buf, "logs/cdLog_bHidden_%d_h_%d_iter_%d.log", 
			CConfig::bReconBinHidden, m_pRBM->m_hNum, maxIter);
	FILE *fpLog = fopen(buf, "w");

	double initMomentum = 0.5;
	double finalMomentum = 0.9;
	for (int iter = 0; iter < maxIter; ++iter)
	{
		clock_t start = clock();

		fprintf(stderr, "Iteration %d \r", iter);
		fprintf(fpLog, "\n\n\nIteration: %d\n ", iter);

		// perform CD training
		vector<double *> randomData = shuffleData(data);
		for (size_t idx = 0; idx < randomData.size(); ++idx)
		{
			if (idx > 0 && idx % 1000 == 0)
				fprintf(stderr, "Iteration %d CDLearning processing %lu samples\r", iter, idx);
			SRBM::s_pool.Recycle();
			UpdateCD1(randomData[idx], m_pRBM->m_vNum, rate, iter > 5 ? initMomentum : finalMomentum);
		}
		
		// computing reconstruction Error
		double reError = 0.0, llLoss = 0.0;
		vector<double> vecV1, vecP1;
		for (int idx = 0; idx < (int) data.size(); ++idx)
		{
			m_pRBM->Reconstruct(data[idx], m_pRBM->m_vNum, vecP1, vecV1);

			// for debuging
			if (m_verbose == true && iter % 4 == 0)
			{
				fprintf(fpLog, "\ninput:\n");
				CMnistPrinter::PrintPicture(data[idx], 0.5e-3, fpLog);
				fprintf(fpLog, "\nrecon(probs):\n");
				CMnistPrinter::PrintPicture(&vecP1[0], 0.1, fpLog);
				fprintf(fpLog, "\n-------------------------\n");
			}
			
			for (int v = 0; v < m_pRBM->m_vNum; ++v)
			{
				reError += (vecV1[v]- data[idx][v]) * (vecV1[v]- data[idx][v]);
				llLoss  += (vecP1[v]- data[idx][v]) * (vecP1[v]- data[idx][v]); 
			}
		}

		fprintf(stderr, "reconstruct error:%.4f, llLoss %.4f, time cost %.2f secs\n",
						reError,  llLoss,  1.0 * (clock() - start)/CLOCKS_PER_SEC);
		
		fprintf(fpLog, "%.4f, %.4f, %.2f\n",
						reError,  llLoss,  1.0 * (clock() - start)/CLOCKS_PER_SEC);
		
	}

	fclose(fpLog);
	fprintf(stderr, "CDLearning done\n");
}
#endif


double CCDLearner::
MiniBatchUpdate(vector<double* >:: iterator iterBeg,
								vector<double* >:: iterator iterEnd,
								double rate, 		double momentum)
{
	memset(m_gVB, 0, sizeof(double) * m_pRBM->m_vNum);
	memset(m_gHB, 0, sizeof(double) * m_pRBM->m_hNum);
	memset(m_gW, 0, sizeof(double) * m_pRBM->m_hNum * m_pRBM->m_vNum);

	double error = 0.0;
	int batchSize = iterEnd - iterBeg;
//	fprintf(stderr, "batchSize %d \r", batchSize);
	for (vector<double *>:: iterator iter = iterBeg; iter != iterEnd ;++iter)
	{
		double *vP0 = &(*iter)[0];
		// 1. sampling h0|v0
		vector<double> vQ0 , vH0 , vP1, vV1, vQ1, vH1;
		m_pRBM->Sampling(vP0,  m_pRBM->m_vNum,  vQ0,  vH0,  SRBM::H_GIVEN_V);
		m_pRBM->Sampling(&vH0[0],  m_pRBM->m_hNum,  vP1,  vV1,  SRBM::V_GIVEN_H);
		m_pRBM->Sampling(&vP1[0],  m_pRBM->m_vNum,  vQ1,  vH1,  SRBM::H_GIVEN_V);

		for (int h = 0; h < m_pRBM->m_hNum; ++h)
			m_gHB[h] += vQ0[h] - vQ1[h];
	
		for (int v = 0; v < m_pRBM->m_vNum; ++v)
		{
			error += (vP0[v] - vP1[v]) * (vP0[v] - vP1[v]);
			m_gVB[v] += vP0[v] - vP1[v];
						
			for (int h = 0; h < m_pRBM->m_hNum; ++h)
				m_gW[v * m_pRBM->m_hNum + h] += vP0[v]*vQ0[h] - vP1[v]*vQ1[h];
		}
	}
				
	scalarMult(m_gVB, 	m_pRBM->m_vNum, 	rate/batchSize);
	scalarMult(m_gHB, 	m_pRBM->m_hNum, 	rate/batchSize);
	scalarMult(m_gW,  m_pRBM->m_hNum * m_pRBM->m_vNum, 	rate/batchSize);
	
	// update RBM
	scalarMult(m_pVBiasInc, 	m_pRBM->m_vNum, 	momentum);
	vecInc(m_pVBiasInc, 	m_gVB, 	m_pRBM->m_vNum);
	vecInc(m_pRBM->m_pVB, 	m_pVBiasInc, 	m_pRBM->m_vNum);
	
	scalarMult(m_pHBiasInc, 	m_pRBM->m_hNum, 	momentum);
	vecInc(m_pHBiasInc, 	m_gHB, 	m_pRBM->m_hNum);
	vecInc(m_pRBM->m_pHB, 	m_pHBiasInc, 	m_pRBM->m_hNum);

	scalarMult(m_pWInc, 	m_pRBM->m_vNum * m_pRBM->m_hNum,	momentum);
	vecInc(m_pWInc,		m_gW, 	m_pRBM->m_vNum * m_pRBM->m_hNum);
				
	for (int v = 0; v < m_pRBM->m_vNum; ++v)
		vecInc(m_pRBM->m_ppW[v], m_pWInc + v * m_pRBM->m_hNum, m_pRBM->m_hNum);
	
	return error;
}




double CCDLearner::
MiniBatchUpdate(vector<vector<double> >:: iterator iterBeg,
								vector<vector<double> >:: iterator iterEnd,
								double rate, 		double momentum)
{
	memset(m_gVB, 0, sizeof(double) * m_pRBM->m_vNum);
	memset(m_gHB, 0, sizeof(double) * m_pRBM->m_hNum);
	memset(m_gW, 0, sizeof(double) * m_pRBM->m_hNum * m_pRBM->m_vNum);

	double error = 0.0;
	int batchSize = iterEnd - iterBeg;
//	fprintf(stderr, "batchSize %d \r", batchSize);
	for (vector<vector<double>>:: iterator iter = iterBeg; 
				iter != iterEnd ;++iter)
	{
		double *vP0 = &(*iter)[0];
		// 1. sampling h0|v0
		vector<double> vQ0 , vH0 , vP1, vV1, vQ1, vH1;
		m_pRBM->Sampling(vP0,  m_pRBM->m_vNum,  vQ0,  vH0,  SRBM::H_GIVEN_V);
		m_pRBM->Sampling(&vH0[0],  m_pRBM->m_hNum,  vP1,  vV1,  SRBM::V_GIVEN_H);
		m_pRBM->Sampling(&vP1[0],  m_pRBM->m_vNum,  vQ1,  vH1,  SRBM::H_GIVEN_V);

		for (int h = 0; h < m_pRBM->m_hNum; ++h)
			m_gHB[h] += vQ0[h] - vQ1[h];
	
		for (int v = 0; v < m_pRBM->m_vNum; ++v)
		{
			error += (vP0[v] - vP1[v]) * (vP0[v] - vP1[v]);
			m_gVB[v] += vP0[v] - vP1[v];
						
			for (int h = 0; h < m_pRBM->m_hNum; ++h)
				m_gW[v * m_pRBM->m_hNum + h] += vP0[v]*vQ0[h] - vP1[v]*vQ1[h];
		}
	}
				
	scalarMult(m_gVB, 	m_pRBM->m_vNum, 	rate/batchSize);
	scalarMult(m_gHB, 	m_pRBM->m_hNum, 	rate/batchSize);
	scalarMult(m_gW,  m_pRBM->m_hNum * m_pRBM->m_vNum, 	rate/batchSize);
	
	// update RBM
	scalarMult(m_pVBiasInc, 	m_pRBM->m_vNum, 	momentum);
	vecInc(m_pVBiasInc, 	m_gVB, 	m_pRBM->m_vNum);
	vecInc(m_pRBM->m_pVB, 	m_pVBiasInc, 	m_pRBM->m_vNum);
	
	scalarMult(m_pHBiasInc, 	m_pRBM->m_hNum, 	momentum);
	vecInc(m_pHBiasInc, 	m_gHB, 	m_pRBM->m_hNum);
	vecInc(m_pRBM->m_pHB, 	m_pHBiasInc, 	m_pRBM->m_hNum);

	scalarMult(m_pWInc, 	m_pRBM->m_vNum * m_pRBM->m_hNum,	momentum);
	vecInc(m_pWInc,		m_gW, 	m_pRBM->m_vNum * m_pRBM->m_hNum);
				
	for (int v = 0; v < m_pRBM->m_vNum; ++v)
		vecInc(m_pRBM->m_ppW[v], m_pWInc + v * m_pRBM->m_hNum, m_pRBM->m_hNum);
	
	return error;
}



double CCDLearner::
UpdateCD1(double *pIn, int inLen, double rate, double momentum)
{
	if (m_pRBM == NULL)
	{
		fprintf(stderr, "Error: learning uninitialized rbm\n");
		exit(0);
	}

	// let Q denote p(h|v), P denote p(v|h)
	int hNum = m_pRBM->m_hNum;
	int vNum = m_pRBM->m_vNum;
	
	// 1. sampling h0|v0
	vector<double> vQ0 , vH0 ;
	m_pRBM->Sampling(pIn,  inLen,  vQ0,  vH0,  SRBM::H_GIVEN_V);

	// 2. sampling v1|h0
	vector<double> vP1, vV1;
	bool reconBySample = true;
	m_pRBM->Sampling(reconBySample ? &vH0[0]:&vQ0[0], hNum,  vP1,  vV1,  SRBM::V_GIVEN_H);

	// 3. smapling h1|v1
	vector<double> vQ1, vH1;
	m_pRBM->Sampling(&vP1[0],  vNum,  vQ1,  vH1,  SRBM::H_GIVEN_V);

	// update parameters
	double llLoss = 0;
	for (int i = 0; i < vNum; ++i)
	{
		m_pVBiasInc[i] = momentum * m_pVBiasInc[i] + rate * (pIn[i] - vP1[i]);// pVSamples1[i]);
		m_pRBM->m_pVB[i] += m_pVBiasInc[i];
		for (int j = 0; j < hNum; ++j)
		{
			m_pWInc[i*hNum + j] = momentum * m_pWInc[i * hNum + j] + rate * (pIn[i] * vQ0[j] - vP1[i] * vQ1[j]);
			m_pRBM->m_ppW[i][j] += m_pWInc[i * hNum + j];
		}

		llLoss += (pIn[i] - vP1[i]) * (pIn[i] - vP1[i]);
	}

	for (int j = 0; j < hNum; ++j)
	{
		m_pHBiasInc[j] = momentum * m_pHBiasInc[j] + rate *(vQ0[j] - vQ1[j]);
		m_pRBM->m_pHB[j] += m_pHBiasInc[j];
	}

	return llLoss;
}



