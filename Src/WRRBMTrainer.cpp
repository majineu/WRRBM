#include <cmath>
#include "WRRBMTrainer.h"

using std::thread;

CWRRBMTrainer::
CWRRBMTrainer(SWRRBM *pRBM, int nThreads, bool verbose)
:m_pRBM(pRBM), m_nThread(nThreads), m_verbose(verbose)
{
	m_gHB 	= new double[pRBM->m_hNum];
	m_incHB = new double[pRBM->m_hNum];
	memset(m_gHB, 	0, sizeof(double) * pRBM->m_hNum);
	memset(m_incHB, 0, sizeof(double) * pRBM->m_hNum);


	m_gVB 	= new double[pRBM->m_vcbSize];
	m_incVB = new double[pRBM->m_vcbSize];
	memset(m_gVB, 	0, sizeof(double) * pRBM->m_vcbSize);
	memset(m_incVB, 0, sizeof(double) * pRBM->m_vcbSize);
	
	int vcbSize = pRBM->m_vcbSize;
	m_gD   = new double[vcbSize * pRBM->m_wDim];
	m_incD = new double[vcbSize * pRBM->m_wDim];
	memset(m_gD,  	0, sizeof(double) * vcbSize * pRBM->m_wDim);
	memset(m_incD,  0, sizeof(double) * vcbSize * pRBM->m_wDim);
	

	for (int i = 0; i < m_pRBM->m_nGram; ++i)
	{
		double *pW = new double[pRBM->m_hNum * pRBM->m_wDim];
		memset(pW,  0, sizeof(double) * pRBM->m_hNum * pRBM->m_wDim);
		m_gW.push_back(pW);
		
		pW = new double[pRBM->m_hNum * pRBM->m_wDim];
		memset(pW,  0, sizeof(double) * pRBM->m_hNum * pRBM->m_wDim);
		m_incW.push_back(pW);
	}
	m_sparseVec.resize(pRBM->m_vcbSize, false);
}


CWRRBMTrainer::
~CWRRBMTrainer()
{
	delete [] m_gHB;
	delete [] m_incHB;

	delete [] m_gVB;
	delete [] m_incVB;

	delete [] m_gD;
	delete [] m_incD;

	for (int i = 0; i < m_pRBM->m_nGram; ++i)
	{
		delete[] m_gW[i];
		delete[] m_incW[i];
	}
}

void CWRRBMTrainer::
threadFunc(vector<vector<int> >::iterator beg,
					 vector<vector<int> >::iterator end, 
					 double *pError, 	double *pHit)
{
	*pError = 0;
	int wDim = m_pRBM->m_wDim;
	double *localVB = new double[m_pRBM->m_vcbSize];
	double *localHB = new double[m_pRBM->m_hNum];
	double *localD  = new double[m_pRBM->m_vcbSize * wDim];
	memset(localVB, 0, sizeof(double) * m_pRBM->m_vcbSize);
	memset(localHB, 0, sizeof(double) * m_pRBM->m_hNum);
	memset(localD,  0, sizeof(double) * m_pRBM->m_vcbSize * wDim);

	vector<double *> localW; 
	for (size_t i = 0; i < m_gW.size(); ++i)
	{
		localW.push_back(new double[m_pRBM->m_hNum * wDim]);	
		memset(localW[i], 0, sizeof(double) * m_pRBM->m_hNum * wDim);
	}
	
	for (vector<vector<int> >::iterator iter = beg; iter != end; ++iter)
	{
		// V, P denotes the samples and probilities of the visuables
		// H, Q denotes that of the hidden units
		vector<int> &vV0 = *iter, vV1;
		vector<double> vH0, vQ0, vH1, vQ1, vP1;
		vector<bool> vHit;

		// Sampling
		m_pRBM->H_Given_V(vV0, vQ0, vH0);
		m_pRBM->V_Given_H(vH0, vV0, vP1, vV1, &vHit);		// sampling visuable using binary hidden state 
		bool failed = false;
		for (size_t i = 0; i < vV0.size(); ++i)
		{
			*pHit += vHit[i];
			if (vV0[i] != vV1[i])
				failed = true;
		}
		if (failed == false)
			continue;

		m_pRBM->H_Given_V(vV1, vQ1, vH1);
		
		// for U(i)_jd : suppose vV0[i] = e(d) i.e. the d-th word
		for (size_t i = 0; i < vV0.size(); ++i)
		{
			if (m_verbose == true)
				fprintf(stderr, "\nposition %lu:\n", i);
			double *gU = localW[i];// *pU = m_pRBM->m_pW[i];//, 			*gU  = m_gW[i];
			double *pDk0 = &m_pRBM->m_D[vV0[i] * wDim],  *pDk1 = &m_pRBM->m_D[vV1[i] * wDim];
			for (int h = 0; h < m_pRBM->m_hNum; ++h)
			{
				for (int d = 0; d < wDim; ++d)
				{
					gU[h * wDim + d] += vQ0[h] * pDk0[d] - vQ1[h] * pDk1[d];
					if (m_verbose == true && (h*wDim + d) % 10000 == 0)
					{
						fprintf(stderr, "vQ0[%d] %.2e, pDK0[%d] %.2e,     vQ1[%d] %.2e, pDk1[%d] %.2e, inc %.2e    ",
										h, vQ0[h], d, pDk0[d], h, vQ1[h], d, pDk1[d],  vQ0[h] * pDk0[d] - vQ1[h] * pDk1[d]);
						fprintf(stderr, "gU[%d] %.2e\n",  h*wDim + d,  gU[h*wDim + d]);
					}
				}
			}

			// compute reconstruction error
			if (vV0[i] != vV1[i])
			{
				*pError += 1;
				failed = true;
			}
		}

		if (m_verbose && ((int)*pError % 10) == 0 && failed == true)
		{
			for (size_t i = 0; i < vV0.size(); ++i)
				fprintf(stderr, "%s ", m_pRBM->m_pMap->GetWord(vV0[i]).c_str());
			
			fprintf(stderr, "   ->   ");
			for (size_t i = 0; i < vV1.size(); ++i)
				fprintf(stderr, "%s ", m_pRBM->m_pMap->GetWord(vV1[i]).c_str());
		}


		// update embedding parameters: matrix D 
		for (size_t i = 0; i < vV0.size(); ++i)
		{
			double *pU = m_pRBM->m_vW[i];
			double *gDk0 = &localD[vV0[i] * wDim],  *gDk1 = &localD[vV1[i] * wDim];

			for (int h = 0; h < m_pRBM->m_hNum; ++h)
			{
				for (int d = 0; d < wDim; ++d)
				{
					gDk0[d] += vQ0[h] * pU[h * wDim + d];
					gDk1[d] -= vQ1[h] * pU[h * wDim + d];
				}
			}

			// also update vbias
			localVB[vV0[i]] += 1.0;
			localVB[vV1[i]] -= 1.0;
		}

		// for hBias
		for (int h = 0; h < m_pRBM->m_hNum; ++h)
			localHB[h] += vQ0[h] - vQ1[h];
//			localHB[h] += vH0[h] - vH1[h];
	}

	// for the shared data
	m_biasMtx.lock();
	for (int i = 0; i < m_pRBM->m_hNum; ++i)
		m_gHB[i] += localHB[i];

	for (int i = 0; i < m_pRBM->m_vcbSize; ++i)
		m_gVB[i] += localVB[i];
	m_biasMtx.unlock();

	// for embedding
	m_DMtx.lock();
	for (int i = 0; i < m_pRBM->m_vcbSize * wDim; ++i)
		m_gD[i] += localD[i];
	m_DMtx.unlock();

	// for weight matrix
	m_WMtx.lock();
	for (int i = 0; i < (int)localW.size(); ++i)
	{
		double *lW = localW[i], *gW = m_gW[i];
		for (int h = 0; h < m_pRBM->m_hNum ; ++h)
			for (int d = 0; d < wDim; ++d)
				gW[h * wDim + d] += lW[h*wDim + d];
	}
	m_WMtx.unlock();


	delete[] localVB;
	delete[] localHB;
	delete[] localD;

	for (size_t i = 0; i < localW.size(); ++i)
		delete[] localW[i];
}

std::pair<double, double> CWRRBMTrainer::
MiniBatchUpdateMt(vector<vector<int> >:: iterator beg,  
									vector<vector<int> >:: iterator end,
									double rate,  double biasRate,	double momentum, 
									double l1Reg, double l2Reg)
{
	int wDim = m_pRBM->m_wDim;
	
	// reset gradients
	memset(m_gVB, 0, sizeof(double) * m_pRBM->m_vcbSize);
	memset(m_gHB, 0, sizeof(double) * m_pRBM->m_hNum);
	memset(m_gD,  0, sizeof(double) * m_pRBM->m_vcbSize * wDim);

	for (size_t i = 0; i < m_gW.size(); ++i)
		memset(m_gW[i], 0, sizeof(double) * m_pRBM->m_hNum * wDim);

	std::fill(m_sparseVec.begin(), m_sparseVec.end(), false);
	

	int nDataPerThread = (end - beg)/m_nThread;
	vector<std::thread> vThreads;
	vector<double> vError(m_nThread, 0.0);
	vector<double> vHit(m_nThread, 0.0);
	for (int i = 0; i < m_nThread - 1; ++i)
		vThreads.push_back(std::thread(&CWRRBMTrainer::threadFunc, this,  
																		beg + i * nDataPerThread, 
																		beg + (i + 1) * nDataPerThread,   
																		&vError[0] + i,
																		&vHit[0] + i));

	vThreads.push_back(std::thread(&CWRRBMTrainer::threadFunc, this,  
							beg + (m_nThread - 1) * nDataPerThread,  end,   
							&vError[0] + m_nThread - 1,
							&vHit[0] + m_nThread - 1));

	for (int i = 0; i < (int)vThreads.size(); ++i)
		vThreads[i].join();

//	fprintf(stderr, "All threads done\n");
	int batchSize = end - beg;
	updateParameter(rate, biasRate, batchSize, momentum, l1Reg, l2Reg);

	double totalError = std::accumulate(vError.begin(), vError.end(), 0.0);
	double totalHit   = std::accumulate(vHit.begin(), vHit.end(), 0.0);
	return std::pair<double, double>(totalError,  totalHit);
}


double
CWRRBMTrainer::
MiniBatchUpdate(vector<vector<int> >:: iterator beg,  vector<vector<int> >:: iterator end,
							  double rate, double biasRate, double momentum, double l1Reg, double l2Reg)
{
	int wDim = m_pRBM->m_wDim;
	
	// reset gradients
	memset(m_gVB, 0, sizeof(double) * m_pRBM->m_vcbSize);
	memset(m_gHB, 0, sizeof(double) * m_pRBM->m_hNum);
	memset(m_gD,  0, sizeof(double) * m_pRBM->m_vcbSize * wDim);

	for (size_t i = 0; i < m_gW.size(); ++i)
		memset(m_gW[i], 0, sizeof(double) * m_pRBM->m_hNum * wDim);

	std::fill(m_sparseVec.begin(), m_sparseVec.end(), false);


	double error = 0;
	for (vector<vector<int> >::iterator iter = beg; iter != end; ++iter)
	{
		// V, P denotes the samples and probilities of the visuables
		// H, Q denotes that of the hidden units
		vector<int> &vV0 = *iter, vV1;
		vector<double> vH0, vQ0, vH1, vQ1, vP1;

		// Sampling
		if (m_verbose == true)
			fprintf(stderr, "\nsampling h0 given v0\n");
		m_pRBM->H_Given_V(vV0, vQ0, vH0);
		m_pRBM->V_Given_H(vH0, vV0, vP1, vV1);		// sampling visuable using binary hidden state 
		
		if (m_verbose == true)
			fprintf(stderr, "\nsampling h1 given v1\n");
		m_pRBM->H_Given_V(vV1, vQ1, vH1);

		bool failed = false;
		// for U(i)_jd : suppose vV0[i] = e(d) i.e. the d-th word
		for (size_t i = 0; i < vV0.size(); ++i)
		{
			double *gU = m_gW[i];// *pU = m_pRBM->m_pW[i];//, 			*gU  = m_gW[i];
			double *pDk0 = &m_pRBM->m_D[vV0[i] * wDim],  *pDk1 = &m_pRBM->m_D[vV1[i] * wDim];
			for (int h = 0; h < m_pRBM->m_hNum; ++h)
			{
				// process positive example
				for (int d = 0; d < wDim; ++d)
					gU[h * wDim + d] += vQ0[h] * pDk0[d] - vQ1[h] * pDk1[d];
			}

			// compute reconstruction error
			if (vV0[i] != vV1[i])
			{
				error += (vV0[i] != vV1[i]);
				failed = true;
			}
		}

		if (false && ((int)error % 10) == 0 && failed == true)
		{
			for (size_t i = 0; i < vV0.size(); ++i)
				fprintf(stderr, "%s ", m_pRBM->m_pMap->GetWord(vV0[i]).c_str());
			
			fprintf(stderr, "   ->   ");
			for (size_t i = 0; i < vV1.size(); ++i)
				fprintf(stderr, "%s ", m_pRBM->m_pMap->GetWord(vV1[i]).c_str());
		}


		// update embedding parameters: matrix D 
		for (size_t i = 0; i < vV0.size(); ++i)
		{
			double *pU = m_pRBM->m_vW[i];
			double *gDk0 = &m_gD[vV0[i] * wDim],  *gDk1 = &m_gD[vV1[i] * wDim];
			m_sparseVec[vV0[i]] = true;
			m_sparseVec[vV1[i]] = true;

			for (int h = 0; h < m_pRBM->m_hNum; ++h)
			{
				for (int d = 0; d < wDim; ++d)
				{
					gDk0[d] += vQ0[h] * pU[h * wDim + d];
					gDk1[d] -= vQ1[h] * pU[h * wDim + d];
				}
			}

			// also update vbias
			m_gVB[vV0[i]] += 1.0;
			m_gVB[vV1[i]] -= 1.0;
		}

		// for hBias
		for (int h = 0; h < m_pRBM->m_hNum; ++h)
			m_gHB[h] += vQ0[h] - vQ1[h];

		if (m_verbose == true)
		{
			fprintf(stderr, "\nvQ0:\n");
			disVec(&vQ0[0], m_pRBM->m_hNum > 10 ? 10: m_pRBM->m_hNum, stderr);
			fprintf(stderr, "vQ1:\n");
			disVec(&vQ1[0], m_pRBM->m_hNum > 10 ? 10: m_pRBM->m_hNum, stderr);
			fprintf(stderr, "gradient HB:\n");
			disVec(m_gHB, m_pRBM->m_hNum > 10 ? 10: m_pRBM->m_hNum, stderr);
		}
	}

	int batchSize = end - beg;
	updateParameter(rate, biasRate, batchSize, momentum, l1Reg, l2Reg);
	if (m_verbose == true)
	{
		fprintf(stderr, "\nsparse recoder\n:");
		for (int i = 0; i < (int)m_sparseVec.size(); ++i)
			if (m_sparseVec[i] == true)
				fprintf(stderr, "%d ", i);
		fprintf(stderr, "\n");
	}

	return error;// = error / batchSize * m_pRBM->m_nGram;
//	return error;
}



void CWRRBMTrainer::
updateParameter(double rate, 		double biasRate, 
								int batchSize, 	double momentum, 
								double l1, 			double l2)
{
	rate /= batchSize;
	biasRate /= batchSize;

	if (m_verbose == true)
		fprintf(stderr, "\n\nupdate parameter rate %.2e bRate %.2e\n", 
						rate, biasRate);

	// H bias
	for (int h = 0; h < m_pRBM->m_hNum; ++h)
	{
		m_incHB[h] = m_incHB[h] * momentum + m_gHB[h] * rate;///batchSize;
		m_pRBM->m_pHB[h] += m_incHB[h] - l2 * m_pRBM->m_pHB[h];
		
		if (isnan(m_pRBM->m_pHB[h]) == true)
		{
			fprintf(stderr, "momentum %.2e, incHB %.2e  gHB %.2e  rate %.2e\n",
							momentum,		m_incHB[h], 	m_gHB[h], 	rate);
			fgetc(stdin);
		}
	}

	// V bias
//	biasRate;
	if (m_verbose == true)
		fprintf(stderr, "\nvisuable bias\n");

	for (int d = 0; d < m_pRBM->m_vcbSize; ++d)
	{
		m_incVB[d] = m_incVB[d] * momentum + m_gVB[d] * biasRate;/// batchSize;
		m_pRBM->m_pVB[d] += m_incVB[d] - l2 * m_pRBM->m_pVB[d];
		
		if (m_verbose == true && d % 1000 == 0)
			fprintf(stderr, "gB[%d] %.2e, B_old[%d] %.2e, B_new[%d] %.2e\n",
				d,	m_gVB[d], 	d, 	m_pRBM->m_pVB[d] - m_incVB[d], 	
				d, 	m_pRBM->m_pVB[d]);
	}

	// weight matrix
	int wDim = m_pRBM->m_wDim;
	for (size_t i = 0; i < m_gW.size(); ++i)
	{
		if (m_verbose == true)
			fprintf(stderr, "\nposition %lu\n", i); 	
		
		double *incW = m_incW[i], *gW = m_gW[i], *pW = m_pRBM->m_vW[i];
		for (int h = 0; h < m_pRBM->m_hNum; ++h)
		{
			for (int d = 0; d < m_pRBM->m_wDim; ++d)
			{
				int idx = h * wDim + d;
				incW[idx] = incW[idx] * momentum + gW[idx] * rate;/// batchSize;
				pW[idx] += incW[idx];// - l2 * pW[idx];
				if (m_verbose == true && idx % 300 == 0)
					fprintf(stderr, "gW[%d] %.2e, W_old[%d] %.2e, W_new[%d] %.2e\n",
							idx, gW[idx], idx, pW[idx] - incW[idx], idx, pW[idx]);
			}
		}
	}

	
	// word embedding:  |v| * d matrix
	if (m_verbose == true)
		fprintf(stderr, "\nupdate embedding\n");
	
	for (int i = 0; i < m_pRBM->m_vcbSize; ++i)
	{
		for (int d = 0; d < wDim; ++d)
		{
			int idx = i * wDim + d;
			m_incD[idx] = m_incD[idx] * momentum + m_gD[idx] * rate;///batchSize;
			m_pRBM->m_D[idx] += m_incD[idx];// - l2 * m_pRBM->m_D[idx];
			
			if (m_verbose == true && idx % 300 == 0)
				fprintf(stderr, "gD[%d] %.2e, D_old[%d] %.2e, D_new[%d] %.2e\n",
					idx, 			m_gD[idx], 			idx, 
					m_pRBM->m_D[idx] - m_gD[idx] * biasRate, 	idx, 	m_pRBM->m_D[idx]);
		}
	}

	m_pRBM->CheckNan();
}

























