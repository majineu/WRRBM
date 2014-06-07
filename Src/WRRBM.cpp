#include <cassert>
#include <algorithm>
#include <map>
#include "WRRBM.h"
using std::map;
SWRRBM::SWRRBM()
{
	m_nGram = m_wDim = m_hNum = m_vcbSize = m_nBurnIn = 0;
	m_pHB = m_pVB = m_D = NULL;
	m_pMap = NULL;
	m_verbose = false;
}

void SWRRBM::
ReAllocate()
{
	int oldSize = m_vcbSize;
	m_vcbSize = m_pMap->size();

	double *oldPVB = m_pVB;
	m_pVB = (double *) m_pool.Allocate(sizeof(double) * m_vcbSize);
	memcpy(m_pVB,  oldPVB,  sizeof(double) *oldSize);
	for (int i = oldSize; i < m_vcbSize; ++i)
		m_pVB[i] = log(m_pGen->GetProb(i));

	double *oldD = m_D;
	m_D = (double *) m_pool.Allocate(sizeof(double) * m_vcbSize * m_wDim);
	memset(m_D, 0, sizeof(double) * m_vcbSize * m_wDim);
	for (int i = 0; i < oldSize; ++i)
		memcpy(m_D + i * m_wDim,  oldD + i *m_wDim,  sizeof(double) *m_wDim); 
}

SWRRBM::
SWRRBM(int left, int right, int hSize, 
			 int wDim, int burnIn, CWordIDMap *pMap, 
			 CRandomGen *pGen,  bool verbose)
: m_nLeft(left), 		m_nRight(right), 
	m_nGram(left + right), 	 m_hNum(hSize), 
	m_wDim(wDim),   m_nBurnIn(burnIn), 
	m_pMap(pMap),   m_pGen(pGen), 
	m_verbose(verbose)
{
	m_vcbSize = (int)m_pMap->size();
	assert(m_nGram > 0);
	assert(m_hNum  > 0);
	assert(m_wDim  > 0);
	assert(m_vcbSize > 0);
	fprintf(stderr, "Create WRRBM: vcbSize %d, ngram %d, hNum %d, wDim %d\n",
					m_vcbSize, m_nGram, m_hNum , m_wDim);

	m_pHB = (double *) m_pool.Allocate(sizeof(double) * m_hNum);
	memset(m_pHB, 0, sizeof(double) * m_hNum);

	// all positions share the same visuable bias
	m_pVB = (double *) m_pool.Allocate(sizeof(double) * m_vcbSize);
	for (int i = 0; i < m_vcbSize; ++i)
		m_pVB[i] = log(m_pGen->GetProb(i));
//	memset(m_pVB, 0, sizeof(double) * m_vcbSize);
	
	// embedding
	m_D = (double *) m_pool.Allocate(sizeof(double) * m_vcbSize * m_wDim);
//	memset(m_D, 0, sizeof(double) * m_vcbSize * m_wDim);
	for (int v = 0; v < m_vcbSize; ++v)
		for (int d = 0; d < m_wDim; ++d)
			m_D[v * wDim + d] = uniform(-0.1, 0.1);
	
	// position dependent weight matrix
	fprintf(stderr, "Init weight matrix ...");
	for (int i = 0; i < m_nGram; ++i)
	{
		double *pW = (double *) m_pool.Allocate(sizeof(double) * m_hNum * m_wDim);
		for (int h = 0; h < m_hNum; ++h)
			for (int d = 0; d < m_wDim; ++d)
				pW[h * m_wDim + d] = uniform(-0.1, 0.1);

		m_vW.push_back(pW);
	}
	fprintf(stderr, "done\n");
}


void SWRRBM::
CheckNan()
{
//	fprintf(stderr, "model checking ...");
	for (int i = 0; i < m_hNum; ++i)
		if (isnan(m_pHB[i]) == true)
		{
			fprintf(stderr, "HB[%d]: %.2f\n", i, m_pHB[i]);
			fgetc(stderr);
		}
	
	for (int i = 0; i < m_vcbSize; ++i)
		if (isnan(m_pVB[i]) == true)
		{
			fprintf(stderr, "VB[%d]: %.2f\n",i, m_pVB[i]);
			fgetc(stderr);
		}

	for (int i = 0; i < m_vcbSize * m_wDim; ++i)
		if (isnan(m_D[i]) == true)
		{
			fprintf(stderr, "D[%d][%d]: %.2f\n", i / m_wDim, i % m_wDim, m_D[i]);
			fgetc(stderr);
		}

	for (int i = 0; i < m_nGram; ++i)
	{
		for (int j = 0; j < m_hNum * m_wDim; ++j)
			if (isnan(m_vW[i][j]) == true)
			{
				fprintf(stderr, "w[%d][%d][%d]: %.2f\n", i, j / m_wDim, j % m_wDim, m_vW[i][j]); 
				fgetc(stderr);
			}
	}
//	fprintf(stderr, "done\n");
}

void SWRRBM::
Display(FILE *fp)
{
	fprintf(fp, "----------model parameters--------\n");

	fprintf(fp, "vbias:\n");
	for (int i = 0; i < m_vcbSize; ++i)
	{
		fprintf(fp, "%.2e ", m_pVB[i]);
		if ((i + 1) % 10 == 0)
			fprintf(fp, "\n");
	}


	fprintf(fp, "\nhBias:\n");
	for (int i = 0; i < m_hNum; ++i)
	{
		fprintf(fp, "%.2e ", m_pHB[i]);
		if ((i + 1) % 10 == 0)
			fprintf(fp, "\n");
	}


	fprintf(fp, "\nembedding:\n");
	for (int i = 0; i < m_vcbSize; ++i)
	{
		if (i % 100 == 0)
		{
			for (int d = 0; d < m_wDim; ++d)
				fprintf(fp, "%.2e ", m_D[i * m_wDim + d]);
			fprintf(fp, "\n");
		}
	}


	fprintf(fp, "\nweight:\n");
	for (size_t i = 0; i < m_vW.size(); ++i)
	{
		fprintf(fp, "\nweight %lu:\n", i);
		for (int j = 0; j < m_hNum; ++j)
		{
			for (int d = 0; d < m_wDim; ++d)
				fprintf(fp, "%.2e ", m_vW[i][j * m_wDim + d]);
			fprintf(fp, "\n");	
		}
	}
	fprintf(fp, "\n");
}


bool SWRRBM::
Save(FILE *fp)
{
	fwrite(&m_nLeft,   sizeof(m_nLeft), 1, fp);
	fwrite(&m_nRight,  sizeof(m_nRight), 1, fp);
	fwrite(&m_nGram,   sizeof(m_nGram), 1, fp);
	fwrite(&m_hNum,    sizeof(m_hNum), 1, fp);
	fwrite(&m_wDim,    sizeof(m_wDim), 1, fp);
	fwrite(&m_vcbSize, sizeof(m_vcbSize), 1, fp);
	fwrite(&m_nBurnIn, sizeof(m_nBurnIn), 1, fp);

	fwrite(m_pVB, sizeof(double), m_vcbSize, fp);
	fwrite(m_pHB, sizeof(double), m_hNum, fp);
	fwrite(m_D, sizeof(double), m_vcbSize * m_wDim, fp);

	for (int i = 0; i < m_nGram; ++i)
		fwrite(m_vW[i], sizeof(double), m_hNum * m_wDim, fp);

	return true;
}



bool SWRRBM::
Load(FILE *fp)
{
	fread(&m_nLeft,   sizeof(m_nLeft), 	 1, fp);
	fread(&m_nRight,  sizeof(m_nRight),  1, fp);
	fread(&m_nGram, 	sizeof(m_nGram), 	 1, fp);
	fread(&m_hNum, 		sizeof(m_hNum), 	 1, fp);
	fread(&m_wDim, 		sizeof(m_wDim), 	 1, fp);
	fread(&m_vcbSize, sizeof(m_vcbSize), 1, fp);
	fread(&m_nBurnIn, sizeof(m_nBurnIn), 1, fp);
	fprintf(stderr, "Loading WRRBM left %d, right %d, ngram %d, hNum %d, wDim %d, vcbSize %d, burnIn %d\n",
					m_nLeft, m_nRight, m_nGram, m_hNum, m_wDim, m_vcbSize, m_nBurnIn);


	m_pVB = (double*) m_pool.Allocate(sizeof(double) * m_vcbSize);
	m_pHB = (double*) m_pool.Allocate(sizeof(double) * m_hNum);
	m_D  =  (double*) m_pool.Allocate(sizeof(double) * m_vcbSize * m_wDim);
	
	fread(m_pVB, sizeof(double), m_vcbSize, fp);
	fread(m_pHB, sizeof(double), m_hNum, fp);
	fread(m_D, sizeof(double), m_vcbSize * m_wDim, fp);

	for (int i = 0; i < m_nGram; ++i)
	{
		double *pW = (double*)m_pool.Allocate(sizeof(double) * m_hNum * m_wDim);
		fread(pW, sizeof(double), m_hNum * m_wDim, fp);
		m_vW.push_back(pW);
	}

	return true;
}



void SWRRBM::
H_Given_V(vector<const char *> &ngram, vector<double> &hProbs, vector<double> &hSamples)
{
	assert((int)ngram.size() == m_nGram);
	vector<int> ids(ngram.size());
	
	// ngram to word ID
	for (size_t i = 0; i < ngram.size(); ++i)
	{
		ids[i] = m_pMap->GetID(ngram[i]);
#if 0
		if (iter != m_pDict->end())
			ids[i] = iter->second;
		else
			ids[i] = (*m_pDict)["<unk>"];
#endif
	}

	H_Given_V(ids, hProbs, hSamples);
}


void SWRRBM:: 
H_Given_V(vector<int> &idx, 
					vector<double> &hProbs, 
					vector<double> &hSamples)
{
	if ((int)idx.size() != m_nGram)
	{
		fprintf(stderr, "Error: input ngram length %d vs m_nGram %d\n",
						(int)idx.size(), m_nGram);
		exit(0);
	}

	if (false && m_verbose == true)
		fprintf(stderr, "\nH_Given_V-----------\n");

	hProbs.resize(m_hNum);
	hSamples.resize(m_hNum);

	for (int j = 0; j < m_hNum; ++j)
	{
		double sumDotProd = 0;
		for (int i = 0; i < m_nGram; ++i)
		{
			// the j-th row of the i-th weight matrix
			double *pU  = m_vW[i];
			double *pUj = &pU[j * m_wDim];						// j-th row: D * 1
			
			// the embedding of the i-th word
			double *pEmbedding = &m_D[idx[i] * m_wDim]; 	// embedding of the i-th word: D * 1
			sumDotProd += dotProduct(pEmbedding, 	m_wDim, 	pUj, 	m_wDim);

			if (false && m_verbose == true)
			{
				fprintf(stderr, "U%d_%d      :",i, j);
				disVec(pUj, m_wDim, stderr);
				fprintf(stderr, "embedding %d:", idx[i]);
				disVec(pEmbedding, m_wDim, stderr);
			}
		}
		double activate = m_pHB[j] + sumDotProd;
		
		hProbs[j] = sigmoid(activate);
		hSamples[j] = uniform(0, 1) < hProbs[j];
		
		if (false && m_verbose == true)
		{
			fprintf(stderr, "%d, HB :%.2e, sumProd: %.2e, act %.2e, p: %.2e, sample %d\n\n", 
						j, 	m_pHB[j], 	sumDotProd, activate,  hProbs[j],  (int)hSamples[j]);
		}
	}
	
	if (false && m_verbose == true)
		fprintf(stderr, "------------------------\n\n");
}


double SWRRBM::
ComputeActivate(int wid, int position, vector<double> &hSamples)
{
	double *pEmbedding = &m_D[wid * m_wDim];
	vector<double> UD(m_hNum, 0);						// a H*1 vector, is there a more efficient way?
	double *Ui = m_vW[position];										// weight matrix of the is position
	for (int h = 0; h < m_hNum; ++h)
	{
		UD[h] = dotProduct(&Ui[h * m_wDim], m_wDim, pEmbedding, m_wDim);
		stopNan(UD[h]);
	}	
	double bias = m_pVB[wid];
	double activate = bias + dotProduct(&UD[0], UD.size(), &hSamples[0], m_hNum);

	static int hit = 0;
	++hit;
	if (m_verbose == true && hit % 50 == 0)
	{
		fprintf(stderr, "bias + hUDe: %-4.2e + %-4.2e\n", 
						bias,   dotProduct(&UD[0], UD.size(), &hSamples[0], m_hNum));
	}
	return activate;
}


void SWRRBM::
V_Given_H(vector<double> &hSamples, vector<int> &vOld,
					vector<double> &vProbs, vector<int> &vNew,
					vector<bool> *pVHit)														// (*pVHit)[i] checks whether vOld[i] has been visited
{
	if ((int)hSamples.size() != m_hNum)
	{
		fprintf(stderr, "Error: hSamples length %lu vs m_hNum %d\n", hSamples.size(), m_hNum);
		exit(0);
	}

	vProbs.resize(m_nGram);
	vNew.resize(m_nGram);
	if (pVHit != NULL)
		pVHit->resize(m_nGram, false);

	vector<int> temp = vOld;

#if 0
	for (int i = 0;  i < m_nGram; ++i)
	{
		temp[i] = m_pGen->Sample();
		if (pVHit != NULL)
			(*pVHit)[i] = temp[i] == vOld[i];
	}
#endif

	for (int i = 0; i < m_nGram; ++i)
	{
		map<int, double> activateCache;
		if (m_verbose == true)
			fprintf(stderr, "\nposition %d\n", i);

		for (int step = 0; step < m_nBurnIn; ++step)				// burn-in
		{
			int widOld = temp[i], widNew = m_pGen->Sample();
			if (widNew == widOld)
				continue;
			
			if (pVHit != NULL)
				(*pVHit)[i] = (*pVHit)[i] || widNew == vOld[i];

			double probOld = m_pGen->GetProb(widOld);
			double probNew = m_pGen->GetProb(widNew);
			double actOld = 0.0, 	actNew = 0.0;
			map<int, double> ::iterator iter = activateCache.find(widOld);
			if (iter != activateCache.end())
				actOld = iter->second;
			else
			{
				actOld = ComputeActivate(widOld, i, hSamples);
				activateCache[widOld] = actOld;
			}

			iter = activateCache.find(widNew);
			if (iter != activateCache.end())
				actNew = iter->second;
			else
			{
				actNew = ComputeActivate(widNew, i, hSamples);
				activateCache[widNew] = actNew;
			}
	
			double transProb = std::min<double>(1.0,  probOld / probNew * exp(actNew - actOld));
			vProbs[i] = transProb;
			if (uniform(0, 1) < transProb)
			{
				temp[i] = widNew;
				if ((widNew == vOld[i] || widOld == vOld[i]) 
						&& m_verbose == true)
				{
					fprintf(stderr, "\nw:%s, bias %.2e, sumprod %.2e, act %.2e |||| ",
							m_pMap->GetWord(widOld).c_str(), 		m_pVB[widOld],
							actOld - m_pVB[widOld],  	actOld);
				
					fprintf(stderr, "w:%s, bias %.2e, sumprod %.2e, act %.2e\n",
							m_pMap->GetWord(widNew).c_str(), 		m_pVB[widNew],
							actNew - m_pVB[widNew],  	actNew);
					
					fprintf(stderr, "old %s, prob %-8.2e, act %-8.2e |||new %s, prob %-8.2e, act %-8.2e  expRatio %.2e  transProb %-8.4e\n",
									m_pMap->GetWord(widOld).c_str(),  probOld,  actOld,  
									m_pMap->GetWord(widNew).c_str(),  probNew,  actNew, 
									exp(actNew - actOld), transProb);
				}
			}
		}
	}

	std::copy(temp.begin(), temp.end(), vNew.begin());
	if (m_verbose == true)
	{
		fprintf(stderr, "-----------final----------\n");
		for (int i = 0; i < m_nGram; ++i)
			fprintf(stderr, "%s ", m_pMap->GetWord(vOld[i]).c_str());
		fprintf(stderr, " -> ");
		for (int i = 0; i < m_nGram; ++i)
			fprintf(stderr, "%s ", m_pMap->GetWord(vNew[i]).c_str());
		fprintf(stderr, "\n");
	}
}



SWRRBM * SWRRBM:: RBMFromFile(const char *prefix)
{
	CWordIDMap *widMap = new CWordIDMap();
	widMap->LoadDict((string(prefix) + string(".dict")).c_str());
	
	// initializing random generator
	CRandomGen *rGen = new CRandomGen();
	rGen->LoadGen((string(prefix) + string(".random")).c_str());
	
	SWRRBM *wrRBM = new SWRRBM();
	if(wrRBM->Load((string(prefix) + string(".model")).c_str()) == false)
		fprintf(stderr, "loading model failed\n");

	wrRBM->m_pGen = rGen;
	wrRBM->m_pMap = widMap;
	return wrRBM;
}


void SWRRBM::
GetEmbeddingBound(double &rMin,  double &rMax,  bool normalized)
{
	rMin = rMax = m_D[0];

	// do the normalization
	if (normalized == true)
	{
		double maxLen = 0, minLen = 1.7e308;
		rMin = 1.7e208, 		rMax = -1.7e308;
		for (int wid = 0; wid < m_vcbSize; ++wid)
		{
			double length = 0;
			for (int d = 0; d < m_wDim; ++d)
				length += m_D[wid * m_wDim + d] * m_D[wid * m_wDim + d];
			length = sqrt(length);

			if (maxLen < length)
				maxLen = length;

			if (minLen > length)
				minLen = length;

			if (length < 1.0e-4)
			{
				fprintf(stderr, "wid %-4d, length %-8.2e, embedding:\n",
						wid, length);
				disVec(m_D + wid * m_wDim, m_wDim);
			}
	
			for (int d = 0; d < m_wDim; ++d)
			{
				m_D[wid * m_wDim + d] /= length;
				double val = m_D[wid * m_wDim + d];
				if (rMin > val)
					rMin = val;
	
				if (rMax < val)
					rMax = val;
			}
		}

		fprintf(stderr, "minLen %.5f, maxLen %.5f\n",
						minLen, maxLen);
	}
	else
	{
		for (int idx = 1; idx < m_vcbSize * m_wDim; ++idx)
		{
			if (m_D[idx] < rMin)
				rMin = m_D[idx];
			else if (m_D[idx] > rMax)
				rMax = m_D[idx];
		}
	}
	

	// make spans
	int nSpan = 10;
	vector<double> span(nSpan), counts(nSpan, 0);
	for (size_t i = 0; i < span.size(); ++i)
		span[i] = rMin + i * (rMax - rMin)/span.size();


	// counting
	for (int idx = 0; idx < m_vcbSize * m_wDim; ++idx)
	{
		for (size_t k = 0; k < span.size() - 1; ++k)
			if (m_D[idx] >= span[k] && m_D[idx] <= span[k + 1])
			{
				counts[k] ++;
				break;
			}

		if (m_D[idx] > span[span.size() - 1])
			counts[span.size() - 1] ++;
	}
	

	fprintf(stderr, "total %d\n", m_vcbSize * m_wDim);
	for (size_t k = 0; k < counts.size() - 1; ++k)
		fprintf(stderr, "[%-4.2f ~ %-4.2f]:%-6.0f, %-6.2f%%\n", 
						span[k],    span[k + 1],
						counts[k],  100.0 * counts[k]/(m_vcbSize * m_wDim));

	fprintf(stderr, "[%-6.2f ~ %-6.2f]:%-6.0f, %-6.2f%%\n", 
					span[span.size() - 1],    rMax,
					counts[span.size() - 1],  100.0 * counts[span.size() - 1]/(m_vcbSize * m_wDim));
}


// whether the embeddings are normalized or not
void SWRRBM::
QuantizationEmbedding(int n, double lower, double upper)
{
	assert(n > 0);

	fprintf(stderr, "Quantization with n: %d....", n);
	vector<double> codebook(n + 1, 0.0);
	double step = (upper - lower)/n;
	for (int i = 0; i < n; ++i)
		codebook[i] = lower + step * i;
	codebook[codebook.size() - 1] = upper;

	for (int w = 0; w < m_vcbSize; ++w)
	{
		double *p = m_D + w * m_wDim;
		for (int d = 0; d < m_wDim; ++d)
		{
			if (p[d] < codebook[0] || p[d] > codebook.back())
				fprintf(stderr, "Quantization Error:%f out of range\n", p[d]);
			
			for (int c = 0; c < n; ++c)
				if (p[d] >= codebook[c] && p[d] <= codebook[c+1])
				{
//					fprintf(stderr, "%.2e ~[%.2e, %.2e]\n", 
//							p[d], codebook[c], codebook[c+1]);
					p[d] = c;
					break;
				}
		}
	}
	fprintf(stderr, "Done\n");
}


