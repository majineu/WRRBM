#include <vector>
#include <ctime>
#include <thread>
#include <cassert>
#include "WRRBM.h"

using std::vector;
using std::thread;

int wDim = 0;

struct SDNode				// data node
{
	double *pData;
	double  distance;
	int cid;
};

struct SCNode				// cluster node
{
	SCNode() {m_pPos = 0; m_nSize = 0;}
	double *m_pPos;
	int m_nSize;
};

//--------------------------------------------------------------------------------
void threadFuncMinDis(vector<SDNode>::iterator beg,
											vector<SDNode>::iterator end,
											const vector<SCNode> & vClusters,
											double *pTotalDis)
{
	vector<SDNode>::iterator iter;
	*pTotalDis = 0.0;
//	fprintf(stderr, "Data size %lu, cluster number %lu\n", end - beg, vClusters.size());
	for (iter = beg; 	iter != end; 	++iter)
	{
		iter->distance = 1.e308;
		iter->cid = -1;
		for  (size_t i = 0; i < vClusters.size(); ++i)
		{
			double dis = Dis(vClusters[i].m_pPos, iter->pData, wDim);
			if (dis < iter->distance)
			{
				iter->distance = dis;
				iter->cid	= i;
			}
		}

		*pTotalDis += iter->distance;
	}
}


//--------------------------------------------------------------------------------
int UpdateCluster(vector<SDNode> & vData, vector<SCNode> & vClusters)
{
//	fprintf(stderr, "Updating clusters ... ");
	for (size_t i = 0; i < vClusters.size(); ++i)
	{
		SCNode *pCNode = &vClusters[0] + i;
		memset(pCNode->m_pPos, 0, sizeof(double) * wDim);
		pCNode->m_nSize = 0;
	}

	for (size_t i = 0; i < vData.size(); ++i)
	{
		if (vData[i].cid < 0 || vData[i].cid > (int)vClusters.size())
		{
			fprintf(stderr, "Error: cid %d of data %lu out of range\n", vData[i].cid, i);
			exit(0);
		}
		SCNode *pCNode = &vClusters[0] + vData[i].cid;
		pCNode->m_nSize ++;
		for (int d = 0; d < wDim; ++d)
			pCNode->m_pPos[d] += vData[i].pData[d];
	}

	int nEmpty = 0;
	for (size_t i = 0; i < vClusters.size(); ++i)
	{
		SCNode *pCNode = &vClusters[0] + i;
		if (pCNode->m_nSize == 0)
			nEmpty ++;
		else
		{
			for (int d = 0; d < wDim; ++d)
				pCNode->m_pPos[d] /= pCNode->m_nSize;
		}
	}
	return nEmpty;
}


//--------------------------------------------------------------------------------
int biSearch(vector<double> &probs, double target)
{
	if (probs.size() == 0)
		return -1;
	if (target < probs[0] || target > probs.back())
		return -1;

	int beg = 0, end = (int)probs.size() - 1;
	while (beg <= end)
	{
		int mid = (beg + end) >> 1;
		if (target < probs[mid])
			end = mid - 1;
		else if (target > probs[mid + 1])
			beg = mid + 1;
		else
			return mid;
	}

	return -1;
}


//--------------------------------------------------------------------------------
void KPPInit(vector<SDNode> &data, vector<SCNode> &cluster, int k)
{
	int id = std::rand() % data.size();
	memcpy(cluster[0].m_pPos, data[id].pData, sizeof(double) * wDim);
	cluster[0].m_nSize = 1;

	vector<double> probVec(data.size() + 1, 0.0);
	vector<double> disVec(data.size(), 0.0);
	fprintf(stderr, "K-means ++ initializing..\r");

	for (size_t d = 0; d < disVec.size(); ++d)
	{
		double dis = Dis(cluster[0].m_pPos, data[d].pData, wDim);
		disVec[d] = dis * dis;
	}
	
	for (size_t c = 1; c < cluster.size(); ++c)
	{
		// re-compute probability
//		fprintf(stderr, "K-means ++ initializing cluster %lu\n", c);
		double total = accumulate(disVec.begin(), disVec.end(), 0.0);
		for (size_t d = 0; d < disVec.size(); ++d)
			probVec[d + 1] = disVec[d] / total + probVec[d];


		// sampling
		double sProb = uniform(0.0, 1.0);
		int dID = biSearch(probVec, sProb);
		memcpy(cluster[c].m_pPos, data[dID].pData, sizeof(double) * wDim);

//		fprintf(stderr, "cluster %lu set to data point %d\n", c, dID);
		
		if (c == cluster.size() - 1)
			break;

		// update distance
		for (size_t d = 0; d < disVec.size(); ++d)
		{
			double dis = Dis(cluster[c].m_pPos, data[d].pData, wDim);
			dis *= dis;
			if (disVec[d] < dis)
				disVec[d] = dis;
		}
	}

	fprintf(stderr, "K-means ++ initializing cluster done\n");
}

void OutputRes(const char *pszPath, vector<SDNode> & data, int k,
							 SWRRBM *pRBM)
{
	FILE *fpOut = fopen(pszPath, "w");
	assert(fpOut);
	vector<vector<int> > widVec(k);
	for (size_t i = 0; i < data.size(); ++i)
		widVec[data[i].cid].push_back(i);

	for (size_t i = 0; i < widVec.size(); ++i)
	{
		for (size_t w = 0; w < widVec[i].size(); ++w)
			fprintf(fpOut, "%s ", pRBM->m_pMap->GetWord(widVec[i][w]).c_str());
		fprintf(fpOut, "\n");
	}

	fclose(fpOut);
}


//--------------------------------------------------------------------------------
void KMeans(const char *pszPath, int k, int nEpoch, const char *pszOutPath)
{
	SWRRBM *pRBM = SWRRBM::RBMFromFile(pszPath);
	wDim = pRBM->m_wDim;
	vector<SDNode> dVec(pRBM->m_vcbSize);
	fprintf(stderr, "Init data points of size %lu\n", dVec.size());
	for (size_t i = 0; i < dVec.size(); ++i)
	{
		dVec[i].pData = &pRBM->m_D[i * wDim];
		dVec[i].cid 	= -1;
		dVec[i].distance = 1.7e308;
	}

	vector<SCNode> cVec(k);
	for (size_t i = 0; i < cVec.size(); ++i)
	{
		SCNode *pNode = &cVec[0] + i;
		pNode->m_nSize = 0;
		pNode->m_pPos = (double *) malloc(sizeof(double) * wDim);
		memset(pNode->m_pPos, 0, sizeof(double) * wDim);
	}


	// initialize with k-means ++
	KPPInit(dVec, cVec, k);


	int nThread = 5, nConverge = 0;
	double disLast = -1.0;
	// here we go....
	for (int epoch = 0; epoch < nEpoch; ++epoch)
	{
		vector<thread> threads;
		vector<double> disVec(nThread, 0.0);

		time_t start, now;
		time(&start);
		size_t span = dVec.size() / nThread;
		for (int i = 0; i < nThread - 1; ++i)
			threads.push_back(thread(threadFuncMinDis, 
															 dVec.begin() + i *span,
															 dVec.begin() + (i + 1) * span,
															 cVec,		&disVec[0] + i));
			
		threads.push_back(thread(threadFuncMinDis, 
														 dVec.begin() + (nThread - 1) *span,
														 dVec.end(), 
														 cVec,	&disVec[0] + nThread - 1));

		for (size_t i = 0; i < threads.size(); ++i)
			threads[i].join();


		int nEmptyCluster = UpdateCluster(dVec, cVec);
		double totalDis = std::accumulate(disVec.begin(), disVec.end(), 0.0);
	
		if (disLast >= 0)
		{
			if (fabs(disLast - totalDis) < 1.0e-3)
			{
				if (++nConverge > 5)
					break;
			}
			else
				nConverge = 0;
		}

		time(&now);
		fprintf(stderr, "epoch %d, nConverge %d, nEmpty %d, dis %.4f, time %.4f\n",
						epoch, nConverge, nEmptyCluster,	totalDis,  difftime(now, start));
		disLast = totalDis;
	}

	for (size_t i = 0; i < cVec.size(); ++i)
		delete [] cVec[i].m_pPos;

	OutputRes(pszOutPath, dVec, k, pRBM); 
}


void usage(const char *name)
{
	fprintf(stderr, "usage: %s  <WRRBM>  k epoch  <resPath>\n", name);
	exit(0);
}

int main(int argc, char **argv)
{
	if (argc != 5)
		usage(*argv);
	
	KMeans(argv[1], atoi(argv[2]), atoi(argv[3]), argv[4]);
	return 0;
}
