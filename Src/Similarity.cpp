#include <queue>
#include <cmath>
#include "WRRBM.h"


using std::pair;
class my_greater
{
public: 
	bool operator ()(const pair<double, string> &l, const pair<double, string> &r)
	{
		return l.first < r.first;
	}
};

double Dis(double *l, double *r, int len)
{
	double sum = 0;
	for (int i = 0; i < len; ++i)
	{
		sum += (l[i] - r[i]) * (l[i] - r[i]);
	}

	return sqrt(sum);
}



void Similarity(const char *prefix)
{
	CWordIDMap widMap;
	widMap.LoadDict((string(prefix) + string(".dict")).c_str());
	
	SWRRBM wrRBM;
	if(wrRBM.Load((string(prefix) + string(".model")).c_str()) == false)
		fprintf(stderr, "loading model failed\n");

	wrRBM.CheckNan();

	// normalizing vectors
	int wDim = wrRBM.m_wDim;
//	for (int i = 0; i < wrRBM.m_vcbSize; ++i)
//		normalize(wrRBM.m_D + i * wDim, wDim);

	std::priority_queue<pair<double, string>, vector<pair<double, string> >, my_greater> heap;
	
	for (int k = 0; k < wrRBM.m_vcbSize; ++k)
	{
		double *pK = wrRBM.m_D + k * wDim;
		for (int i = 0; i < wrRBM.m_vcbSize; ++i)
		{
			if (k == i)
				continue;
			double *pI = wrRBM.m_D + i * wDim;
			double score = Dis(pK,  pI, wDim);

			if (false && score < 1.0)
			{
				fprintf(stderr, "%s, %s\n", 
								widMap.GetWord(k).c_str(), widMap.GetWord(i).c_str());
				disVec(pK, wDim);
				disVec(pI, wDim);
			}
//			double score = dotProduct(pK,  wDim,  pI,  wDim);

			// update heap
			if (heap.size() < 10)
				heap.push(std::pair<double, string> (score, widMap.GetWord(i)));
			else if (heap.top().first > score)
			{
				heap.pop();
				heap.push(std::pair<double, string> (score, widMap.GetWord(i)));
			}

		}

		fprintf(stdout, "%s----\n", widMap.GetWord(k).c_str());
		while (heap.size() > 0)
		{
			fprintf(stdout, "\t%s:%.4f\n", 
							heap.top().second.c_str(), 
							heap.top().first);

			heap.pop();
		}
		fprintf(stdout, "------------------------------\n");
	}
}


void usage(const char *pszName)
{
	fprintf(stderr, "usage: %s  <modelPrefix>\n", pszName);
	exit(0);
}



int main(int argc, const char ** argv)
{
	if (argc != 2)
		usage(*argv);
	Similarity(argv[1]);
	return 0;
}
