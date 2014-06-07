/*******************************************************************************
 *
 *      Filename:  SimCos.cpp
 *
 *       Version:  1.0
 *       Created:  2013-11-08 09:31:44
 *      Revision:  none
 *      Compiler:  g++
 *
 *        Author:  Ao Zhang (NLP-LAB), zhangao.cs@gmail.com
 *  Organization:  NEU, China
 *
 *   Description:
 *                 
 *
 ******************************************************************************/

#include <queue>
#include <cmath>
#include "WRRBM.h"

struct SNode
{
	double first;
	string second;
	double	len;
	SNode(double d, string s, int l):
		first(d), second(s), len(l) {}
};



using std::pair;
class my_greater
{
public: 
//	bool operator ()(const pair<double, string> &l, const pair<double, string> &r)
//	{
//		return l.first > r.first;
//	}
	bool operator ()(const SNode &l, const SNode &r)
	{
		return l.first > r.first;
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

//	std::priority_queue<pair<double, string>, vector<pair<double, string> >, my_greater> heap;
	std::priority_queue<SNode, vector<SNode>, my_greater> heap;
	
	for (int k = 0; k < wrRBM.m_vcbSize; ++k)
	{
		double *pK = wrRBM.m_D + k * wDim;
		double kLen = vecLen(pK, wDim);
		for (int i = 0; i < wrRBM.m_vcbSize; ++i)
		{
			if (k == i)
				continue;
			double *pI = wrRBM.m_D + i * wDim;
			double iLen = vecLen(pI, wDim);
//			double score = Dis(pK,  pI, wDim);
			double score = dotProduct(pK,  wDim,  pI,  wDim)/kLen / iLen;

			if (false && score < 1.0)
			{
				fprintf(stderr, "%s, %s\n", 
								widMap.GetWord(k).c_str(), widMap.GetWord(i).c_str());
				disVec(pK, wDim);
				disVec(pI, wDim);
			}

			// update heap
			if (heap.size() < 30)
				heap.push(SNode (score, widMap.GetWord(i), iLen));
				//heap.push(std::pair<double, string> (score, widMap.GetWord(i)));
			else if (heap.top().first < score)
			{
				heap.pop();
				heap.push(SNode (score, widMap.GetWord(i), iLen));
				//heap.push(std::pair<double, string> (score, widMap.GetWord(i)));
			}
		}

		fprintf(stdout, "%s len %.2f----\n", widMap.GetWord(k).c_str(), kLen);
		while (heap.size() > 0)
		{
			fprintf(stdout, "\t%s:%.4f, %.2f\n", 
							heap.top().second.c_str(), 
							heap.top().first,
							heap.top().len);

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
