#ifndef __UTIL_H__
#define __UTIL_H__

#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<vector>
#include<algorithm>

using std::vector;

double *scalarMult(double *pV, int len, double scalar);
double dotProduct(double *pV1, int len1,  double *pV2, int len2);
void getCol(double **pMatrix, int rNum, int cNum, int cIdx, double *pRes);
void biSampling(double *pProbs, double *pValues, int len);
void biSampling(double *pProbs, vector<double> &rValues, int len);
void disVec(double *pVec, int len, FILE *fp = stderr);
void disSparseVec(double *pVec, int len, FILE *fp = stderr);
void vecInc(double *p1, double *p2, int len);
vector<char *> Split(char *pBuf, const char *pDelem);
void normalize(double *pVec, int len);
double vecLen(double *pVec, int len);
void disVec(double *p, double thres, double upperBound, 
						int window, int dim, FILE *fp);

double Dis(double *l, double *r, int len);
inline void stopNan(double val)
{
	if (isnan(val))
	{
		fprintf(stderr, "find nan\n");
		fgetc(stdin);
	}
}

inline double sigmoid(double x)
{
  return 1.0 / (1.0 + exp(-x));
}

template<class t>
inline vector<t> shuffleData(vector<t> & rVec)
{
	vector<t> res(rVec.begin(), rVec.end());
	std::random_shuffle(res.begin(), res.end());
	return res;
}

inline double uniform(double min, double max)
{
	return rand() / (RAND_MAX + 1.0) * (max - min) + min;
}

#if 0 
inline double binomial(int n, double p)
{
		double c = 0;
		if (p < 0 || p > 1)
			return 0;
		double r;

		for (int i = 0; i < n; i++)
		{
			r = rand() / (RAND_MAX + 1.0);
			if (r < p)
				c++;
		}
		return c;
}
#endif

#endif  /*__UTIL_H__*/
