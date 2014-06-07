#include <vector>
#include <cstdio>
using std::vector;

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
		fprintf(stderr, "mid %d, prob %.3f\n",
				mid, probs[mid + 1]);
		if (target < probs[mid])
			end = mid - 1;
		else if (target > probs[mid + 1])
			beg = mid + 1;
		else
			return mid;
	}

	return -1;
}


int main()
{
	vector<double> seq;
	for (int i = 0; i < 100; ++i)
		seq.push_back(i * 0.1);

	fprintf(stderr, "range: %.30e ~ %.30e\n", seq.front(), seq.back());

	double f = 0.0;
	while(scanf("%lf", &f) == 1)
	{
		fprintf(stderr, "input: %.30f\n", f);
		int res = biSearch(seq, f);
		if (res != -1)
			fprintf(stderr, "%d, [%.30f ~ %.30f]\n",
							res, seq[res], seq[res + 1]);
		else
			fprintf(stderr, "-1\n");
	}
	return 0;
}
