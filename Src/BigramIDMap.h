#ifndef __EMBEDDING_DICT_H__
#define __EMBEDDING_DICT_H__
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <cassert>
#include "util.h"
#include "Pool.h"


using std::unordered_map;
using std::string;
typedef std::unordered_map<string, int> DICT_TYPE;
#define NUM_RESEARSE 3
class CWordIDMap
{
public:
	CWordIDMap()
	{
		m_vWord.push_back("<unk>");
		m_vWord.push_back("<s>");
		m_vWord.push_back("</s>");
		m_vCount.resize(NUM_RESEARSE, 1);
		for (size_t i = 0; i < m_vWord.size(); ++i)
			m_dict[m_vWord[i]] = i;

		m_vBigram.push_back("<unk>");
		m_vBiCounter.push_back(1);
		m_biDict[m_vBigram[0]] = 1;
	}

	int Inc(const string &word, bool inMode = true)
	{
		DICT_TYPE::iterator iter = m_dict.find(word);
		if (iter == m_dict.end())
		{
			if (inMode == true)
			{
				int id = (int)m_dict.size();
				m_dict[word] = id;
				m_vWord.push_back(word);
				m_vCount.push_back(1);
				return id;
			}
			else
				return m_dict["<unk>"];
		}
		else
		{
			m_vCount[iter->second] += 1;
			return iter->second;
		}
	}

	int IncBigram(const string &w1, const string &w2, bool inMode = true)
	{
		const string bigram(w1 + " " + w2);
		DICT_TYPE::iterator iter = m_biDict.find(bigram);
		if (iter == m_biDict.end())
		{
			if (inMode == true)
			{
				int id = (int)m_dict.size();
				m_biDict[bigram] = id;
				m_vBigram.push_back(bigram);
				m_vBiCounter.push_back(1);
				return id;
			}
			return m_biDict["<unk>"];
		}
		else
		{
			m_vBiCount[iter->second] += 1;
			return iter->second;
		}
	}


	
	void SaveDict(const char *pszPath)
	{
		FILE *fp = fopen(pszPath, "w");
		assert(fp);
		fprintf(fp, "%d\n", (int)m_vWord.size());
		for (size_t i = 0; i < m_vWord.size(); ++i)
			fprintf(fp, "%s %d\n", m_vWord[i].c_str(), m_vCount[i]);


		// save bigrams 
		for (size_t i = 0; i < m_biDict.size(); ++i)
			fprintf(fp, "%s %d\n", m_vBigram[i].c_str(), m_vBiCounter[i]);
		fclose(fp);
	}

	void LoadDict(const char *pszPath)
	{
		FILE *fp = fopen(pszPath, "r");
		assert(fp);
		char buf[65536], key[1024];
		int count;
		DICT_TYPE:: iterator iter = m_dict.end();
		
		int nUnigram = 0, lineId = 0;
		fgets(buf, 65535, fp);
		sscanf(buf, "%d", &nUnigram);
		fprintf(stderr, "total %d unigrams\n", nUnigram);
		
		while (fgets(buf, 65535, fp) != NULL)
		{
			lineId ++;
			for (int i = 0; i < (int)strlen(buf); ++i)
				if (buf[i] == '\r' || buf[i] == '\n')
				{
					buf[i] = 0;
					break;
				}
			

			sscanf(buf, "%s %d", key, &count);
			
			if (lineId > nUnigram)
			{
				iter = m_biDict.find(key);
				if (iter == m_biDict.end())
				{
					m_biDict[key] = m_vBigram.size();
					m_vBigram.push_back(key);
					m_vBiCount.push_back(count);
				}
				else
					m_vBiCount[iter->second] = count;
			}
			else
			{
				iter = m_dict.find(key);
				if (iter == m_dict.end())
				{
					m_dict[key] = m_vWord.size();
					m_vWord.push_back(key);
					m_vCount.push_back(count);
				}
				else
					m_vCount[iter->second] = count;
			}
		}
		fclose(fp);
	}


	void ExtractDictFromText(const char *pszDict)
	{
		char buf[65536];
		FILE *fp = fopen(pszDict, "r");
		if (fp == NULL)
		{
			fprintf(stderr, "Error: open %s failed\n", pszDict);
			exit(0);
		}
		int lineNum = 0;
		DICT_TYPE:: iterator iter = m_dict.end();
		bool insertMode = true;
		while (fgets(buf, 65535, fp) != NULL)
		{
			if (++lineNum % 10000 == 0)
				fprintf(stderr, "Processing %d line\r", lineNum);
		
			//fprintf(stderr, "len %-5d:%s\n", (int)strlen(buf), buf);
			vector<char *> words = Split(buf, " \r\t\n");
			for (size_t i = 0; i < words.size(); ++i)
			{
				Inc(words[i], insertMode);
				IncBigram(i>0 ? words[i - 1]:"<s>",
									words[i], 	insertMode)
			}
		}
		
		fprintf(stderr, "\ntotal %d line, dict size %lu, biDict size %lu\n", 
						lineNum,  m_dict.size(),  m_biDict.size());
		fclose(fp);
	}

	DICT_TYPE * GetDict()
	{
		return &m_dict;
	}
	
	DICT_TYPE * GetBiDict()
	{
		return &m_biDict;
	}

	int GetID(const string & key)
	{
		DICT_TYPE ::iterator iter = m_dict.find(key);
		if (iter != m_dict.end())
			return iter->second;

		// unk id is 0;
		return 0;
	}

	int GetBiGramID(const string &bigram)
	{
		DICT_TYPE::iterator iter = m_biDict.find(bigram);
		if (iter != m_biDict.end())
			return iter->second;

		// unk id is 0;
		return 0;
	}

	string GetWord(int id)
	{
		if (id < 0 || id > (int)m_dict.size())
		{
			fprintf(stderr, "word id out of range %d\n", id);
			exit(0);
		}
		return m_vWord[id];
	}

	string GetBigram(int id)
	{
		if (id < 0 || id > (int)m_biDict.size())
		{
			fprintf(stderr, "bigram id out of range %d\n", id);
			exit(0);
		}
		return m_vBigram[id];
	}

	void FilterBigram(int thres)
	{
		int nRemove = 0;
		fprintf(stderr, "filter bigram dict with threshold %d, ", thres);
		clock_t start = clock();
		for (size_t i = 1; i < m_vBiCounter.size(); ++i)
		{
			if (m_vBiCounter[i] <= thres)
			{
				++ nRemove;
				m_vBiCounter[i] = -1;//erase(m_vWord.begin() + i);
			}
		}
		fprintf(stderr, "total %d words removed\n", nRemove);
		
		// re-hash
		m_biDict.clear();
		int id = 0;
		for (size_t i = 0; i < m_vBiCounter.size(); ++i)
		{
			if (m_vBiCounter[i] > 0)
			{
				m_biDict[m_vBigram[i]] = id;
				m_vBigram[id] = m_vBigram[i];
				m_vBiCounter[id] = m_vBiCounter[i];
				++id;
			}
		}
		m_vBigram.resize(id);
		m_vBiCounter.resize(id);
		m_vBiCounter[m_biDict["<unk>"]] = nRemove / 10 > 0 ? nRemove/10 : 1;
		fprintf(stderr, "Filter bigram takes %.2f seconds\n", 
						1.0 * (clock() - start)/CLOCKS_PER_SEC);
	}

	void Filter(int thres)
	{
		int nRemove = 0;
		fprintf(stderr, "filter dict, threshold %d, ", thres);
		clock_t start = clock();
		for (size_t i = NUM_RESEARSE; i < m_vCount.size(); ++i)
		{
			if (m_vCount[i] <= thres)
			{
				++ nRemove;
				m_vCount[i] = -1;
			}
		}

		fprintf(stderr, "total %d words removed\n", nRemove);
		m_dict.clear();
		int id = 0;
		for (size_t i = 0; i < m_vWord.size(); ++i)
		{
			if (m_vCount[i] > 0)
			{
				m_dict[m_vWord[i]] = id;
				m_vWord[id] = m_vWord[i];
				m_vCount[id] = m_vCount[i];
				++id;
			}
		}
		m_vWord.resize(id);
		m_vCount.resize(id);
		m_vCount[m_dict["<unk>"]] = nRemove / 10 > 0 ? nRemove/10 : 1;
		fprintf(stderr, "Filter unigram takes %.2f seconds\n", 
						1.0 * (clock() - start)/CLOCKS_PER_SEC);
	}
	
	size_t size()
	{
		return m_vCount.size();
	}

	vector<int> & GetCount()
	{
		return m_vCount;
	}

	vector<int> & GetBigramSize()
	{
		return m_vBigram.size();
	}

	vector<int> & GetBigramCount()
	{
		return m_vBiCount;
	}

private:
	DICT_TYPE m_dict;
	vector<string> m_vWord;
	vector<int>  m_vCount;
	
	DICT_TYPE m_biDict;							// bigram Dict
	vector<string> m_vBigram;				
	vector<int>  m_vBiCounter;			// bigram counter
};


#endif  /*__EMBEDDING_DICT_H__*/
