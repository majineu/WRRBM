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
	}

	void SaveDict(const char *pszPath)
	{
		FILE *fp = fopen(pszPath, "w");
		assert(fp);
		for (size_t i = 0; i < m_vWord.size(); ++i)
			fprintf(fp, "%s\n", m_vWord[i].c_str());
		fclose(fp);
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

	void LoadDict(const char *pszPath)
	{
		FILE *fp = fopen(pszPath, "r");
		assert(fp);
		char buf[65536];
		DICT_TYPE:: iterator iter = m_dict.end();
		while (fgets(buf, 65535, fp) != NULL)
		{
			for (int i = 0; i < (int)strlen(buf); ++i)
			{
				if (buf[i] == '\r' || buf[i] == '\n')
				{
					buf[i] = 0;
					break;
				}
			}
			iter = m_dict.find(buf);
			if (iter == m_dict.end())
			{
				m_dict[buf] = m_vWord.size();
				m_vWord.push_back(buf);
			}
		}
		m_vCount.resize(m_vWord.size(), 0);
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
		while (fgets(buf, 65535, fp) != NULL)
		{
			if (++lineNum % 10000 == 0)
				fprintf(stderr, "Processing %d line\r", lineNum);
		
			//fprintf(stderr, "len %-5d:%s\n", (int)strlen(buf), buf);
			vector<char *> words = Split(buf, " \r\t\n");
			for (size_t i = 0; i < words.size(); ++i)
			{
				iter = m_dict.find(words[i]);
				//fprintf(stderr, "%s\n", words[i]);
				if (iter == m_dict.end())
				{
					m_dict[words[i]] = m_vWord.size();
					m_vWord.push_back(words[i]);
					m_vCount.push_back(1);
				}
				else
					m_vCount[iter->second] += 1;
			}
		}
		
		fprintf(stderr, "\ntotal %d line, dict size %lu\n", lineNum, m_dict.size());
		fclose(fp);
	}

	DICT_TYPE * GetDict()
	{
		return &m_dict;
	}

	int GetID(const string & key)
	{
		DICT_TYPE ::iterator iter = m_dict.find(key);
		if (iter != m_dict.end())
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

	void Filter(int thres)
	{
		int nRemove = 0;
		fprintf(stderr, "filter dict, threshold %d, ", thres);
		for (size_t i = NUM_RESEARSE; i < m_vCount.size(); ++i)
		{
			if (m_vCount[i] <= thres)
			{
				++ nRemove;
				m_vCount.erase(m_vCount.begin() + i);
				m_vWord.erase(m_vWord.begin() + i);
				--i;
			}
		}

		fprintf(stderr, "total %d words removed\n", nRemove);
		m_dict.clear();
		for (size_t i = 0; i < m_vWord.size(); ++i)
			m_dict[m_vWord[i]] = i;
		m_vCount[m_dict["<unk>"]] = nRemove / 10 > 0 ? nRemove/10 : 1;
	}
	
	size_t size()
	{
		return m_vCount.size();
	}
	vector<int> & GetCount()
	{
		return m_vCount;
	}

private:
	DICT_TYPE m_dict;
	vector<string> m_vWord;
	vector<int>  m_vCount;
};


#endif  /*__EMBEDDING_DICT_H__*/
