/*******************************************************************************
 *
 *      Filename:  WRRBMNeural.h
 *
 *       Version:  1.0
 *       Created:  2013-11-29 16:34:53
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


#ifndef __W_R_R_B_M_NEURAL_H__
#define __W_R_R_B_M_NEURAL_H__
#include "WRRBM.h"
class CWRRBMNeural
{
public:
	int	Feedfoword(int *wids, int ngram, double *pOut);


private:
	CWRRBM 	*m_pRBM;
};

#endif  /*__W_R_R_B_M_NEURAL_H__*/
