#ifndef __DSEVIDENCEFUSE_H__
#define __DSEVIDENCEFUSE_H__
#include <stdio.h>
#include <stdlib.h>

int num_evidence = 2;       //证据数量即声音和震动两通道
int num_focalelement = 3;   //焦元数量  即分类结果
int num_framerelate = 5;        //5帧  取相近5帧结果为参考，-1为无

float bpa_evidence[2][3]={0.2,0.3,0.5,0.2,0.4,0.4};   //bpa分布
float network_acc[2] = {0.9,0.8};                     //网络准确率
int relate_5_frame[2][5] = {-1,-1,-1,1,1,1-1,-1,0,1,1};    //近5帧输入信号


int dsEvidenceTheoryInterface(int num_evidence, int num_focalelement, int num_framerelate, float bpa_evidence[num_evidence][num_focalelement],float network_acc[num_evidence], int relat_5_frame[num_evidence][num_framerelate]);   // DS证据理论函数

#endif
