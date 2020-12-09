/*****************************************
 * filename:    DSEvidenceFuse.c
 * author  :    Seafood
 * discribe:    使用改进D-S证据理论融合证据
 * ******************************************/

#include "DSEvidenceFuse.h"

float trustDiscountCalculate(float network_acc_item, int *relat_5_frame_item);          //信任折扣计算

float resetTheBPA(float bpa_evidence_item,float trust_discount_item);			//重置bpa

float kconflictCalculate(int num_focalelement, float *mass_evidence1,float *mass_evidence2);                //计算冲突程度

/**
*@name：DSEvidenceTheoryInterface()
*@return:void
*@function：DS证据理论接口函数
*@para：int num_evidence, 证据的数量，即传感器数量 int num_focalelement焦元数量int num_framerelate:相关帧数量*bpa_evidence:输入证据，二维数组 float *network_acc:网络正确率 *relat_5_frame:近五帧输出
*其他要注意的地方
**/
int dsEvidenceTheoryInterface(int num_evidence, int num_focalelement, int num_framerelate, float bpa_evidence[num_evidence][num_focalelement],float network_acc[num_evidence], int relat_5_frame[num_evidence][num_framerelate])
{
    //根据信任折扣重新分配bpa基本概率分配
    float* trust_discount = (float*)malloc(num_evidence*sizeof(float));         //信任折扣
    float** new_bpa = (float**)malloc(num_evidence*sizeof(float*));        //申请空间bpa
    for(int i = 0; i< num_evidence; i++){
        new_bpa[i]=(float*)malloc(num_focalelement*sizeof(float));
    }

    //信任折扣计算和bpa重新分布
    for(int i=0;i<num_evidence;i++){
        trust_discount[i]=trustDiscountCalculate(network_acc[i], relat_5_frame[i]);     //信任折扣确定
        for(int j = 0; j< num_focalelement; j++){
            new_bpa[i][j] = resetTheBPA(bpa_evidence[i][j],trust_discount[i]);      //新概率分布计算
        }
    }

    //融合过程
    //融合结果存储
    float* mass_tmp = (float*)malloc(sizeof(float)*num_focalelement);       //申请空间
    //将new[0]放入mass_tmp[0]
    for(int i = 0; i< num_focalelement; i++){
        mass_tmp[i]=new_bpa[0][i];
    }
    //循环进行融合
    for(int item = 0 ; item< num_evidence-1; item++){
        float k = kconflictCalculate(num_focalelement,mass_tmp,new_bpa[item+1]);
        float con_mass = 1/(1-k);
        for(int j = 0; j< num_focalelement; j++){
            mass_tmp[j]=con_mass*mass_tmp[j]*new_bpa[item+1][j];            //计算融合mass函数
        }
    }
    int max_mass_index=0;               //最大位置的数字，即分类结果
    for(int i = 0; i < num_focalelement; i++){
        if(mass_tmp[i]>mass_tmp[max_mass_index]){
            max_mass_index = i;
        }
    }
    //释放空间
    free(mass_tmp);
    free(trust_discount);
    free(new_bpa);

    return max_mass_index;

}

/**
*@name：trustDiscountCalculate()
*@return:float
*@function：信任折扣确定准则
*@para：float network_acc_item,网络正确率 int *relat_5_frame_item  近五帧识别结果
*其他要注意的地方
**/
float trustDiscountCalculate(float network_acc_item, int relat_5_frame_item[5])
{
    int num_relatedframe = sizeof(relat_5_frame_item);  //相关帧数
    int count_sameframe=0;      //计算相同帧
    //不足五帧返回准确率
    if(relat_5_frame_item[0]==-1){
        return network_acc_item;
    }
    //计算相同帧数量
    for(int i = 0; i < num_relatedframe-1; i++){
        if(relat_5_frame_item[i] == relat_5_frame_item[num_relatedframe-1]){
            count_sameframe++;
        }
    }
    //比例
    float rate_lastframe_all = (float)count_sameframe/num_relatedframe;
    //返回准确率
    //信任折扣确定准则
    if(rate_lastframe_all - 0.8 >= 0){
        return 0.95;
    }
    else if(rate_lastframe_all - 0.4 <= 0){
        return 0.8*network_acc_item;
    }
    else{
        return network_acc_item;
    }
}

/**
*@name：resetTheBPA()
*@return:float
*@function：根据信任折扣重置bpa
*@para：bpa_evidence_item:概率分布,trust_discount_item：信任折扣
*其他要注意的地方
**/
float resetTheBPA(float bpa_evidence_item,float trust_discount_item)
{
    return bpa_evidence_item*trust_discount_item + 0.5 * (1-bpa_evidence_item)*(1-trust_discount_item);
}


/**
*@name：kconflictCalculate()
*@return:float
*@function：计算两证据之间的冲突程度
*@para：int num_focalelement：焦元数量, float *mass_evidence1：证据1的mass函数,float *mass_evidence2：证据2的mass函数
*其他要注意的地方
**/
float kconflictCalculate(int num_focalelement, float *mass_evidence1,float *mass_evidence2)
{
    float sum=0;
    for(int i = 0; i< num_focalelement; i++){
        for(int j = 0; j < num_focalelement; j++){
            if(i != j){
                sum = sum+ mass_evidence1[i]*mass_evidence2[j];
            }
        }
    }
    return sum;
}

void main(int argc, char **argvs)
{
    printf("%d\n",dsEvidenceTheoryInterface(num_evidence, num_focalelement, num_framerelate, bpa_evidence, network_acc, relate_5_frame));

}
