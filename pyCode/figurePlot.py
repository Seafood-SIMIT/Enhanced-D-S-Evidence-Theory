#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:40:36 2020

@author: seafood
"""

import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl 
from matplotlib.font_manager import FontProperties
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/arphic/uming.ttc') 
#import matplotlib
import os
import MediumScaleModel
import UrbanSoundModel
import ReadData


# reright the data
def resetTheBpa(out_BPA,acc):
    return out_BPA*acc+0.5*(1-out_BPA)*(1-acc)

def trustyCalculate(trusty_yangben,out_predlabel,acc):
    #if out_predlabel == trusty_yangben[4] and out_predlabel == trusty_yangben[3] and out_predlabel == trusty_yangben[2] and out_predlabel == trusty_yangben[1]:
    #    acc = 0.95;
    #elif out_predlabel == trusty_yangben[4] or out_predlabel == trusty_yangben[3]:
        #acc = acc;
    #else:
    count=0;
    for i in range(len(trusty_yangben)):
        if out_predlabel == trusty_yangben[i]:
            count+=1;
    if count<=2:
        #acc = acc * (count/len(trusty_yangben));
        acc = 0.8*acc;
    elif count >=4:
        acc=0.95;
    else:
        acc = acc;
        #acc = acc;
    return acc
## parament 
num_classes = 3
frame_length = 1024;
inc = 1024;
path=r'/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/sequenceDataSet'
#semi_path=r'/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/test_semi'
savedir='../output/'

## read the semi model
model_semi=MediumScaleModel.SeismicNet(class_num=5)
model_semi.load_state_dict(torch.load('../output/MediumScaleModel-20200703_210118_Epoch093[81.626%].pth',map_location='cpu'))
model_semi.eval()

## read the aco model
model_aco=UrbanSoundModel.UrbanSound8KModel()
model_aco.load_state_dict(torch.load('../output/model_aco.pth',map_location='cpu'))
model_aco.eval()

files=ReadData.list_all_files(rootdir=path)

accs_aco=np.zeros(int(len(files)/2))
accs_semi=np.zeros(int(len(files)/2))
accs_fuse=np.zeros(int(len(files)/2))
accs_fuse_improved=np.zeros(int(len(files)/2))
k_compare_fig1=np.zeros(int(len(files)/2))
k_improved_compare_fig1=np.zeros(int(len(files)/2))
count_index_in_files=0
for item_file in files:
    
    if item_file.startswith('/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/sequenceDataSet/[A]'):
        if item_file.lower().find('largewheel')!=-1:
            true_label=0
        else:
            if item_file.lower().find('smallwheel')!=-1:
                true_label=1
            if item_file.lower().find('track')!=-1:
                true_label=2
        data=np.loadtxt(item_file)
        data=data[:,0]#信号多通道时取其中一个通道
        data=data[::8]
        datas_aco=data
        flag=0
        for item_sfile in files:
            if item_sfile.startswith('/media/seafood/3CE4B50EE4B4CC00/Database/Acoustic-and-seismic-synchronous-signal/20200805ASSSFromLYan/sequenceDataSet/[S]'):
                if item_sfile[-30:-1] == item_file[-30:-1]:
                    data=np.loadtxt(item_sfile)
                    datas_semi = data[::8]
                    flag=1
            else:
                flag=1
        if flag == 0:
            print('no match semi files with',item_file[-30:-1])
                    
        num_length = math.floor((len(datas_aco)-frame_length+inc)/inc)
## read the data
#true_label,datas_aco,datas_semi,num_length = ReadData.get_data_infile(path,frame_length,inc)

        count_wrong_aco_be4_fusion = 0;
        count_wrong_semi_be4_fusion = 0;
        count_wrong_fused_after_fusion = 0;
        count_wrong_fused_after_fusion_improved = 0;
        trusty_aco_yangben=[0,0,0,0,0];
        trusty_semi_yangben=[0,0,0,0,0];
        labels_pred = np.zeros((num_length,4))
        k_sum=0
        k_improved_sum=0
        #test
        for i in range(num_length):
            data_aco = datas_aco[i*inc:i*inc+frame_length]
            data_semi = datas_semi[i*inc:i*inc+frame_length]
    
            #yuchuli
            mfccs = UrbanSoundModel.yuchuli_aco(data_aco)
            features_semi = MediumScaleModel.yuchuli_semi(data_semi)
    
            #throw into the models
            out_aco_BPA = model_aco(mfccs).data.numpy()
            out_aco_BPA = out_aco_BPA/np.sum(out_aco_BPA)
            out_aco_predlabel = np.argmax(out_aco_BPA)
            labels_pred[i,0] = out_aco_predlabel;
            #tmp_
            out_semi_BPA = model_semi(features_semi).data.numpy()
            #out_semi_BPA = [out_semi_BPA_tmp[0],out_semi_BPA_tmp[2],out_semi_BPA_tmp[3]]
            out_semi_BPA = out_semi_BPA[:,[0,2,3]]
            out_semi_BPA = out_semi_BPA/np.sum(out_semi_BPA)
            out_semi_predlabel = np.argmax(out_semi_BPA)
                
            labels_pred[i,1] = out_semi_predlabel;
            #ans before Fusion
            if out_aco_predlabel!=true_label:
                count_wrong_aco_be4_fusion+=1
            if out_semi_predlabel!=true_label:
                count_wrong_semi_be4_fusion+=1
    
            #reset the BPA
            acc_aco = 0.67;
            acc_semi =0.77;
            #set the trusty
            if i < 5:
                trusty_aco_yangben[i]=out_aco_predlabel;
                trusty_semi_yangben[i]=out_semi_predlabel;
                
            else:
                acc_aco=trustyCalculate(trusty_aco_yangben,out_aco_predlabel,acc_aco)
                acc_semi=trustyCalculate(trusty_semi_yangben,out_semi_predlabel,acc_semi)
                #update yangben
                trusty_aco_yangben[0:3]=trusty_aco_yangben[1:4]
                trusty_aco_yangben[4]=out_aco_predlabel;
                trusty_semi_yangben[0:3]=trusty_semi_yangben[1:4]
                trusty_semi_yangben[4]=out_semi_predlabel;
        
            out_aco_BPA_improved = resetTheBpa(out_aco_BPA, acc=acc_aco) 
            out_semi_BPA_improved = resetTheBpa(out_semi_BPA, acc=acc_semi) 
    
            #Fusion
            con_matrix = np.zeros((3,3))+1-np.diag((1,1,1))
            k = np.dot(np.dot(out_aco_BPA, con_matrix),out_semi_BPA.T)
            k_improved = np.dot(np.dot(out_aco_BPA_improved, con_matrix),out_semi_BPA_improved.T)
                
            con_mass = 1/(1-k)
            con_mass_improved = 1/(1-k_improved)
            #if k < 0.6:
            mass = con_mass * np.multiply(out_aco_BPA, out_semi_BPA)
            mass_improved = con_mass_improved * np.multiply(out_aco_BPA_improved, out_semi_BPA_improved)
            #else:
                #mass = out_semi_BPA
            fuse_ans = mass.argmax()
            fuse_ans_improved = mass_improved.argmax()
            labels_pred[i,2] = fuse_ans;
            labels_pred[i,3] = fuse_ans_improved;
            #ans after Fusion
            if fuse_ans != true_label:
                count_wrong_fused_after_fusion+=1;
            if fuse_ans_improved != true_label:
                count_wrong_fused_after_fusion_improved+=1;
            k_sum=k_sum+k
            k_improved_sum=k_improved_sum=k_improved
            #summer and printf
        k_compare_fig1[count_index_in_files]=k_sum/num_length
        k_improved_compare_fig1[count_index_in_files]=k_improved_sum/num_length
        accs_aco[count_index_in_files]=(1-count_wrong_aco_be4_fusion/num_length)
        accs_semi[count_index_in_files]=(1-count_wrong_semi_be4_fusion/num_length)
        accs_fuse[count_index_in_files]=(1-count_wrong_fused_after_fusion/num_length)
        accs_fuse_improved[count_index_in_files]=(1-count_wrong_fused_after_fusion_improved/num_length)
        print('[',count_index_in_files,'] ',item_file[-30:-1],' aco  accurate before fusion is ',accs_aco[count_index_in_files])
        print('[',count_index_in_files,'] ',item_file[-30:-1],' semi accurate before fusion is ',accs_semi[count_index_in_files])
        print('[',count_index_in_files,'] ',item_file[-30:-1],' fuse accurate after  origin   fusion is ',accs_fuse[count_index_in_files])
        print('[',count_index_in_files,'] ',item_file[-30:-1],' fuse accurate after  improved fusion is ',accs_fuse_improved[count_index_in_files])
        print('-------------------------------------------------------------------------------------')
        count_index_in_files+=1
        
print ('k_origin :',np.mean(k_compare_fig1),'k_improved :',np.mean(k_improved_compare_fig1))
print('acc_aco is ',np.mean(accs_aco),'and acc_semi is ',np.mean(accs_semi),'while acc_origin_fuse is ',np.mean(accs_fuse),'while acc_improved_fuse is ',np.mean(accs_fuse_improved))
#
compare_ans=plt.figure(num=0)
plt.plot(range(count_index_in_files),k_compare_fig1,color='k',linestyle='--')
plt.plot(range(count_index_in_files),k_improved_compare_fig1,color='k',linestyle='-')

#plt.title('The conflict coefficient k is reduced')
plt.title(u'原始融合规则和改进融合规则证据冲突系数',fontproperties=zhfont)
plt.xlabel(u'数据文件',fontproperties=zhfont)
plt.ylabel(u'冲突度',fontproperties=zhfont)
#plt.title('Evidence Conflict Factor for Original and Improved Fusion Rules')
#plt.xlabel('DataFiles')
#plt.ylabel('Conflict Factor')
#plt.legend(('aco','semi','improved_fuse'))
#plt.legend(('Origin D-S fuse Conflict','Improved D-S fuse Conflict'))
plt.legend((u'原始D-S证据理论冲突系数',u'改进D-S证据理论融合冲突系数'),prop=zhfont)
plt.grid()
plt.savefig('../output/grayfig_conflict.png',dpi=720)
plt.show()

#
origin_ans=plt.figure(num=1)
plt.plot(range(count_index_in_files),accs_aco,color='k',linestyle='-.')
plt.plot(range(count_index_in_files),accs_semi,color='k',linestyle=':')
plt.plot(range(count_index_in_files),accs_fuse,color='k',linestyle='-')
#plt.title('Promotion effect of original D-S evidence theory')
#plt.legend(('aco accuary','semi accuary','origin_fuse accuary'))
plt.title(u'原始D-S证据理论融合识别正确率结果',fontproperties=zhfont)
plt.legend((u'声音',u'震动',u'原始D-S证据理论融合'),prop=zhfont)
plt.xlabel(u'数据文件',fontproperties=zhfont)
plt.ylabel(u'准确率',fontproperties=zhfont)
plt.grid()
plt.savefig('../output/grayfig_origin.png',dpi=720)
plt.show()
#
improved_ans=plt.figure(num=2)
plt.plot(range(count_index_in_files),accs_aco,color='k',linestyle='-.')
plt.plot(range(count_index_in_files),accs_semi,color='k',linestyle=':')
plt.plot(range(count_index_in_files),accs_fuse_improved,color='k',linestyle='-')
#plt.title('Promotion effect of improved D-S evidence theory')
#plt.legend(('aco accuary','semi accuary','improved_fuse accuary'))
plt.title(u'改进D-S证据理论融合正确率结果',fontproperties=zhfont)
plt.legend((u'声音',u'震动',u'改进D-S证据理论融合'),prop=zhfont)
plt.xlabel(u'数据文件',fontproperties=zhfont)
plt.ylabel(u'准确率',fontproperties=zhfont)
plt.grid()
plt.savefig('../output/grayfig_improved.png',dpi=720)
plt.show()
#
compare_ans=plt.figure(num=3)
plt.plot(range(count_index_in_files),accs_fuse,color='k',linestyle='--')
plt.plot(range(count_index_in_files),accs_fuse_improved,color='k',linestyle='-')
#plt.title('Comparison between D-S evidence theory and improved D-S evidence theory')
#plt.legend(('improved_fuse accuary','origin_fuse accuary'))
plt.title(u'改进D-S证据理论与原始D-S证据理论融合正确率比较',fontproperties=zhfont)
plt.legend((u'原始D-S证据理论',u'改进D-S证据理论融合'),prop=zhfont)
plt.xlabel(u'数据文件',fontproperties=zhfont)
plt.ylabel(u'准确率',fontproperties=zhfont)
plt.grid()
plt.savefig('../output/grayfig_compare.png',dpi=720)
plt.imshow()
