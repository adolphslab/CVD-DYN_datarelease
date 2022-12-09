# -*- coding: utf-8 -*-
#preprocessing functions for preprocessing script

#import codecs
import glob
#import os
#import re
#import time
#import random
#try:
#    import cPickle as pickle
#except:
#    import pickle

import pandas as pd
import numpy as np
import os
#from scipy import optimize
#import scipy as sp
from scipy import stats as stats
#import matplotlib.pyplot as plt
#import statsmodels.api as sm
#import altair as alt#

# Functions used later.

def longString(ser):
    strs=[]
    strCtr=1
    lval=ser.iloc[0]
    for ind in range(1,ser.shape[0]):
        if ser.iloc[ind]==lval:
            strCtr+=1
        else:
            strs.append(strCtr)
            strCtr=1
            lval=ser.iloc[ind]

    if len(strs)==0:
        strs.append(strCtr)
        
    maxLongStr=np.max(strs)
    meanStrLength=np.mean(strs)
    countLongStr=np.sum(np.array(strs)>5)
    
    out={'maxLongStr':maxLongStr,
           'meanLongString':np.round(meanStrLength,2),
           'countLongStr':countLongStr,
#            'strs':strs,
        }
           
    
    return out 


def process_tr_1s(fdir):
    # Load in and preprocess trust rating data
    #     fname=('/Users/dstanley/week1_tasks/tr_1s/tr_1s_54924b8efdf99b77ccedc1d5_SESSION_2020-04-05_11h12.53.432.csv') # for testing
    tr_1s_df=pd.DataFrame()
    for fname in glob.iglob(fdir+'/tr_1s*.csv'):
        file_loaded = False
        load_idx = 0
        while file_loaded == False:
            try:
                dat=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
                file_loaded = True
            except:
                load_idx += 1
                print(load_idx)
        
        sid=dat.subject_id.iloc[0].lstrip(' -').rstrip(' -')
        dat.dropna(subset=['trialrace'], inplace=True)
        dat['normedkey']=stats.zscore(dat.key_press)
        tmpMeanRat=dat.groupby(['trialrace']).mean().key_press
        tmpMeanRat.rename(sid, inplace=True)    
        tmpNormMeanRat=dat.groupby(['trialrace']).mean().normedkey
        tmpNormMeanRat.rename(sid, inplace=True)
        tmpNormMeanRat=tmpNormMeanRat.to_frame().transpose()
        tmpNormMeanRat.columns=['nm_'+col for col in tmpNormMeanRat.columns]
        tmp_tr_1s=tmpMeanRat.to_frame().transpose().join(tmpNormMeanRat)
        strQuant=pd.Series(longString(dat['key_press']),name=sid)
        tmp_tr_1s=tmp_tr_1s.join(strQuant.to_frame().transpose())
        tmp_tr_1s.loc[sid,'rt_pctlt_300']=np.sum(dat.rt<300)/dat.shape[0]
        tmp_tr_1s.loc[sid,'medianRT'] = dat.rt.median()
        if np.isnan(dat['normedkey'].iloc[0]):
            tmp_tr_1s.loc[sid,'noVar']=1
        else:
            tmp_tr_1s.loc[sid,'noVar']=0
        tmp_tr_1s.loc[sid,'totalTime']=dat.totalTime.iloc[0]/1000/60
        fname_part = fname.partition('/tr_1s/')
        fname_clean = fname_part[2]
        tmp_tr_1s.loc[sid,'fname']=fname_clean
        tr_1s_df=tr_1s_df.append(tmp_tr_1s)
    
    # Eliminate later runs of duplicates
    tr_1s_df=tr_1s_df[~tr_1s_df['fname'].isin([i for i in tr_1s_df[tr_1s_df.index.duplicated()].fname])]
    tr_1s_df.loc[:,'administered'] = True
    # rename columns
    tr_1s_df.columns=['tr_1s_'+col for col in tr_1s_df.columns]

    return tr_1s_df


def process_pro_iat(fdir):
    # Load in and preprocess iat data
    iat=pd.DataFrame()
#     fname='../week1_tasks/pro_iat/pro_iat_562c6951733ea000111631be_SESSION_2020-04-05_11h11.34.549.csv' # for testing
    for fname in glob.iglob(fdir+'/pro_iat*.csv'):
        file_loaded = False
        load_idx = 0
        while file_loaded == False:
            try:
                dat_iat=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
                file_loaded = True
            except:
                load_idx += 1
                print(load_idx)
        
        sid=dat_iat.subject_id[0].lstrip(' -').rstrip(' -')
        score=pd.Series(dat_iat.iatd[0], name=sid, index=['IATD'])
        score=score.to_frame().transpose()
        score['include']=score['IATD'].notna().astype(int)
        score.loc[sid,'totalTime']=dat_iat.totalTime[0]/1000/60
        score.loc[sid,'fname']=fname
        iat=iat.append(score)

    # Eliminate later runs of duplicates
    iat=iat[~iat['fname'].isin([i for i in iat[iat.index.duplicated()].fname])]
    iat.loc[:,'administered'] = True
    # rename columns
    iat.columns=['iat_'+col for col in iat.columns]

    return iat

def process_biat(fdir):
#     # Load in and preprocess iat data
    biat=pd.DataFrame()
    for fname in glob.iglob(fdir+'/biat*.csv'):
        file_loaded = False
        load_idx = 0
        while file_loaded == False:
            try:
                dat_iat=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
                file_loaded = True
            except:
                load_idx += 1
                print(load_idx)        

        dat_iat=dat_iat[dat_iat.trial_tag=='LIVE'].reset_index(drop=True)
        dat_iat['subject_id'] = dat_iat.subject_id[0].lstrip(' -').rstrip(' -')
        iatCols=np.sort(dat_iat.exp_type.unique())
        iatCols=np.append(np.append(np.append(np.append(iatCols, iatCols+'_pctRtGt10k'), iatCols+'_pctRtGt2k'), iatCols+'_pctRtLt400'), iatCols+'_pctRtLt300')
        iat=pd.DataFrame(columns=iatCols)
        iat.loc[dat_iat['subject_id'][0],'include']=0
        iat.loc[dat_iat['subject_id'][0],'fname']=fname
        iat.loc[dat_iat['subject_id'][0],'totalTime']=dat_iat.totalTime[0]/1000/60

        for i in np.sort(dat_iat.exp_type.unique()):
            tmp=dat_iat[dat_iat['exp_type']==i].reset_index(drop=True)
            pctRtOver10000=np.sum(tmp.frt>=10000)/tmp.shape[0]
            tmp=tmp[tmp.frt<10000]
            pctRtOver2000=np.sum(tmp.frt>=2000)/tmp.shape[0]
            tmp.loc[tmp.frt>2000,'frt']=2000
            pctRtUnder400=np.sum(tmp.frt<400)/tmp.shape[0]
            pctRtUnder300=np.sum(tmp.frt<300)/tmp.shape[0]
            iat.loc[dat_iat['subject_id'][0],i+'_pctRtGt10k']=pctRtOver10000
            iat.loc[dat_iat['subject_id'][0],i+'_pctRtGt2k']=pctRtOver2000
            iat.loc[dat_iat['subject_id'][0],i+'_pctRtLt400']=pctRtUnder400
            iat.loc[dat_iat['subject_id'][0],i+'_pctRtLt300']=pctRtUnder300

            if (np.sum(tmp['frt']<300)>tmp.shape[0]*.1) or (pctRtOver2000>.5):
                iat.loc[dat_iat['subject_id'][0],i]=float('NaN')
            else:
                tmp.loc[tmp.frt<400,'frt']=400
                D1=(tmp.loc[16:31, 'frt'].mean()-tmp.loc[0:15, 'frt'].mean())/tmp.loc[0:31,'frt'].std()
                D2=(tmp.loc[48:63, 'frt'].mean()-tmp.loc[32:47, 'frt'].mean())/tmp.loc[32:63,'frt'].std()
                iat.loc[dat_iat['subject_id'][0],i]=np.mean([D1,D2])

                
                
        biat=biat.append(iat)

    # Eliminate later runs of duplicates
    biat=biat[~biat['fname'].isin([i for i in biat[biat.index.duplicated()].fname])]
    biat['include']=biat.iloc[:,:3].notna().transpose().sum()
    biat.loc[:,'administered'] = True
    
    # rename columns
    biat.columns=['biat_'+col for col in biat.columns]

    return biat


def process_cvd_amp(fdir):
# Load in and preprocess amp data
    amp=pd.DataFrame()
    # fname='../week1_tasks/cvd_amp/cvd_amp_542473a4fdf99b691fb38455_SESSION_2020-04-05_22h12.06.268.csv' # for testing
    for fname in glob.iglob(fdir+'/cvd_amp*.csv'):
        file_loaded = False
        load_idx = 0
        while file_loaded == False:
            try:
                dat_amp=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
                file_loaded = True
            except:
                load_idx += 1
                print(load_idx) 
        
        sid=dat_amp.subject_id[0].lstrip(' -').rstrip(' -')
        tmpMResp=dat_amp.loc[0,'mean_black':'mean_grey'].to_frame().transpose().rename(index={0 : sid})
        tmpMResp.loc[sid,'pct_good_rts']=dat_amp.prct_usable_trials[0]
        strQuant=pd.Series(longString(dat_amp.loc[dat_amp['phase']=='datacol'].resp_val), name=sid)
        tmpMResp=tmpMResp.join(strQuant.to_frame().transpose())
        tmpMResp.loc[sid,'totalTime']=dat_amp.loc[0,'totalTime']/1000/60
        tmpMResp.loc[sid,'medianRT'] = dat_amp.rt.median()
        tmpMResp.loc[sid,'fname']=fname
        amp=amp.append(tmpMResp)

    amp['pct_bad_rts']= 1- amp['pct_good_rts']
    # Eliminate later runs of duplicates
    amp=amp[~amp['fname'].isin([i for i in amp[amp.index.duplicated()].fname])]
    amp.loc[:,'administered'] = True

    # rename columns
    amp.columns=['amp_'+col for col in amp.columns]
    
    return amp

def process_cvd_pgg(fdir, w1=False):
    # Load in and preprocess pgg data
    fdir = os.path.expanduser(fdir)
    pgg=pd.DataFrame()
    # fname='../week1_tasks/cvd_pgg/cvd_pgg_542473a4fdf99b691fb38455_SESSION_2020-04-05_22h01.45.624.csv' # for testing
    for fname in glob.iglob(fdir+'/cvd_pgg*.csv'):
        file_loaded = False
        load_idx = 0
        while file_loaded == False:
            try:
                dat_pgg=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
                file_loaded = True
            except:
                load_idx += 1
                print(load_idx)
        
        
        tmp_pgg=pd.DataFrame()
        sid=dat_pgg.subject_id[0].lstrip(' -').rstrip(' -')
        

        if w1:
            tmp_pgg.loc[sid,'invest']=dat_pgg[dat_pgg['trial_tag']=='investFB'].response_feedback.iloc[0]
            tmp_pgg.loc[sid,'rt']=dat_pgg[dat_pgg['trial_type']=='survey-text'].rt.iloc[0]
        else:
            tmp_pgg.loc[sid,'invest']=dat_pgg[dat_pgg['trial_tag']=='choice']['button_pressed'].iloc[0]
            tmp_pgg.loc[sid,'rt']=dat_pgg[dat_pgg['trial_tag']=='choice']['rt'].iloc[0]

        tmp_pgg.loc[sid,'totalTime']=dat_pgg.totalTime[0]/1000/60
        fname_part = fname.partition('/cvd_pgg/')
        fname_clean = fname_part[2]
        tmp_pgg.loc[sid,'fname']=fname_clean
        pgg=pgg.append(tmp_pgg)
        
    # Eliminate later runs of duplicates
    pgg=pgg[~pgg['fname'].isin([i for i in pgg[pgg.index.duplicated()].fname])]
    pgg.loc[:,'administered'] = True

    # rename columns
    pgg.columns=['pgg_'+col for col in pgg.columns]
    
    return pgg

def process_cvd_altt(fdir):
    # Load in and preprocess amp data
    fdir = os.path.expanduser(fdir)
    altt=pd.DataFrame()
    rand_trial=pd.DataFrame()
    altt_summary=pd.DataFrame()
    for fname in glob.iglob(fdir+'/cvd_altt*.csv'):
        file_loaded = False
        load_idx = 0
        while file_loaded == False:
            try:
                dat_altt=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
                file_loaded = True
            except:
                load_idx += 1
        
        sid=dat_altt.subject_id[0].lstrip(' -').rstrip(' -')
        dat_altt_tmp = dat_altt[dat_altt['trial_tag']=='response'].reset_index()
        dat_altt_tmp = dat_altt_tmp.drop(['index'],axis = 1) 
        dat_altt_tmp['response'] = float('nan')
      
        # define responses    
        equ_resp = dat_altt_tmp['equal_resp'].str[0]
        unequ_resp = dat_altt_tmp['unequal_resp'].str[0]
        dat_altt_tmp.loc[(dat_altt_tmp['choice_type']=='selfish') & (dat_altt_tmp['key_press']==unequ_resp),['response']]= -1        
        dat_altt_tmp.loc[(dat_altt_tmp['choice_type']=='generous') & (dat_altt_tmp['key_press']==unequ_resp),['response']]= 1       
        dat_altt_tmp.loc[(dat_altt_tmp['key_press']==equ_resp),['response']]= 0      

        
        if 'respRand' in dat_altt_tmp.columns: # randomized response options
            dat_altt_tmp = dat_altt_tmp.drop(['unequal_resp','equal_resp','stimulus','trial_tag','trial_type','trial_index','internal_node_id','startTime','view_history'], axis = 1)
            dat_altt_tmp['respRand'] = dat_altt_tmp['respRand'].mean()
            outString = longString(dat_altt_tmp.key_press)
      
        
        else: # non randomized response options
            dat_altt_tmp = dat_altt_tmp.drop(['unequal_resp','equal_resp','stimulus','trial_tag','trial_type','trial_index','internal_node_id','startTime','view_history'], axis = 1)
            dat_altt_tmp['respRand'] = 0
         
            
        if 'event' in  dat_altt_tmp.columns:
            dat_altt_tmp = dat_altt_tmp.drop(['event','trial','time'], axis = 1)

        # mean RT and mean reponses
        tmpDF=dat_altt.loc[0,['totalTime']].to_frame().transpose().rename(index={0 : sid})
        tmpDF.loc[sid,'meanResp']=np.mean(dat_altt_tmp['response'])
        tmpDF.loc[sid,'medianRT']=np.median(dat_altt_tmp['rt'])
        tmpDF.loc[sid,'meanRT']=np.mean(dat_altt_tmp['rt'])
        tmpDF.loc[sid,'rt_pctlt_300']= np.sum(dat_altt_tmp['rt']<300)/len(dat_altt_tmp)
        tmpDF.loc[sid,'totalTime']=tmpDF.loc[sid,'totalTime']/1000/60
        
        # mean age response
        tmpDF.loc[sid,'age_Young'] = dat_altt_tmp.loc[dat_altt_tmp['ageIdent']=='young','response'].mean()
        tmpDF.loc[sid,'age_Old'] =   dat_altt_tmp.loc[dat_altt_tmp['ageIdent']=='old','response'].mean()
        # mean age differences 
        tmpDF.loc[sid,'age_OldDifYoung'] = tmpDF.loc[sid,'age_Old'] - tmpDF.loc[sid,'age_Young']
        
        # mean race/ethnicity response
        tmpDF.loc[sid,'raceEth_Asian'] =  dat_altt_tmp.loc[dat_altt_tmp['race']=='Asian','response'].mean()
        tmpDF.loc[sid,'raceEth_Black'] =  dat_altt_tmp.loc[dat_altt_tmp['race']=='Black','response'].mean()
        tmpDF.loc[sid,'raceEth_White'] =  dat_altt_tmp.loc[dat_altt_tmp['race']=='White','response'].mean()
        tmpDF.loc[sid,'raceEth_Hisp'] =   dat_altt_tmp.loc[dat_altt_tmp['race']=='Hispanic','response'].mean()
        # mean reace/ethnicity differences 
        tmpDF.loc[sid,'raceEth_WhiteDifAsian'] = tmpDF.loc[sid,'raceEth_White'] - tmpDF.loc[sid,'raceEth_Asian']
        tmpDF.loc[sid,'raceEth_WhiteDifBlack'] = tmpDF.loc[sid,'raceEth_White'] - tmpDF.loc[sid,'raceEth_Black']
        tmpDF.loc[sid,'raceEth_WhiteDifHisp'] =  tmpDF.loc[sid,'raceEth_White'] - tmpDF.loc[sid,'raceEth_Hisp']
 
         # mean polit ident response
        tmpDF.loc[sid,'polit_Indep'] = dat_altt_tmp.loc[dat_altt_tmp['polit_ident']=='Independent','response'].mean()
        tmpDF.loc[sid,'polit_Rep'] =   dat_altt_tmp.loc[dat_altt_tmp['polit_ident']=='Republican','response'].mean()
        tmpDF.loc[sid,'polit_Dem'] =   dat_altt_tmp.loc[dat_altt_tmp['polit_ident']=='Democrat','response'].mean()
        # mean polit. ident. differences 
        tmpDF.loc[sid,'polit_DemDifRep'] = tmpDF.loc[sid,'polit_Dem'] - tmpDF.loc[sid,'polit_Rep']
        
        if (dat_altt_tmp['respRand'] != 0).all(axis=0): 
            # response string length
            tmpDF.loc[sid,'maxLongString'] = outString['maxLongStr']
            tmpDF.loc[sid,'meanLongString'] = outString['meanLongString']
            tmpDF.loc[sid,'countLongStr'] = outString['countLongStr']
        elif (dat_altt_tmp['respRand'] == 0).all(axis=0): # non randomized response options
            tmpDF.loc[sid,'maxLongString'] = np.nan
            tmpDF.loc[sid,'meanLongString'] = np.nan
            tmpDF.loc[sid,'countLongStr'] = np.nan

        # summary data 
        tmpDF.loc[sid,'fname']=fname
        fname_part = fname.partition('/cvd_altt/')
        fname_clean = fname_part[2]
        tmpDF.loc[sid,'fname']=fname_clean
        altt_summary=altt_summary.append(tmpDF)
        
        # all sub full data 
        altt = altt.append(dat_altt_tmp)
        # randomly sample one trial for lottery
        rand_trial = rand_trial.append(dat_altt_tmp.sample())

    # Eliminate later runs of duplicates
    altt_summary=altt_summary[~altt_summary['fname'].isin([i for i in altt_summary[altt_summary.index.duplicated()].fname])]
    altt_summary.loc[:,'administered'] = True
    
    # rename columns
    altt.columns=['altt_'+col for col in altt.columns]
    altt_summary.columns=['altt_'+col for col in altt_summary.columns]

    
    return altt,altt_summary,rand_trial



def process_cvd_consp(fdir):
     # Load in and combine trust rating data
    cvd_consp_df=pd.DataFrame()
    for fname in glob.iglob(fdir+'/cvd_consp*.csv'):
        file_loaded = False
        load_idx = 0

        while file_loaded == False:
            try:
                dat=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
                file_loaded = True
                sid=dat.subject_id.iloc[0].lstrip(' -').rstrip(' -')
            except:
                load_idx += 1
                print(load_idx)

        sid=dat.subject_id.iloc[0].lstrip(' -').rstrip(' -')
        dat['task'] = 'cvd_consp'
        dat = dat.dropna(subset = ['stim_type'])
        dat = dat.loc[(dat.practice == False) | (dat.practice == "false"), :]
        dat.reset_index(drop = True, inplace = True)
        
        # save reponse string length 
        tmpDF=dat.loc[0,['totalTime']].to_frame().transpose().rename(index={0 : sid})

        outString = longString(dat.key_press)        
        tmpDF.loc[sid,'maxLongString'] = outString['maxLongStr']
        tmpDF.loc[sid,'meanLongString'] = outString['meanLongString']
        tmpDF.loc[sid,'countLongStr'] = outString['countLongStr']
        tmpDF.loc[sid,'countLongStr'] = outString['countLongStr']
        tmpDF.loc[sid,'rt_pctlt_300']= np.sum(dat['rt']<300)/len(dat)
        tmpDF.loc[sid,'administered'] = True

        
        # Eliminate later runs of duplicates
        if sum(cvd_consp_df.index == sid) == 0:
            cvd_consp_df=cvd_consp_df.append(tmpDF)
    
    #cvd_consp_df.index.names = ['PROLIFIC_PID']
    # rename columns
    cvd_consp_df.columns=['cvd_consp_'+col for col in cvd_consp_df.columns]

    return cvd_consp_df






def cvdpenGen(inccsv, lucsv, ofname, exccsvs=[]):
    lut=pd.read_csv(lucsv)
    lut.set_index('PID', inplace=True)
    lut.index=[i.lstrip('- ').rstrip('- ') for i in lut.index]
    
    if len(inccsv)>0:
        inc=pd.read_csv(inccsv, header=None).rename(columns={0:'PID'})
        inc.set_index('PID', inplace=True)
    else:
        inc=lut

    # trim cvdpen list to only use included participants
    lut=lut.loc[list(inc.index)]

    # Create Strings from lists for QUALTRICS PEN/CVD code generation
    sidstr=''
    cvdstr=''
    penstr=''
    for sid,cvd,pen in zip(lut.index, lut['CVD'], lut['PEN']):
        sidstr += "'" + sid + "', "
        cvdstr += "'" + np.str(cvd) + "', "
        penstr += "'" + np.str(pen) + "', "

    f=open(ofname,'w')
    print('var ID_block_table = {\n' +
        '\tID: ['+sidstr[:-2]+'],\n' +
        '\tCVD: ['+cvdstr[:-2]+'],\n' +
        '\tPEN: ['+penstr[:-2]+'],\n' +
          '\t};',
         file=f)   
    f.close()
    
    print('Output code exported to '+ofname)
    
    

    
    
def cvdpenGen_CONTE(lucsv, ofname, exccsvs=[]):
    lut=pd.read_csv(lucsv)
    lut.set_index('CCID', inplace=True)
    lut.index=[i.lstrip('- ').rstrip('- ') for i in lut.index]
    
    # Create Strings from lists for QUALTRICS PEN/CVD code generation
    sidstr=''
    cvdstr=''
    penstr=''
    for sid,cvd,pen in zip(lut.index, lut['CVD'], lut['PEN']):
        sidstr += "'" + sid + "', "
        cvdstr += "'" + np.str(cvd) + "', "
        penstr += "'" + np.str(pen) + "', "

    f=open(ofname,'w')
    print('var ID_block_table = {\n' +
        '\tID: ['+sidstr[:-2]+'],\n' +
        '\tCVD: ['+cvdstr[:-2]+'],\n' +
        '\tPEN: ['+penstr[:-2]+'],\n' +
          '\t};',
         file=f)   
    f.close()
    
    print('Output code exported to '+ofname)
    
    
    
    
def cvdpenembodGen(inccsv, lucsv, embodcsv, ofname, exccsvs=[]):
    lut=pd.read_csv(lucsv)
    lut.set_index('PID', inplace=True)
    lut.index=[i.lstrip('- ').rstrip('- ') for i in lut.index]
    
    embod=pd.read_csv(embodcsv)
    embod.set_index('PID', inplace=True)
    embod.index=[i.lstrip('- ').rstrip('- ') for i in embod.index]
    
    if len(inccsv)>0:
        inc=pd.read_csv(inccsv, header=None).rename(columns={0:'PID'})
        inc.set_index('PID', inplace=True)
    else:
        inc=lut

    # trim cvdpen list to only use included participants
    lut=lut.loc[list(inc.index)]

    lut=lut.join(embod)
    
    # Create Strings from lists for QUALTRICS PEN/CVD code generation
    sidstr=''
    cvdstr=''
    penstr=''
    embstr=''
    for sid,cvd,pen,emb in zip(lut.index, lut['CVD'], lut['PEN'], lut['Emb_im']):
        sidstr += "'" + sid + "', "
        cvdstr += "'" + np.str(cvd) + "', "
        penstr += "'" + np.str(pen) + "', "
        embstr += "'" + np.str(emb) + "', "

    f=open(ofname,'w')
    print('var ID_block_table = {\n' +
        '\tID: ['+sidstr[:-2]+'],\n' +
        '\tCVD: ['+cvdstr[:-2]+'],\n' +
        '\tPEN: ['+penstr[:-2]+'],\n' +
        '\tEmb_im: ['+embstr[:-2]+'],\n' +
          '\t};',
         file=f)   
    f.close()
    
    print('Output code exported to '+ofname)

    
    
def cvdpenembodGen_CONTE(lucsv, embodcsv, ofname, exccsvs=[]):
    lut=pd.read_csv(lucsv)
    lut.set_index('CCID', inplace=True)
    lut.index=[i.lstrip('- ').rstrip('- ') for i in lut.index]

    embod=pd.read_csv(embodcsv)
    embod.set_index('CCID', inplace=True)
    embod.index=[i.lstrip('- ').rstrip('- ') for i in embod.index]
    
    lut=lut.join(embod)

    # Create Strings from lists for QUALTRICS PEN/CVD code generation
    sidstr=''
    cvdstr=''
    penstr=''
    embstr=''
    for sid,cvd,pen,emb in zip(lut.index, lut['CVD'], lut['PEN'], lut['Emb_im']):
        sidstr += "'" + sid + "', "
        cvdstr += "'" + np.str(cvd) + "', "
        penstr += "'" + np.str(pen) + "', "
        embstr += "'" + np.str(emb) + "', "

    f=open(ofname,'w')
    print('var ID_block_table = {\n' +
        '\tID: ['+sidstr[:-2]+'],\n' +
        '\tCVD: ['+cvdstr[:-2]+'],\n' +
        '\tPEN: ['+penstr[:-2]+'],\n' +
        '\tEmb_im: ['+embstr[:-2]+'],\n' +
          '\t};',
         file=f)   
    f.close()
    
    print('Output code exported to '+ofname)
