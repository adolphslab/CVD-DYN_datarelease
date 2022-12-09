import glob
import pandas as pd
import numpy as np
import os


def process_tr_1s(fdir, wave):
    # Load in and combine trust rating data
    tr_1s_df=pd.DataFrame()
    tr_1s_df["subject_id"] = ""
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
        dat = dat.loc[dat['trialtype']=='TR',:]
        dat['wave']= wave
        dat['trial_nr']=range(1,len(dat)+1)
        dat['task'] = 'tr_1s'
                
        dat = dat[['task','subject_id','wave', 'trial_nr','trial_type','trial_index',
                   'time_elapsed','startTime','totalTime',
                   'rt','stimulus', 'key_press', 'trialrace']]
        
        # Eliminate later runs of duplicates
        if sum(tr_1s_df.subject_id == sid) == 0:
            tr_1s_df= pd.concat([tr_1s_df,dat])
   
    # rename columns
    tr_1s_df.columns=['tr_1s_'+col for col in tr_1s_df.columns]
    tr_1s_df.reset_index(drop=True, inplace=True)

    return tr_1s_df


def process_pro_iat(fdir, wave):
    # Load in and preprocess iat data
    iat_df=pd.DataFrame()
    iat_df["subject_id"] = ""
    for fname in glob.iglob(fdir+'/pro_iat*.csv'):
        dat=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
        sid=dat.subject_id[0].lstrip(' -').rstrip(' -')
        dat = dat.dropna(subset=['condition'])
        dat['wave']= wave
        dat['trial_nr']=range(1,len(dat)+1)
        dat['task'] = 'pro_iat'
        
        dat = dat[['task','subject_id','wave', 'trial_nr','trial_type','trial_index',
                   'time_elapsed','iatcond','iathand','startTime','totalTime','iatd',
                   'rt','stimulus', 'key_press', 'correct','frt', 'condition',
                   'hand']]
         
        # Eliminate later runs of duplicates
        if sum(iat_df.subject_id == sid) == 0:
            iat_df= pd.concat([iat_df,dat])
            
    # rename columns
    iat_df.columns=['iat_'+col for col in iat_df.columns]
    iat_df.reset_index(drop=True, inplace=True)

    return iat_df


def process_biat(fdir, wave):
    biat_df_raw = pd.DataFrame()
    biat_df_summary = pd.DataFrame()
    biat_df_raw["subject_id"] = ""

    if os.path.exists(fdir):
        if len(os.listdir(fdir))>0:
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
                sid = dat_iat.loc[0,'subject_id']
                
                # BIAT RAW
                dat_raw = dat_iat.copy()
                dat_raw['wave']= wave
                dat_raw['trial_nr']=range(1,len(dat_raw)+1)
                dat_raw['task'] = 'biat'
                dat_raw = dat_raw[['task','subject_id','wave', 'trial_nr','trial_type','trial_index',
                           'time_elapsed','iatcond','startTime','totalTime',
                           'rt','stimulus', 'key_press', 'correct','frt', 'condition','hand',
                           'exp_type','stim_assoc','trial_tag']]
                
                # ignore later runs of duplicates
                if sum(biat_df_raw.subject_id == sid) == 0:
                        biat_df_raw= pd.concat([biat_df_raw,dat_raw])
               
                # BIAT SUBJECT SUMMARY 
                iatCols=np.sort(dat_iat.exp_type.unique())
                iatCols=np.append(np.append(np.append(np.append(iatCols, iatCols+'_pctRtGt10k'), iatCols+'_pctRtGt2k'), iatCols+'_pctRtLt400'), iatCols+'_pctRtLt300')
                #iatCols = pd.concat([iatCols, iatCols+'_pctRtGt10k', iatCols+'_pctRtGt2k', iatCols+'_pctRtLt400', iatCols+'_pctRtLt300'])
                iat=pd.DataFrame(columns=iatCols)
                iat.loc[dat_iat['subject_id'][0],'subject_id']=dat_iat.subject_id[0]
                iat.loc[dat_iat['subject_id'][0],'wave']= wave
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
                biat_df_summary = pd.concat([biat_df_summary, iat])

                # Eliminate later runs of duplicates
                biat_df_summary=biat_df_summary[~biat_df_summary['fname'].isin([i for i in biat_df_summary[biat_df_summary.index.duplicated()].fname])]
                biat_df_summary['include']=biat_df_summary.iloc[:,:3].notna().transpose().sum()
                biat_df_summary.loc[:,'administered'] = True

    # rename columns biat raw
    biat_df_raw.columns=['biat_'+col for col in biat_df_raw.columns]
    biat_df_raw.reset_index(drop=True, inplace=True)
    
    # rename columns biat summary 
    biat_df_summary.columns=['biat_'+col for col in biat_df_summary.columns]
    biat_df_summary.reset_index(drop=True, inplace=True)

    return biat_df_raw, biat_df_summary


def process_cvd_amp(fdir,wave):
# Load in and preprocess amp data
    amp_df=pd.DataFrame()
    amp_df["subject_id"] = ""
    for fname in glob.iglob(fdir+'/cvd_amp*.csv'):
        file_loaded = False
        load_idx = 0
        while file_loaded == False:
            try:
                dat=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
                file_loaded = True
            except:
                load_idx += 1
                print(load_idx)
        sid=dat.subject_id[0].lstrip(' -').rstrip(' -')
        dat = dat.dropna(subset=['phase'])
        dat['wave']= wave
        dat['task'] = 'amp'
        dat['trial_nr']=range(1,len(dat)+1)
        dat = dat[['task','subject_id','wave', 'trial_nr','trial_type','trial_index',
                   'time_elapsed','hand','startTime','totalTime','mean_black',
                   'mean_white', 'mean_asian', 'mean_grey','prct_usable_trials',
                   'rt','stimulus','key_press','test_part', 'resp_val', 'phase', 'cue']]
        
        # Eliminate later runs of duplicates
        if sum(amp_df.subject_id == sid) == 0:
            amp_df=pd.concat([amp_df,dat])

    # rename columns
    amp_df.columns=['amp_'+col for col in amp_df.columns]
    amp_df.reset_index(drop=True, inplace=True)
    return amp_df
    
def process_cvd_pgg(fdir, wave):
    # Load in and preprocess pgg data
    fdir = os.path.expanduser(fdir)
    pgg=pd.DataFrame()
    for fname in glob.iglob(fdir+'/cvd_pgg*.csv'):
        dat_pgg=pd.read_csv(fname, encoding = "ISO-8859-1", engine='python')
        tmp_pgg=pd.DataFrame()
        sid=dat_pgg.subject_id[0].lstrip(' -').rstrip(' -')
        tmp_pgg.loc[sid,'subject_id']=sid
        tmp_pgg.loc[sid,'wave']=wave
        
        if wave == 1:
            tmp_pgg.loc[sid,'invest']=dat_pgg[dat_pgg['trial_tag']=='investFB'].response_feedback.iloc[0]
            tmp_pgg.loc[sid,'rt']=dat_pgg[dat_pgg['trial_type']=='survey-text'].rt.iloc[0]
        else:
            tmp_pgg.loc[sid,'invest']=dat_pgg[dat_pgg['trial_tag']=='choice']['button_pressed'].iloc[0]
            tmp_pgg.loc[sid,'rt']=dat_pgg[dat_pgg['trial_tag']=='choice']['rt'].iloc[0]

        tmp_pgg.loc[sid,'totalTime']=dat_pgg.totalTime[0]/1000/60
        fname_part = fname.partition('/cvd_pgg/')
        fname_clean = fname_part[2]
        tmp_pgg.loc[sid,'fname']=fname_clean
        pgg = pd.concat([pgg,tmp_pgg])
    if len(pgg)>1:  
        # Eliminate later runs of duplicates
        pgg=pgg[~pgg['fname'].isin([i for i in pgg[pgg.index.duplicated()].fname])]
        
    # rename columns
    pgg.columns=['pgg_'+col for col in pgg.columns]
    pgg.reset_index(drop = True, inplace = True)
    return pgg

def process_cvd_altt(fdir, wave):
     # Load in and combine trust rating data
    cvd_altt_df=pd.DataFrame()
    cvd_altt_df["subject_id"] = ""
    for fname in glob.iglob(fdir+'/cvd_altt*.csv'):
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
        dat = dat.loc[dat['trial_tag']=='response',:]
        dat['wave']= wave
        dat['trial_nr']=range(1,len(dat)+1)
        dat['task'] = 'cvd_altt'
        
        
        dat.equal_resp = dat.equal_resp.str[0]
        dat.unequal_resp = dat.unequal_resp.str[0]
        dat.loc[dat.key_press == dat.equal_resp, 'response'] = 'EQU'
        dat.loc[dat.key_press == dat.unequal_resp, 'response'] = 'UNEQU'
        
        if np.isin('respRand', dat.columns).item():
            dat['respRand'] = True
        else:
            dat['respRand'] = False

                
        dat = dat[['task','subject_id','wave', 'trial_nr','trial_type','trial_index',
                   'time_elapsed','startTime','totalTime',
                   'rt','response', 'key_press','equal_resp', 'unequal_resp',
                   'choice_type', 'respRand','occupation', 'race', 'polit_ident',
                   'age', 'ageIdent']]
        
        # Eliminate later runs of duplicates
        if sum(cvd_altt_df.subject_id == sid) == 0:
            cvd_altt_df = pd.concat([cvd_altt_df,dat])
   
    # rename columns
    cvd_altt_df.columns=['cvd_altt_'+col for col in cvd_altt_df.columns]
    
    cvd_altt_df.reset_index(drop=True, inplace=True)

    return cvd_altt_df


def process_cvd_consp(fdir, wave):
     # Load in and combine trust rating data
    cvd_consp_df=pd.DataFrame()
    cvd_consp_df["subject_id"] = ""
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
        dat['wave']= wave
        dat['task'] = 'cvd_consp'
                
        # Eliminate later runs of duplicates
        if sum(cvd_consp_df.subject_id == sid) == 0:
            cvd_consp_df= pd.concat([cvd_consp_df,dat])
   
    # rename columns
    cvd_consp_df.columns=['cvd_consp_'+col for col in cvd_consp_df.columns]
    cvd_consp_df.reset_index(drop=True, inplace=True)

    return cvd_consp_df

def transform_pid_to_cvdid(data, pid_col_name, pid_to_cvdid):
    # replace PROLIFIC_PID with anonymous CVDID
    pid_to_cvdid = pid_to_cvdid.rename(columns={'PROLIFIC_PID': pid_col_name})
    data = data.merge(pid_to_cvdid, on =pid_col_name, how = 'left')
    data = data.drop(pid_col_name, axis=1)
    
    return data