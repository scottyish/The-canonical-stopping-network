#!/usr/bin/env python
# coding: utf-8

# # Run GLMS

# In[1]:


import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import numpy as np
import shutil
import itertools
import statannot

import glob
import re
import seaborn as sns
import statsmodels.api as sm
import statsmodels
import scipy
from scipy import signal
import nideconv
from nideconv import GroupResponseFitter, HierarchicalBayesianModel
from scipy import signal

import pickle as pkl
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.backends.backend_pdf

get_ipython().run_line_magic('matplotlib', 'inline')

def flatten(t):
    return [item for sublist in t for item in sublist]


# ## functions

# In[2]:


dataset = 'Leipzig_7T_SM'
model_n = '0'

if dataset == 'Leipzig_7T_GdH':
    task = 'stop'
    tr = 2.0
    runs = [1,2,3]
    excluded_runs = [('GAIT',task, '2'),
                 ('GAIT',task, '3'),
                 ('NM3T',task, '2'),
                 ('PF5T',task, '1')]
    
elif dataset == 'Leipzig_7T_SM':
    task = 'stop'
    tr = 3.0
    runs = [1,2,3]
    excluded_runs = [('12',task,'1'),
                     ('12',task,'2'),
                     ('12',task,'3'),
                     ('17',task,'3')]
    
elif dataset == 'NTNU_7T_SJSI':
    task = 'sst'
    tr = 1.38
    runs = [1,2]
    excluded_runs = []
    
elif dataset == 'aron_3T':
    task = 'stopsignal'
    tr = 2.0
    runs = [1,2,3]
    excluded_runs = []

elif dataset == 'openfmri_3T':
    task = 'stopsignal'
    tr = 2.0
    runs = [1]
    excluded_runs = [('10570',task,'1')]


# In[3]:


def get_events(events, model_n=0):
    if model_n == 0:
        events = events.loc[events['event_type'].isin(['fs', 'ss','go'])]
        
#     elif model_n == 2:
#         events.loc[events['event_type'].isin(['cue_ACC', 'cue_SPD']), 'event_type'] = 'cue'
#     elif model_n == 3:
#         events = events.loc[events.event_type!='feedback_PE']
        
    # always remove stimulus and feedback - those are automatically re-added
#     events = events.loc[~events.event_type.isin(['stimulus', 'feedback'])]
    # model 0 returns all events
    return events


# In[4]:


def fit_glm(timeseries, events, confounds, include_rois, model_n=0, t_r=1.38, oversample_design_matrix=10, concatenate_runs=False, fit_type='ols'):
    events = get_events(events, model_n=model_n)
    
    events_1 = events.reset_index().set_index(['subject', 'run', 'event_type']) #.loc[(slice(None), slice(None), include_events),:]
    # events_1 = events_1.reset_index().set_index(['subject', 'run', 'event_type'])
    events_1.onset -= t_r/2   # stc

    print(events_to_run:=['fs','ss','go'])
    
    glm1 = GroupResponseFitter(timeseries.copy()[include_rois],
                               events_1,
                               confounds=confounds.copy().reset_index() if confounds is not None else None,
                               input_sample_rate=1/t_r, 
                               oversample_design_matrix=oversample_design_matrix,
                               concatenate_runs=concatenate_runs)
    
    for event_type in events_to_run:

        glm1.add_event(event_type, basis_set='canonical_hrf_with_time_derivative',  interval=[0, 18], show_warnings=False)

    glm1.fit(type=fit_type)
    return glm1

# def fit_single_roi(roi, df, events,  model_n):
#     os.makedirs(f'./fit_models/model-{model_n}', exist_ok=True)
#     with open(f'./fit_models/model-{model_n}/_{roi}.pkl', 'wb') as f:
#         pkl.dump('', f)
    
#     print(f'Fitting {roi}...')
#     glm1 = fit_glm(timeseries=df, events=events, confounds=None, include_rois=[roi], model_n=model_n)
    
#     model = HierarchicalBayesianModel.from_groupresponsefitter(glm1)
#     model.build_model(subjectwise_errors=True)
#     model.sample(chains=6, iter=2300, init_ols=True, n_jobs=6, warmup=300)
    
#     with open(f'./fit_models/model-{model_n}/_{roi}.pkl', 'wb') as f:
#         pkl.dump(model, f)
        
def fit_single_roi(roi, df, events, model_n, dataset, t_r, overwrite=False, confounds=None):

    os.makedirs(roi_dir:=f'../derivatives/hierarchical_roi_glm/{dataset}/model-{model_n}', exist_ok=True)
    if not os.path.isfile(roi_pkl:=os.path.join(roi_dir,f'{roi}.pkl')) or overwrite:

        with open(os.path.join(roi_dir,f'{roi}.pkl'), 'wb') as f:
            pkl.dump('', f)

        print(f'Fitting {roi}...')
        glm1 = fit_glm(timeseries=df, events=events, confounds=confounds, include_rois=[roi], model_n=model_n, t_r=t_r)

        model = HierarchicalBayesianModel.from_groupresponsefitter(glm1)
        model.build_model(subjectwise_errors=True)
        model.sample(chains=6, iter=2300, init_ols=True, n_jobs=6, warmup=300)
        
        stan_summary = model._model.results.stansummary()
        group_traces = model._model.get_group_traces()
        mean_group_timecourse = model.get_mean_group_timecourse()
        mean_subject_timecourse = model.get_mean_subject_timecourses()
        subject_traces = model.get_subject_timecourse_traces()
        group_params = model._model.get_group_parameters()
        dataframe = model._model.results.to_dataframe()

        # save everything
        with open(os.path.join(roi_dir,f'{roi}.pkl'), 'wb') as f:
            pkl.dump(model, f)
            
        # save dataframe
        with open(os.path.join(roi_dir,f'{roi}_dataframe.pkl'), 'wb') as e:
            pkl.dump(dataframe, e)    
            
        # save summary only
        with open(os.path.join(roi_dir,f'{roi}_stansum.pkl'), 'wb') as k:
            pkl.dump(stan_summary, k)
                    
        # save traces
        with open(os.path.join(roi_dir,f'{roi}_traces.pkl'), 'wb') as g:
            pkl.dump(group_traces, g)
            
        # save mean group timecourse
        with open(os.path.join(roi_dir,f'{roi}_group_timecourses.pkl'), 'wb') as a:
            pkl.dump(mean_group_timecourse, a)
            
        # save mean subject timecourse
        with open(os.path.join(roi_dir,f'{roi}_subject_timecourses.pkl'), 'wb') as b:
            pkl.dump(mean_subject_timecourse, b)
            
        # save subject traces
        with open(os.path.join(roi_dir,f'{roi}_subject_traces.pkl'), 'wb') as c:
            pkl.dump(subject_traces, c)
            
        # save group params
        with open(os.path.join(roi_dir,f'{roi}_group_params.pkl'), 'wb') as d:
            pkl.dump(group_params, d)


    else:
        print(f'{roi_pkl} already run.. skipping..')


# In[5]:


def is_interactive():
    import __main__ as main
    return not hasattr(main, '__file__')
is_interactive()


# In[6]:


if __name__ == '__main__' and not is_interactive():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('roi', type=str)
    parser.add_argument('model_n', type=int)
    parser.add_argument('t_r', type=float)
    args = parser.parse_args()
    
    signals = pd.read_pickle(f'../derivatives/hierarchical_roi_glm/{dataset}/timeseries/{dataset}_timeseries.pkl')
    events = pd.read_pickle(f'../derivatives/hierarchical_roi_glm/{dataset}/events/{dataset}_events.pkl')
#     confounds = pd.read_pickle(f'../derivatives/hierarchical_roi_glm/{dataset}/confounds/{dataset}_confounds.pkl')
    confounds=None
    dataset = 'Leipzig_7T_SM'
    overwrite = True
    fit_single_roi(args.roi, signals, events, args.model_n, dataset, args.t_r, overwrite, confounds)


# In[7]:


if is_interactive():
    # call fit
    import subprocess
    import itertools
    os.system('jupyter nbconvert --to script 07b_hierarchical_ROI_GLM_SM.ipynb')  # make script from this notebook
    
    gm_nuclei = ['IFG','SMA','M1','SN','STN','GPe','Tha']
    include_rois = [roi + '-' + hemi for roi in gm_nuclei for hemi in ['l', 'r']]

    models = [0]
    to_fit = [x + (tr,) for x in list(itertools.product(include_rois, models))]

    def call_shell(roi, model_n, t_r):
        subprocess.run(["ipython", "07b_hierarchical_ROI_GLM_SM.py", roi, str(model_n), str(t_r)]) 

    import joblib
    joblib.Parallel(n_jobs=3)(joblib.delayed(call_shell)(roi, model_n, t_r) for roi, model_n, t_r in to_fit)


# # -----------

# In[12]:


# signals = pd.read_pickle(f'../derivatives/hierarchical_roi_glm/{dataset}/timeseries/{dataset}_timeseries.pkl')
# events = pd.read_pickle(f'../derivatives/hierarchical_roi_glm/{dataset}/events/{dataset}_events.pkl')


# In[2]:


# def to_psc(x): # calculate percent signal change
#     return x / x.mean() * 100 - 100

# def load_events_confounds(sub, dataset, task, run, include_physio=True, include_cosines=True):
#     event_fn = f'../derivatives/event_files/{dataset}/sub-{sub}/func/sub-{sub}_task-{task}_run-{run}_events.tsv'
# #    regressor_fn = f'../derivatives/behavior/sub-{sub}/ses-{ses}/func/sub-{sub}_ses-{ses}_task-{task}_run-{run}_desc-model-regressors.tsv'
#     confounds_fn = f'../derivatives/fmriprep/{dataset}/fmriprep/fmriprep/sub-{sub}/func/sub-{sub}_task-{task}_run-{run}_desc-confounds_timeseries.tsv'
    
#     events = pd.read_csv(event_fn, sep='\t', index_col=None)
#     events['duration'] = .001
            
#     # get confounds, cosines
#     confounds = pd.read_csv(confounds_fn, sep='\t').fillna(method='bfill')
#     include_confs = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z', 'dvars', 'framewise_displacement', 'global_signal']
#     if include_cosines:
#         include_confs += [x for x in confounds.columns if 'cos' in x]
#     confounds = confounds[include_confs]
    
#     if include_physio:
#         run_idx = run#+1
#         # just use acompcor
#         a_comp_cor = pd.read_csv(confounds_fn, sep='\t')[['a_comp_cor_' + str(x).zfill(2) for x in range(20)]]

#     return events, confounds

# def load_timeseries(atlas_type='MASSP', dataset='Leipzig_7T_SM', task='msit'):

#     signal_fns = sorted(glob.glob(f'../derivatives/extracted_signals/{dataset}/sub-*/func/*task-{task}*{atlas_type}-signals*.tsv'))

#     regex = re.compile(f'.*sub-(?P<sub>\d+)_task-{task}_run-(?P<run>\d)_desc-{atlas_type}-signals.tsv')
#     dfs = []
#     for signal_fn in signal_fns:
#         signals = pd.read_csv(signal_fn, sep='\t')
#         gd = regex.match(signal_fn).groupdict()

#         if 'time' in signals.columns:
#             signals = signals.rename(columns={'time': 'volume'})


#         signals = signals.set_index(['volume']).apply(to_psc).reset_index()  # to PSC
#         signals['time'] = signals['volume'] * 1.38

#         del signals['volume']
#         signals['sub'] = gd['sub']
#         signals['run'] = int(gd['run'])
#         signals = signals.set_index(['sub', 'run', 'time'])
#         dfs.append(signals)

#     df = pd.concat(dfs)
    
#     return df

# def sort_data(df, ses, task):
    
#     all_events = []
#     all_confounds = []

#     for sub, run in df.reset_index().set_index(['sub', 'run']).index.unique():
#     #     if sub == '010' and run == '2':    # no RETROICOR
#     #         continue
# #         if sub == '007' and run == '1':    # no RETROICOR
# #             continue
#         events, confounds = load_events_confounds(sub, ses, task, run, include_physio=True)
#         events['sub'] = sub
#         events['run'] = run
#         confounds['sub'] = sub
#         confounds['run'] = run

#         all_events.append(events)
#         all_confounds.append(confounds)

#     events = pd.concat(all_events).set_index(['sub', 'run'])
#     confounds = pd.concat(all_confounds).set_index(['sub', 'run'])

#     events = events.rename(columns={'trial_type': 'event_type'})

#     events['duration'] = 0.001  # stick function
        
#     # make psc
#     confounds['global_signal'] = (confounds['global_signal']-confounds['global_signal'].mean())/confounds['global_signal'].std()

#     # change subject number ordering
#     subs = df.reset_index()['sub'].unique()
#     mapping = {y:x+1 for x,y in enumerate(subs)}

#     events = events.reset_index()
#     events['sub'] = events['sub'].replace(mapping).astype(str)
#     events = events.set_index(['sub', 'run'])

#     df = df.reset_index()
#     df['sub'] = df['sub'].replace(mapping).astype(str)
#     df = df.set_index(['sub', 'run', 'time'])

#     confounds = confounds.reset_index()
#     confounds['sub'] = confounds['sub'].replace(mapping).astype(str)
#     confounds = confounds.set_index(['sub', 'run'])

#     events.index = events.index.rename('subject', level=0)
#     df.index = df.index.rename('subject', level=0)
#     confounds.index = confounds.index.rename('subject', level=0)
    
#     return events, df, confounds


# In[3]:


# def butter_highpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
#     return b, a

# def butter_highpass_filter(data, cutoff, fs, order=5):
#     b, a = butter_highpass(cutoff, fs, order=order)
#     y = signal.filtfilt(b, a, data)
#     return y


# ## start running

# In[4]:


# # which dataset?
# dataset = 'Leipzig_7T_SM'
# model_n = '0'
# # dataset = 'Leipzig_7T_GdH'
# # dataset = 'NTNU_7T_SJSI'


# In[5]:


# if dataset == 'Leipzig_7T_GdH':
#     task = 'stop'
#     t_r = 2.0
#     runs = [1,2,3]
#     excluded_runs = [('GAIT',task, '2'),
#                  ('GAIT',task, '3'),
#                  ('NM3T',task, '2'),
#                  ('PF5T',task, '1')]
    
# elif dataset == 'Leipzig_7T_SM':
#     task = 'stop'
#     t_r = 3.0
#     runs = [1,2,3]
#     excluded_runs = [('12',task,'1'),
#                      ('12',task,'2'),
#                      ('12',task,'3'),
#                      ('17',task,'3')]
    
# elif dataset == 'NTNU_7T_SJSI':
#     task = 'sst'
#     t_r = 1.38
#     runs = [1,2]
#     excluded_runs = []


# In[6]:


# # collect together all regions of interest for each sub.
# subs = [x.split('/')[-1].split('-')[-1] for x in glob.glob(f'../derivatives/extracted_signals/{dataset}/sub-*')]
# print(len(subs)) 

# for sub in subs:
# #     print(sub)
#     for run in runs:
# #         print(run)
#         all_sigs = []#pd.DataFrame()
#         signals = sorted(glob.glob(f'../derivatives/extracted_signals/{dataset}/sub-{sub}/func/*task-{task}_run-{run}*_hp.tsv'))
#         if signals == []:
#             print(f'no data for sub {sub} run {run} task {task}')
#             continue
#         signals = [x for x in signals if 'ALL' not in x]
#         for index, sig in enumerate(signals):
#             if index == 0:
#                 all_sigs = pd.read_csv(sig, sep='\t')
#             else:
#                 all_sigs = pd.concat([all_sigs, pd.read_csv(sig, sep='\t')],axis=1)

#         all_sigs = all_sigs.loc[:,~all_sigs.columns.duplicated()]
#         all_sigs.to_csv(sig.replace('harvard','ALL'),'\t') # replace last loaded mask name with ALL


# In[7]:


# ## Load timeseries
# atlas_type = 'ALL'
# signal_fns = sorted(glob.glob(f'../derivatives/extracted_signals/{dataset}/sub-*/func/*task-{task}*{atlas_type}*hp.tsv'))
# # signal_fns = [x for x in signal_fns if not any(s in x for s in to_remove)] # remove subs that didnt do task right
# signal_fns

# filter_out_confounds = True
# filter_hp = False # already highpassed so no need

# regex = re.compile(f'.*sub-(?P<sub>[a-zA-Z0-9]+)_task-(?P<task>\S+)_run-(?P<run>\d)_desc-{atlas_type}-signals_hp.tsv')
# dfs = []
# for signal_fn in signal_fns:
#     signals = pd.read_csv(signal_fn, sep='\t',index_col=0)
#     signals = signals.loc[:, ~signals.columns.str.contains('^Unnamed')]
#     gd = regex.match(signal_fn).groupdict()
# #     print(tuple(gd.values()))
#     if tuple(gd.values()) in excluded_runs:
#         # run was excluded
#         print(tuple(gd.values()))
#         continue
    
#     if 'time' in signals.columns:
#         # if there's a column named 'time', it's called 'time' but it's really volume number..
#         signals = signals.rename(columns={'time': 'volume'})
#     signals = signals.set_index('volume')
# #     print(signals)
#     # psc?
#     signals = signals.apply(to_psc)  # to PSC
# #     print(signals)

#     # filter out confounds?
#     if filter_out_confounds:
#         _, confounds = load_events_confounds(sub=gd['sub'], dataset=dataset, task=gd['task'], run=gd['run'])
#         confounds['intercept'] = 1   # add intercept!
#         betas, residuals, rank, s, = np.linalg.lstsq(a=confounds, b=signals)
#         signals_hat = confounds@betas
#         signals_hat.index = signals.index
#         signals_hat.columns = signals.columns
#         signals -= signals_hat   # residuals
        
#     # high pass?
#     if filter_hp:
#         signals = signals.apply(lambda x: butter_highpass_filter(x, cutoff=1/128, fs=1/t_r) + x.mean(), axis=0)
    
#     # index to time
#     signals.index *= t_r
#     signals.index.name = 'time'
#     signals['subject'] = gd['sub']
#     signals['run'] = int(gd['run'])
        
#     signals = signals.reset_index().set_index(['subject', 'run', 'time'])
#     dfs.append(signals)
# df = pd.concat(dfs)
# df = df.reindex(sorted(df.columns), axis=1)

# if atlas_type == 'ALL':
#     df.rename(columns = {'lM1':'M1-l', 'rM1':'M1-r', 'gpel':'GPe-l','gper':'GPe-r','gpil':'GPi-l','gpir':'GPi-r',
#                          'rnl':'RN-l','rnr':'RN-r','snl':'SN-l','snr':'SN-r','stnl':'STN-l','stnr':'STN-r',
#                          'strl':'Str-l','strr':'Str-r','thal':'Tha-l','thar':'Tha-r','vtal':'VTA-l','vtar':'VTA-r'}, inplace = True)
# #     rois_ = sorted(['IFG-l','IFG-r','ACC-l','ACC-r','M1-l','M1-r','SMA-l','SMA-r','PaCG-l','PaCG-r','Ins-l','Ins-r','GPe-l','GPe-r','GPi-l','GPi-r','RN-l','RN-r','SN-l','SN-r',
# #              'STN-l','STN-r','Str-l','Str-r','Tha-l','Tha-r','VTA-l','VTA-r'])
#     rois_ = ['ACC-l','ACC-r','IFG-l','IFG-r','Ins-l','Ins-r','M1-l','M1-r','pSG-l','pSG-r','SMA-l','SMA-r','SPL-l','SPL-r','GPe-l','GPe-r','GPi-l','GPi-r','RN-l','RN-r','SN-l','SN-r',
#              'STN-l','STN-r','Str-l','Str-r','Tha-l','Tha-r','VTA-l','VTA-r']
#     # all cortical masks from harvard-oxford atlas
#     # all subcortical masks from MASSP atalas
#     df = df[rois_]
    
# elif atlas_type == 'ATAG':
#     rois_ = ['lM1','rM1','lPreSMA','rPreSMA','rIFG','lSTN','rSTN','lSN','rSN']#,'ACC','THA']
#     df = df[rois_]

# os.makedirs(timeseries_dir:=f'../derivatives/hierarchical_roi_glm/{dataset}/timeseries', exist_ok=True)
# with open(os.path.join(timeseries_dir, f'{dataset}_timeseries.pkl'), 'wb') as f:
#             pkl.dump('', f)
# with open(os.path.join(timeseries_dir, f'{dataset}_timeseries.pkl'), 'wb') as f:
#             pkl.dump(df, f)
    
# df.head()


# ## load event, confounds

# In[8]:


# # model_n = 'all_events'
# all_events = []
# all_confounds = []

# for sub, run in df.reset_index().set_index(['subject', 'run']).index.unique():
#     events, confounds = load_events_confounds(sub, dataset, task, run, include_physio=True)
#     events['subject'] = sub
#     events['run'] = run
#     confounds['subject'] = sub
#     confounds['run'] = run
# #     print(sub,run)
    
#     all_events.append(events)
#     all_confounds.append(confounds)
    
# # print(events)
# events = pd.concat(all_events).set_index(['subject', 'run'])
# confounds = pd.concat(all_confounds).set_index(['subject', 'run'])
# # print(events)

# events = events.rename(columns={'trial_type': 'event_type'})

# events['duration'] = 0.001  # stick function

# os.makedirs(event_dir:=f'../derivatives/hierarchical_roi_glm/{dataset}/events', exist_ok=True)
# with open(os.path.join(event_dir, f'{dataset}_events.pkl'), 'wb') as f:
#             pkl.dump('', f)
# with open(os.path.join(event_dir, f'{dataset}_events.pkl'), 'wb') as f:
#             pkl.dump(events, f)


# ## glm function

# In[16]:


# def fit_glm(timeseries, events, confounds, include_rois, model_n=0, t_r=1.38, oversample_design_matrix=10, concatenate_runs=False, fit_type='ols'):
# #     events = get_events(events, model_n=model_n)
    
#     events_1 = events.reset_index().set_index(['subject', 'run', 'event_type']) #.loc[(slice(None), slice(None), include_events),:]
#     # events_1 = events_1.reset_index().set_index(['subject', 'run', 'event_type'])
#     events_1.onset -= t_r/2   # stc

#     glm1 = GroupResponseFitter(timeseries.copy()[include_rois],
#                                events_1,
#                                confounds=confounds.copy().reset_index() if confounds is not None else None,
#                                input_sample_rate=1/t_r, 
#                                oversample_design_matrix=oversample_design_matrix,
#                                concatenate_runs=concatenate_runs)
#     for event_type in events_1.reset_index().event_type.unique():
#         if event_type in ['fs','ss','go']:
#             glm1.add_event(event_type, basis_set='canonical_hrf_with_time_derivative', interval=[0, 19.32], show_warnings=False)

#     glm1.fit(type=fit_type)
#     return glm1

# def fit_single_roi(roi, df, events,  model_n, dataset, overwrite=True):

#     os.makedirs(roi_dir:=f'../derivatives/hierarchical_roi_glm/{dataset}/model-{model_n}', exist_ok=True)
#     if not os.path.isfile(roi_pkl:=os.path.join(roi_dir,f'{roi}.pkl')) or overwrite:

#         with open(os.path.join(roi_dir,f'{roi}.pkl'), 'wb') as f:
#             pkl.dump('', f)

#         print(f'Fitting {roi}...')
#         glm1 = fit_glm(timeseries=df, events=events, confounds=None, include_rois=[roi], model_n=model_n)

#         model = HierarchicalBayesianModel.from_groupresponsefitter(glm1)
#         model.build_model(subjectwise_errors=True)
#         model.sample(chains=6, iter=2300, init_ols=True, n_jobs=6, warmup=300)

#         with open(os.path.join(roi_dir,f'{roi}.pkl'), 'wb') as f:
#             pkl.dump(model, f)

#     else:
#         print(f'{roi_pkl} already run.. skipping..')
        
#     return model


# # fit glm and run hierarchically

# In[17]:


# if atlas_type == 'ATAG':
#     gm_nuclei = ['M1', 'PreSMA', 'IFG','STN','SN'] #'ACC',
#     include_rois = [hemi + roi for roi in gm_nuclei for hemi in ['l','r']]
#     include_rois.remove('lIFG')
# elif atlas_type=='MASSP':
#     gm_nuclei = ['Amg', 'Cl', 'GPe', 'GPi', 'PAG', 'PPN', 'RN', 'SN', 'STN', 'Str', 'Tha', 'VTA']
#     include_rois = [roi + '-' + hemi for roi in gm_nuclei for hemi in ['l', 'r']]
# elif atlas_type == 'CORT':
#     gm_nuclei = ['ACC','aSG','IFG','Ins','PaCG','pCC','postcG','precC','precGy','pSG','SMA','SPL']
#     include_rois = [roi + '-' + hemi for roi in gm_nuclei for hemi in ['l', 'r']]
# elif atlas_type == 'ALL':
#     #     gm_nuclei = sorted(['IFG','ACC','M1','SMA','PaCG','Ins','GPe','GPi','RN','SN','STN','Str','Tha','VTA'])
#     gm_nuclei = ['ACC','IFG','Ins','M1','pSG','SMA','SPL','GPe','GPi','RN','SN','STN','Str','Tha','VTA']
#     include_rois = [roi + '-' + hemi for roi in gm_nuclei for hemi in ['l', 'r']]

# # include_events=['response_left','response_right','fs','ss','go']
# # include_events = ['fs','ss','go']
# # glm1 = fit_glm(timeseries=df, events=events, confounds=confounds, include_events=include_events, include_rois=include_rois)

# for roi in include_rois[:1]:
    
#     this_mod = fit_single_roi(roi, df, events, model_n='0', dataset=dataset)
    


# In[11]:


# to_fit = [(x, model_n, dataset) for x in include_rois]
# to_fit


# In[13]:


# import joblib

# def mp_fit(tup):
    
#     roi = tup[0]
#     model_n = tup[1]
#     dataset = tup[2]
#     events = pd.read_pickle(f'../derivatives/hierarchical_roi_glm/{dataset}/events/{dataset}_events.pkl')
#     df = pd.read_pickle(f'../derivatives/hierarchical_roi_glm/{dataset}/timeseries/{dataset}_timeseries.pkl')
    
#     fit_single_roi(roi, df, events, model_n=model_n, dataset=dataset)

# joblib.Parallel(n_jobs=3)(joblib.delayed(mp_fit)(this_info) for this_info in to_fit)


# In[30]:


# events = pd.read_pickle('/home/Public/trondheim/scripts/hierarchical_bayesian_GLM/data/events.pkl')


# In[31]:


# events


# In[32]:


# events = pd.read_pickle(f'../derivatives/hierarchical_roi_glm/Leipzig_7T_SM//events/Leipzig_7T_SM_events.pkl')
# events


# # run hierarchically

# In[ ]:


# # call fit
# import subprocess
# import itertools

# # gm_nuclei = ['Amg', 'Cl', 'GPe', 'GPi', 'PAG', 'PPN', 'RN', 'SN', 'STN', 'Str', 'Tha', 'VTA']
# # include_rois = [roi + '-' + hemi for roi in gm_nuclei for hemi in ['l', 'r']]

# gm_nuclei = ['ACC','IFG','Ins','M1','pSG','SMA','SPL','GPe','GPi','RN','SN','STN','Str','Tha','VTA']
# include_rois = [roi + '-' + hemi for roi in gm_nuclei for hemi in ['l', 'r']]

# # models = [0,1,2,3]
# datasets = ['Leipzig_7T_SM']
# to_fit = list(itertools.product(include_rois, datasets))

# def call_shell(roi, model_n):
#     subprocess.run(["python", "fit_models.py", roi, str(model_n)]) 
    
# def run_hierarchical(roi, dataset):
    
#     all_events = []
#     all_confounds = []

#     for sub, run in df.reset_index().set_index(['subject', 'run']).index.unique():
#         events, confounds = load_events_confounds(sub, dataset, task, run, include_physio=True)
#         events['subject'] = sub
#         events['run'] = run
#         confounds['subject'] = sub
#         confounds['run'] = run
#     #     print(sub,run)

#         all_events.append(events)
#         all_confounds.append(confounds)

#     events = pd.concat(all_events).set_index(['subject', 'run'])
#     confounds = pd.concat(all_confounds).set_index(['subject', 'run'])
#     events = events.rename(columns={'trial_type': 'event_type'})
#     events['duration'] = 0.001  # stick function
    

# import joblib
# joblib.Parallel(n_jobs=6)(joblib.delayed(call_shell)(roi, model_n) for roi, model_n in to_fit)
# joblib.Parallel(n_jobs=6)(joblib.delayed(run_hierarchical)(roi, dataset) for roi, dataset in to_fit)


# # PLOT posteriors

# In[15]:


# dataset = 'Leipzig_7T_SM'
# model_n = '0'


# In[16]:


# rois_fitted = [x.split('/')[-1].split('.')[0] for x in sorted(glob.glob(f'../derivatives/hierarchical_roi_glm/{dataset}/model-{model_n}/*.pkl'))]
# rois_fitted


# In[17]:


# def get_difference_traces(event_a, event_b, model):
#     return model._model.get_group_traces()[event_a]['intercept']['HRF'] - model._model.get_group_traces()[event_b]['intercept']['HRF']

# def plot_traces(traces, n_chains=6, n_samples_per_chain=2000):
#     ###
#     n_col = traces.shape[1]
#     n_rows = int(np.floor(n_col/3))
#     if (n_col % 3) > 0:
#         n_rows += 1

#     ###
#     fig = plt.figure(figsize=(15,n_rows*3))
#     gs0 = fig.add_gridspec(n_rows, 3)

#     row_n = 0
#     for col_n, column_name in enumerate(traces.columns):
#         col_n = col_n % 3
#         # ax0 = fig.add_subplot(gs0[row_n, col_n])
#         gs1 = gs0[row_n, col_n].subgridspec(1, 2, width_ratios=[.7,.3], wspace=0)
#         ax00 = fig.add_subplot(gs1[0])
#         ax01 = fig.add_subplot(gs1[1], sharey=ax00)

#         for chain in range(n_chains):
#             ax00.plot(np.arange(n_samples_per_chain),traces.iloc[(chain*n_samples_per_chain):(chain*n_samples_per_chain+n_samples_per_chain)][column_name])
#             ax00.set_title(column_name[0] + ' ' + column_name[2])

#         sns.kdeplot(traces.iloc[(chain*n_samples_per_chain):(chain*n_samples_per_chain+n_samples_per_chain)][column_name], vertical=True, ax=ax01)
#         ax01.axis('off')
#         ylims = ax01.get_ylim()
#         pdf = stats.norm.pdf(np.arange(ylims[0],ylims[1], .01), 0, 2.5)
#         ax01.plot(pdf, np.arange(ylims[0],ylims[1], .01)) # including h here is crucial

#         if col_n == 2:
#             row_n += 1

#     fig.tight_layout()
    
#     return fig


# In[22]:


# model


# In[23]:


# do_plot_traces = False
# all_traces = []

# for roi in rois_fitted:
#     with open(f'../derivatives/hierarchical_roi_glm/{dataset}/model-{model_n}/{roi}.pkl', 'rb') as f:
#         model = pkl.load(f)
    
#     if model == '':
#         print(roi)
#         continue
        
#     a = model._model.results.stansummary()
#     print('{}: Max Rhat: {}'.format(roi, pd.DataFrame([x.split() for x in a.split('\n')[5:-5]], columns=['parameter'] + a.split('\n')[4].split())['Rhat'].max()))
    
#     traces = model._model.get_group_traces()
# #     traces['response_right-response_left'] = traces[('response_right', 'intercept', 'HRF')] - traces[('response_left', 'intercept', 'HRF')]
# #     traces['contralateral_ipsilateral'] = traces['response_right-response_left'].copy()
# #     if roi.endswith('-r'):
# #         traces['contralateral_ipsilateral'] *= -1
    
#     traces['fs-go'] = traces[('fs', 'intercept', 'HRF')] - traces[('go', 'intercept', 'HRF')]
#     traces['fs-ss'] = traces[('fs', 'intercept', 'HRF')] - traces[('ss', 'intercept', 'HRF')] 
#     traces['ss-go'] = traces[('ss', 'intercept', 'HRF')] - traces[('go', 'intercept', 'HRF')] 
    
#     if do_plot_traces:
#         f = plot_traces(traces)
#         os.makedirs(f'../derivatives/{dataset}/traceplots', exist_ok=True)
#         f.savefig(f'../derivatives/{dataset}/traceplots/{roi}.pdf', bbox_inches='tight')
#         plt.close()
    
#     traces = traces.melt()
#     traces['roi'] = roi
#     all_traces.append(traces)
# all_traces = pd.concat(all_traces)


# ### Plot posterior distributions

# In[24]:


# # Get hemisphere & ROI separate
# all_traces['hemisphere'] = all_traces['roi'].apply(lambda x: x[-1])
# all_traces['roi_nohemi'] = all_traces['roi'].apply(lambda x: x[:-2])

# # some shortcuts for indexing
# # intercept_cov = (all_traces['covariate'] == 'modulation') & (all_traces['regressor'].isin(['canonical HRF', 'HRF']))
# # include_idx = (all_traces['event type'].isin(['contralateral_ipsilateral', 'cue_SPD-cue_ACC'])) | ((all_traces['event type'].isin(['stimulus_value_difference', 'feedback_PE'])) & intercept_cov)

# include_idx = all_traces['event type'].isin(['fs-go','fs-ss', 'ss-go'])
# # include_idx = all_traces['event type'].isin(['contralateral_ipsilateral', 'fs-go','fs-ss', 'ss-go'])
# # all_traces.loc[include_idx, ['event type', 'value', 'roi']].to_csv('./all_rois_posteriors.csv')  # why did I want to save this again?


# #### Get statistics of distributions

# In[25]:


# def get_quantiles(x):
#     return {'mean': x.mean(), 
#             '2.5%': np.quantile(x, .025), 
#             '97.5%': np.quantile(x, 0.975), 
#             '50%': np.quantile(x, 0.5), 
#             '0.5%': np.quantile(x, .005), 
#             '99.5%': np.quantile(x, 0.995)}


# In[26]:


# stats = all_traces[include_idx].groupby(['hemisphere', 'roi_nohemi', 'event type'])['value'].apply(get_quantiles).unstack().reset_index() #lambda x: {'mean': x.mean(), 'lower': np.quantile(x, .025), 'upper': np.quantile(x, 0.975)}).unstack().reset_index()
# stats['yerr_lower'] = stats['mean']-stats['2.5%']
# stats['yerr_upper'] = stats['97.5%']-stats['mean']

# include_values = ['2.5%', '50%', '97.5%']   # ['mean', '0.5%', '2.5%', '50%', '97.5%', '99.5%']
# # stats_table = stats.loc[stats['event type']!='stimulus_value_difference'].pivot_table(values=include_values, columns='event type', index=['roi_nohemi', 'hemisphere']) #.swaplevel()
# stats_table = stats.pivot_table(values=include_values, columns='event type', index=['roi_nohemi', 'hemisphere']) #.swaplevel()
# stats_table = stats_table.swaplevel(axis='columns').sort_index(axis='columns', level=0).round(4)
# # stats_table = stats_table.rename(columns={'contralateral_ipsilateral': 'Contralateral > ipsilateral motor response', 'fs-go': 'fs > go', 'fs-ss': 'fs > ss', 'ss-go': 'ss > go'})
# stats_table = stats_table.rename(columns={'fs-go': 'fs > go', 'fs-ss': 'fs > ss', 'ss-go': 'ss > go'})
# display(stats_table)
# # print(stats_table.to_latex())


# In[27]:


# f, ax = plt.subplots(2,2, figsize=(10,7))
    
# # for ax_, contrast_name, ylabel in zip(ax.ravel(), 
# #                               ['contralateral_ipsilateral', 'fs-go','fs-ss', 'ss-go'],
# #                               ['Contralateral > Ipsilateral motor response', 'fs > go intercept', 'fs > ss intercept', 'ss > go intercept']):
# for ax_, contrast_name, ylabel in zip(ax.ravel(), 
#                               ['fs-go','fs-ss', 'ss-go'],
#                               ['fs > go intercept', 'fs > ss intercept', 'ss > go intercept']):
#     sns.barplot(x='roi_nohemi', y='value', data=all_traces[include_idx & (all_traces['event type'] == contrast_name)], ci=None,
#                 ax=ax_, hue='hemisphere')
#     x_locs = []
#     for i in range(len(ax_.patches)):
#         x_locs.append(ax_.patches[i]._x0 + ax_.patches[i]._width/2)
#     ax_.errorbar(x_locs, 
#                  y=stats.loc[stats['event type'] == contrast_name, 'mean'].values, color='black', #yerr=0.01, 
#                  yerr=stats.loc[stats['event type'] == contrast_name, ['yerr_lower', 'yerr_upper']].values.T, #,yerr=0.01, 
#                  fmt='none', capsize=2)
#     ## asterices
#     ## 99% above
#     tmp = stats.loc[stats['event type'] == contrast_name]
#     idx = (tmp['0.5%'] > 0)
#     x_locs_asterices = np.array(x_locs)[idx]
#     y_locs_asterices = np.array(tmp['97.5%'])[idx]
#     for x,y in zip(x_locs_asterices, y_locs_asterices):
#         ax_.text(x,y,s='**', ha='center')
#     ## 95% above
#     idx = (tmp['2.5%'] > 0) & (tmp['0.5%'] < 0)
#     x_locs_asterices = np.array(x_locs)[idx]
#     y_locs_asterices = np.array(tmp['97.5%'])[idx]
#     for x,y in zip(x_locs_asterices, y_locs_asterices):
#         ax_.text(x,y,s='*', ha='center')

#     ## 99% below
#     tmp = stats.loc[stats['event type'] == contrast_name]
#     idx = (tmp['97.5%'] < 0)
#     x_locs_asterices = np.array(x_locs)[idx]
#     y_locs_asterices = np.array(tmp['2.5%']-0.001)[idx]
#     for x,y in zip(x_locs_asterices, y_locs_asterices):
#         ax_.text(x,y,s='**', ha='center', va='top')
#     ## 95% below
#     idx = (tmp['97.5%'] < 0) & (tmp['99.5%'] > 0)
#     x_locs_asterices = np.array(x_locs)[idx]
#     y_locs_asterices = np.array(tmp['2.5%']-0.001)[idx]
#     for x,y in zip(x_locs_asterices, y_locs_asterices):
#         ax_.text(x,y,s='*', ha='center', va='top')
        
#     ax_.axhline(0, linestyle='--', color='grey')
#     ax_.set_xlabel('')
#     ax_.set_ylabel('')
#     ax_.set_title(ylabel)
    
# #f, ax_ = plt.subplots(1,1)
# # ax_ = ax[-1, -1]
# # sns.barplot(x='roi_nohemi', y='r', data=corr_coefs, ci=None, ax=ax_, hue='hemisphere')
# # x_locs = []
# # for i in range(len(ax_.patches)):
# #     x_locs.append(ax_.patches[i]._x0 + ax_.patches[i]._width/2)
# # ax_.errorbar(x_locs, 
# #              y=stats2['mean'].values, color='black',              # yerr=0.01, 
# #              yerr=stats2[['yerr_lower', 'yerr_upper']].values.T,  # yerr=0.01, 
# #              fmt='none', capsize=2)
# # ## 99% above
# # idx = (stats2['0.5%'] > 0)
# # x_locs_asterices = np.array(x_locs)[idx]
# # y_locs_asterices = np.array(stats2['97.5%'] + 0.0075)[idx]
# # for x,y in zip(x_locs_asterices, y_locs_asterices):
# #     ax_.text(x,y,s='**', ha='center')
# # ## 95% above
# # idx = (stats2['2.5%'] > 0) & (stats2['0.5%'] < 0)
# # x_locs_asterices = np.array(x_locs)[idx]
# # y_locs_asterices = np.array(stats2['97.5%'] + 0.0075)[idx]
# # for x,y in zip(x_locs_asterices, y_locs_asterices):
# #     ax_.text(x,y,s='*', ha='center')
    
    
# #x_locs_asterices
# # ax_.text(x=x_locs_asterices, y=y_locs_asterices, s='test')

# # ax_.axhline(0, linestyle='--', color='grey')
# # ax_.set_xlabel('')
# # ax_.set_title('SPD > ACC cue covariance with threshold')
# # ax_.set_ylabel('')

# ax[0,0].set_ylabel('Percent signal change')
# ax[0,1].set_ylabel('Percent signal change')
# ax[1,0].set_ylabel('Percent signal change')
# # ax[1,1].set_ylabel('Correlation coefficient')

# # ax[0,0].legend(title='Hemisphere', loc='upper left', labels=['Left', 'Right'])
# ax[0,1].get_legend().remove()
# # ax[1,1].get_legend().remove()
# ax[1,0].get_legend().remove()

# ax[0,0].legend_.set_title('Hemisphere')
# ax[0,0].legend_.texts[0].set_text('Left')
# ax[0,0].legend_.texts[1].set_text('Right')

# for ax_ in ax.ravel():
#     ax_.grid(axis='y', linestyle='--')
#     ax_.set_axisbelow(True)

# sns.despine()
# f.tight_layout()

# # f.savefig('./GLM_subjectwise_errors_with_plausible_values_newmodel.pdf', bbox_inches='tight')


# In[ ]:





# In[ ]:




