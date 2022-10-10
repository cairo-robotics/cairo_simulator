import os
import json
import glob

import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

__all__ = ['ip_style_to_name', 'planning_bias_to_name', 'participant_to_name', 'import_data_as_dataframe', 'PlanningSuccessAnalysis', 'PlanningTimeAnalysis', 'PathLengthAnalysis', 'A2SConfigSpaceAnalysis', 'A2STaskSpaceAnalysis', 'A2FAnalysis']

def ip_style_to_name(ip_style):
    if ip_style == "kf":
        return "Keyframe Only"
    if ip_style == "opt":
        return "Optimization Only"
    if ip_style == "optkf":
        return "Optimization + Keyframe"

def planning_bias_to_name(planning_bias):
    if planning_bias == "1":
        return "Biased Planning"
    if planning_bias == "0":
        return "Unbiased Planning"

def participant_to_name(participant):
    return "Participant {}".format(participant)

def import_data_as_dataframe(data_path):
    path = os.path.join(data_path, "*.json")
    files_names = glob.glob(path)
    dfs = []
    for filene_name in files_names:
        with open(filene_name, 'r') as f:
            data = json.loads(f.read())
            dfs.append(pd.json_normalize(data, record_path=['trials'], meta=[
                'planning_bias',
                'ip_style'
            ]))
    return pd.concat(dfs)

class PlanningSuccessAnalysis():

    def __init__(self, dataframe):
        self.dataframe = dataframe


    # def bar_chart(self):

    #     df = self.dataframe[["participant", "planning_bias", "ip_style", "planning_time"]]
    #     df = df.set_index(['participant', 'planning_bias', 'ip_style'])     
    #     # plot dfl
    #     ax = sns.barplot(data=df)  # RUN PLOT   
    #     plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    #     plt.close()
        
    def analyze(self):
        df = self.dataframe[["participant", "planning_bias", "ip_style", "success"]]
        return df.groupby(['participant', 'planning_bias', 'ip_style']).mean()
class PlanningTimeAnalysis():

    def __init__(self, dataframe):
        self.dataframe = dataframe


    # def bar_chart(self):

    #     df = self.dataframe[["participant", "planning_bias", "ip_style", "planning_time"]]
    #     df = df.set_index(['participant', 'planning_bias', 'ip_style'])     
    #     # plot dfl
    #     ax = sns.barplot(data=df)  # RUN PLOT   
    #     plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    #     plt.close()
        
    def analyze(self):
        rslt_df = self.dataframe[self.dataframe["planning_time"] > 0]
        df = rslt_df[["participant", "planning_bias", "ip_style", "planning_time"]]
        return df.groupby(['participant', 'planning_bias', 'ip_style']).mean()
    
class PathLengthAnalysis():

    def __init__(self, dataframe):
        self.dataframe = dataframe


    # def bar_chart(self):

    #     df = self.dataframe[["participant", "planning_bias", "ip_style", "path_length"]]
    #     df = df.set_index(['participant', 'planning_bias', 'ip_style'])     
    #     # plot dfl
    #     ax = sns.barplot(data=df)  # RUN PLOT   
    #     plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    #     plt.close()
        
    def analyze(self):
        rslt_df = self.dataframe[self.dataframe["path_length"] > 0]
        df = rslt_df[["participant", "planning_bias", "ip_style", "path_length"]]
        return df.groupby(['participant', 'planning_bias', 'ip_style']).mean()

class A2SConfigSpaceAnalysis():

    def __init__(self, dataframe):
        self.dataframe = dataframe

    # def bar_chart(self):

    #     df = self.dataframe[["participant", "planning_bias", "ip_style", "path_length"]]
    #     df = df.set_index(['participant', 'planning_bias', 'ip_style'])     
    #     # plot dfl
    #     ax = sns.barplot(data=df)  # RUN PLOT   
    #     plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    #     plt.close()
        
    def analyze(self):
        rslt_df = self.dataframe[self.dataframe["a2s_cspace_distance"] > 0]
        df = rslt_df[["participant", "planning_bias", "ip_style", "a2s_cspace_distance"]]
        return df.groupby(['participant', 'planning_bias', 'ip_style']).mean()
    
class A2STaskSpaceAnalysis():

    def __init__(self, dataframe):
        self.dataframe = dataframe

    # def bar_chart(self):

    #     df = self.dataframe[["participant", "planning_bias", "ip_style", "path_length"]]
    #     df = df.set_index(['participant', 'planning_bias', 'ip_style'])     
    #     # plot dfl
    #     ax = sns.barplot(data=df)  # RUN PLOT   
    #     plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    #     plt.close()
        
    def analyze(self):
        rslt_df = self.dataframe[self.dataframe["a2s_taskspace_distance"] > 0]
        df = rslt_df[["participant", "planning_bias", "ip_style", "a2s_taskspace_distance"]]
        return df.groupby(['participant', 'planning_bias', 'ip_style']).mean()
    
class A2FAnalysis():

    def __init__(self, dataframe):
        self.dataframe = dataframe

    # def bar_chart(self):

    #     df = self.dataframe[["participant", "planning_bias", "ip_style", "path_length"]]
    #     df = df.set_index(['participant', 'planning_bias', 'ip_style'])     
    #     # plot dfl
    #     ax = sns.barplot(data=df)  # RUN PLOT   
    #     plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    #     plt.close()
        
    def analyze(self):
        rslt_df = self.dataframe[self.dataframe["a2f_percentage"] > 0]
        df = rslt_df[["participant", "planning_bias", "ip_style", "a2f_percentage"]]
        return df.groupby(['participant', 'planning_bias', 'ip_style']).mean()

