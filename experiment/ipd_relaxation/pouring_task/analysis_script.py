import os
import json

from cairo_planning.evaluation.analysis import *

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    participants = ["1"]
    data = {}
    for participant in participants:
        INPUT_DIR = os.path.join(FILE_DIR, "participant_{}/output".format(participant))
        print(INPUT_DIR)
        data[participant] = import_data_as_dataframe(INPUT_DIR)
    
    #############################
    # Planning Success Analysis #
    #############################
    for participant, participant_df in data.items():
        psa = PlanningSuccessAnalysis(participant_df)
        print(psa.analyze())

    ##########################
    # Planning Time Analysis #
    ##########################
    for  participant, participant_df in data.items():
        pta = PlanningTimeAnalysis(participant_df)
        print(pta.analyze())
    
    ########################
    # Path Length Analysis #
    ########################
    for  participant, participant_df in data.items():
        pla = PathLengthAnalysis(participant_df)
        print(pla.analyze())
        
    #############################
    # A2S Config Space Analysis #
    #############################
    for  participant, participant_df in data.items():
        a2scs = A2SConfigSpaceAnalysis(participant_df)
        print(a2scs.analyze())
    
    ###########################
    # A2S Task Space Analysis #
    ###########################
    for  participant, participant_df in data.items():
        a2sts = A2STaskSpaceAnalysis(participant_df)
        print(a2sts.analyze())
    
    ################
    # A2F Analysis #
    ################
    for  participant, participant_df in data.items():
        a2f = A2FAnalysis(participant_df)
        print(a2f.analyze())
    
    