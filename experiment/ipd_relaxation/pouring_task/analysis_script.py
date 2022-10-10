import os
import json

from cairo_planning.evaluation.analysis import *

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":

    #########################
    # CONSTANTS / VARIABLES #
    #########################
    EVAL_OUTPUT_DIRECTORY = os.path.join(FILE_DIR, "data/output_combined")
    df = import_data_as_dataframe(EVAL_OUTPUT_DIRECTORY)
    
    #############################
    # Planning Success Analysis #
    #############################
    psa = PlanningSuccessAnalysis(df)
    print(psa.analyze())

    ##########################
    # Planning Time Analysis #
    ##########################
    pta = PlanningTimeAnalysis(df)
    print(pta.analyze())
    
    ########################
    # Path Length Analysis #
    ########################
  
    pla = PathLengthAnalysis(df)
    print(pla.analyze())
        
    #############################
    # A2S Config Space Analysis #
    #############################
    a2scs = A2SConfigSpaceAnalysis(df)
    print(a2scs.analyze())
    
    ###########################
    # A2S Task Space Analysis #
    ###########################
    a2sts = A2STaskSpaceAnalysis(df)
    print(a2sts.analyze())
    
    ################
    # A2F Analysis #
    ################
    a2f = A2FAnalysis(df)
    print(a2f.analyze())
    
    