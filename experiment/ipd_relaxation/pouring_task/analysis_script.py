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
    print()
    ##########################
    # Planning Time Analysis #
    ##########################
    pta = PlanningTimeAnalysis(df)
    pts_mean, pta_std = pta.analyze()
    print(pts_mean)
    print(pta_std)
    print()
    ########################
    # Path Length Analysis #
    ########################
    pla = PathLengthAnalysis(df)
    pla_mean, pla_std = pla.analyze()
    print(pla_mean)
    print(pla_std)
    print()
    #############################
    # A2S Config Space Analysis #
    #############################
    a2scs = A2SConfigSpaceAnalysis(df)
    a2scs_mean, a2scs_std = a2scs.analyze()
    print(a2scs_mean)
    print(a2scs_std)
    print()
    ###########################
    # A2S Task Space Analysis #
    ###########################
    a2sts = A2STaskSpaceAnalysis(df)
    a2sts_mean, a2sts_std = a2sts.analyze()
    print(a2sts_mean)
    print(a2sts_std)
    print()
    ################
    # A2F Analysis #
    ################
    a2f = A2FAnalysis(df)
    a2f_mean, a2f_std = a2f.analyze()
    print(a2f_mean)
    print(a2f_std)
    
    