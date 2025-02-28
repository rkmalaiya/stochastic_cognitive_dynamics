from cme.utils import fit_model as fm
import os
current_directory = os.getcwd()
print(f"The current working directory is: {current_directory}")


folder = f"{current_directory}/data"
#folder = f"{current_directory}/data/Double_2017"

datasets = [
            "sim_Single Stage - Fast RT_20",
            "sim_Single Stage - Slow RT_20",              
            #"rpm"
            ]

# 0.1 -> no scale
# 0.2 -> no scale, 11 states

model_details = fm.ModelDetails(folder=folder,
                 file_pre="",
                 file_posts=datasets,
                 version=0.2,
                 n_states=11,
                 start_width=2,
                 response_width=1,
                 delta=1,
                 measurement_prob=0.7,
                 predictive_n = 50, 
                 batch_size=20, #25
                 num_warmup = 1000, 
                 samples_n = 1500,
                 params_type="Centralized",
                 model_type="Quantum",
                 transition_type="TIMESTEP",
                 likelihood_type="SINGLE",
                 sampling_type="GEN",
                 scale=None,#"SQRT", #
                 csv_header = False,
                 is_test = False
                 )

fm.fit_model(model_details)
