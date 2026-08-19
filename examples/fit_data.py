import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_directory))

from cme.utils import fit_model as fm

os.chdir(current_directory)
os.makedirs("export", exist_ok=True)
print(f"The current working directory is: {current_directory}")


folder = f"{current_directory}/data"
#folder = f"{current_directory}/data/Double_2017"

datasets = [
            "sim_Single Stage - Fast RT_2",
            "sim_Single Stage - Slow RT_2",
            #"rpm"
            ]

# 0.1 -> no scale
# 0.2 -> no scale, 11 states

model_details = fm.ModelDetails(folder=folder,
                 file_pre="",
                 data=datasets,
                 version=0.2,
                 n_states=[11],
                 start_width=2,
                 response_width=[1],
                 delta=1,
                 measurement_prob=0.7,
                 predictive_n = 10,
                 batch_size=2, #25
                 num_warmup = 10,
                 samples_n = 10,
                 num_chains = 1,
                 max_tree_depth = 10,
                 params_type="Centralized",
                 model_type=["Markov","Quantum"],
                 transition_type="TIMESTEP",
                 likelihood_type="SINGLE",
                 estimation_type="MCMC",
                 execution_type="Both",
                 sampling_type="GEN",
                 scale=None,#"SQRT", #
                 conf_scale=[None],
                 csv_header = False,
                 is_test = False,
                 is_parallel=False
                 )

fm.fit_model(model_details)
