import pandas as pd
import glob as gl
import pickle
import numpy as np
def collect_dataframes(file_pre, file_post, file_list, size=None, version="", batch_size=0, header=True, is_glob=False):
    if not is_glob: 
        df = (pd.concat([pd.read_csv(f"{file_pre}{d}{file_post}",header="infer" if header else header)
                           .assign(dataset=d,size=size)
                            for d in file_list], axis=0)
                           )
        
        return df
    else:
        return pd.concat(
            [
                pd.read_csv(f"{f}",header="infer" if header else header)
                  .assign(size=size, subfile_id = f.split(str(version)+"_")[1].removesuffix(".csv"))  
                  .assign(dataset=d)
                  #.assign(dims = lambda df:df.params.str.split("[", expand=True)[1].str.removesuffix("]")) 
                  #.sort_values(["subfile_id", "part_id", "items"])
                  .assign(id = lambda df: df.part_id.astype(int) + (df.subfile_id.astype(int) * (batch_size)))
                  .reset_index(drop=True) 
                for d in file_list
                for f in gl.glob(file_pre + d + file_post + "_*.csv") 
            ]).reset_index(drop=True)
    
def collect_response_from_model_output(folder, model, version, datasets, batch_size):
    def make_dataframe(arr, indicator, folder, dataset):

        return (pd.DataFrame(arr.squeeze())
                .assign(dataset = dataset,
                        file = indicator.replace(folder+"/mcmc_samples_","").removesuffix(".pkl"),
                        subfile_id = indicator.split(f"{version}_")[1].removesuffix(".pkl")
                        ))

    #dataset = indicator.split("_")[3], 
    #size = indicator.split("_")[4], 
    #subfile_id = indicator.split(f"{version}_")[1]
    keys = ['post_samples', 'phi_t', 'mean_conf', 'prior_pd_samples', 'post_pd_samples', 'RT', 'X'] #, 
    keys_process = ["RT", "X", "mean_conf"]
    keys_process_dict = {key: [] for key in keys}
    dataset_dict = {}
    for dataset in datasets:

        keys_dict = {key: [] for key in keys}

        pkl_file = f"{folder}/mcmc_samples_{dataset}_{model}_{version}_*.pkl"
        pkl_files = gl.glob(pkl_file)

        for file in pkl_files: 
            with open(file, "rb") as pkl:
                    model_out = pickle.load(pkl)
                    for key in keys:
                        if key in keys_process:
                            keys_process_dict[key].append(make_dataframe(model_out[key], file, folder, dataset))
                        else:
                            keys_dict[key].append(model_out[key])
        dataset_dict[dataset] = keys_dict
    df_observed_rt = pd.concat(keys_process_dict["RT"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + df.subfile_id.astype(int) * batch_size).drop(["part_id", "subfile_id"], axis=1)
    df_observed_ra = pd.concat(keys_process_dict["X"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + df.subfile_id.astype(int) * batch_size).drop(["part_id", "subfile_id"], axis=1)      
    df_mean_final_conf = pd.concat(keys_process_dict["mean_conf"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + df.subfile_id.astype(int) * batch_size).drop(["part_id", "subfile_id"], axis=1)
    return df_observed_rt, df_observed_ra, df_mean_final_conf, dataset_dict


