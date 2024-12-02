import pandas as pd
import glob as gl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cme.utils.common_logging as cm_log
log = cm_log.get_logger()
import itertools as iter

def collect_dataframes(file_pre, file_post, datasets, size=None, model = [], version="", batch_size=0, header=True, is_glob=False):
    if not is_glob: 
        df = (pd.concat([pd.read_csv(f"{file_pre}{d}{file_post}",header="infer" if header else header)
                           .assign(dataset=d,size=size)
                            for d in datasets], axis=0)
                           )
        
        return df
    else:
        log.info(f"Getting files from {file_pre + '*' + file_post + f'_{model}_{version}_*.csv'}")
        return pd.concat(
            [
                pd.read_csv(f"{f}",header="infer" if header else header)
                  .assign(size=size, subfile_id = f.split(str(version)+"_")[1].removesuffix(".csv"))  
                  .assign(dataset=d)
                  .assign(model = m)
                  #.assign(dims = lambda df:df.params.str.split("[", expand=True)[1].str.removesuffix("]")) 
                  #.sort_values(["subfile_id", "part_id", "items"])
                  .assign(part_id = lambda df: df.part_id if "part_id" in df.columns else df.index)
                  .assign(id = lambda df: df.part_id.astype(int) + (df.subfile_id.astype(int) * (batch_size)))
                  .reset_index(drop=True) 
                for d in datasets
                for m in model
                for f in gl.glob(file_pre + d + file_post + f"_{m}_{version}_*.csv") 
                
            ]).reset_index(drop=True)

def collect_response_from_model_output(folder, models, version, datasets, batch_size=0):
    def make_dataframe(arr, indicator, folder, dataset, model):

        return (pd.DataFrame(arr.squeeze())
                .assign(dataset = dataset, model = model,
                        file = indicator.replace(folder+"/mcmc_samples_","").removesuffix(".pkl"),
                        subfile_id = indicator.split(f"{version}_")[1].removesuffix(".pkl")
                        ))

    #dataset = indicator.split("_")[3], 
    #size = indicator.split("_")[4], 
    #subfile_id = indicator.split(f"{version}_")[1]
    keys = ['post_samples', 'phi_t','phi_0', "mean_init_conf", "mean_final_conf", 'prior_pd_samples', 'post_pd_samples', 'RT', 'X'] #, 
    keys_process = ["RT", "X", "mean_init_conf", "mean_final_conf"]
    keys_process_dict = {key: [] for key in keys}
    dataset_dict = {}
    for dataset, model in iter.product(datasets, models):

        keys_dict = {key: [] for key in keys}

        pkl_file = f"{folder}/mcmc_samples_{dataset}_{model}_{version}_*.pkl"
        pkl_files = gl.glob(pkl_file)

        for file in pkl_files: 
            with open(file, "rb") as pkl:
                    model_out = pickle.load(pkl)
                    for key in keys:
                        if key in keys_process:
                            keys_process_dict[key].append(make_dataframe(model_out[key], file, folder, dataset, model))
                        else:
                            keys_dict[key].append(model_out[key])
        dataset_dict[model+"_"+dataset] = keys_dict
    df_observed_rt = pd.concat(keys_process_dict["RT"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + df.subfile_id.astype(int) * batch_size).drop(["part_id", "subfile_id"], axis=1)
    df_observed_ra = pd.concat(keys_process_dict["X"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + df.subfile_id.astype(int) * batch_size).drop(["part_id", "subfile_id"], axis=1)      
    df_mean_init_conf = pd.concat(keys_process_dict["mean_init_conf"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + df.subfile_id.astype(int) * batch_size).drop(["part_id", "subfile_id"], axis=1)
    df_mean_final_conf = pd.concat(keys_process_dict["mean_final_conf"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + df.subfile_id.astype(int) * batch_size).drop(["part_id", "subfile_id"], axis=1)
    return df_observed_rt, df_observed_ra, df_mean_init_conf, df_mean_final_conf, dataset_dict


def get_predictive_plot(df_pred, df_rt, ncols, nrows, datasets, type="Prior|Posterior", 
                        alpha=0.2, figsize=(18,7), title_size=10, mean=True, title=""):

    fig, axs = plt.subplots(ncols = ncols, nrows=nrows, figsize=figsize)

    #df_pred = df_pred.assign(RT = lambda df: np.exp(df.RT))

    for k, ax in zip(datasets,axs.flatten()):
        print(k)
        df = df_pred.query("dataset == @k") #groupby(["dataset"])
        RT_p = df_rt.query("dataset == @k").drop(["dataset", "file", "id"], axis=1).to_numpy().flatten()
        #RT_p = np.exp(RT_p)
        sns.kdeplot(df, x="RT", hue="param_sample_id", fill=True, common_norm=False, 
                    legend=False, palette=["grey"], alpha=alpha, ax=ax
                    )

        for _, df_sample in df.groupby("param_sample_id"):
            ax.axvline(np.mean(df_sample["RT"]), color = "grey", lw=0 if not mean else 1.5)
        ax.set_label("Posterior Predicted")

        #ax.axvline(np.mean(df_sample["RT"]), color = "blue", label="Posterior Predicted")
        sns.kdeplot(RT_p, color="black", lw=3.5, ax=ax)
        ax.axvline(np.mean(RT_p), color="black", label = "DGP(Observed)", lw=0 if not mean else 3.5)
        #ax.set_title(s + " participants")
        ax.set_title(k, fontsize=title_size)
        ax.set_xlabel("Response Time (secs)", fontsize=title_size)
        ax.set_ylabel("")
    plt.suptitle(title, fontsize=title_size)
    plt.tight_layout()
    plt.legend()
    return fig, axs 