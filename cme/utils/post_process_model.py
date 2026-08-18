import pandas as pd
import glob as gl
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cme.utils.common_logging as cm_log
import cme.decision_models.confidence_accumulation as ca
import cme.decision_models.diffusion_discrete as dd
import cme.decision_models.quantum_discrete as qd
import arviz as az
import xarray as xa
import polars as pl
log = cm_log.get_logger("post_process_model")
import itertools as iter

def collect_inference_data(file_pre, part_pre, file_post, data_mod_ver):
    return_dict = {}
    for d,m_v in data_mod_ver.items():
        for m,version in m_v:
            post_smpl = []
            obs_data = []
            log_likl = []
            sample_stats = []
            filename=file_pre + d + file_post + f"_{m}_{version}_*.nc"
            log.info(f"***********Getting files from {filename}*************")
            for fl_az in gl.glob(filename):
                subfile_id = fl_az.split(str(version)+"_")[1].removesuffix(".nc")
                fl_df=part_pre + d + file_post + f"_{m}_{version}_{subfile_id}.csv"
                post_smpl.append(az.from_netcdf(fl_az))#.assign_coords({"part_id": pl.read_csv(fl_df)["0"].to_numpy()}).posterior)
                #obs_data.append(az.from_netcdf(fl_az))#.assign_coords({"part_id": pl.read_csv(fl_df)["0"].to_numpy()}).observed_data )
                #log_likl.append(az.from_netcdf(fl_az))#.assign_coords({"part_id": pl.read_csv(fl_df)["0"].to_numpy()}).log_likelihood )
                #sample_stats.append(az.from_netcdf(fl_az))#.assign_coords({"part_id": pl.read_csv(fl_df)["0"].to_numpy()}).sample_stats)
            return_dict[d+"_"+m] = xa.concat(post_smpl, dim="part_id")
                # az.InferenceData(posterior=xa.concat(post_smpl, dim="part_id"),
                #                                 observed_data=xa.concat(obs_data, dim="part_id"),
                #                                 log_likelihood=xa.concat(log_likl, dim="part_id"),
                #                                 sample_stats=xa.concat(sample_stats, dim="part_id"))
    return return_dict

def collect_dataframes(file_pre, file_post, data_mod_ver, size=None, batch_size=0, header=True, is_glob=False):
    if not is_glob: 
        df = (pd.concat([pd.read_csv(f"{file_pre}{d}{file_post}",header="infer" if header else header)
                           .assign(dataset=d,size=size)
                            for d in data_mod_ver.keys()], axis=0)
                           )
        
        return df
    else:
        df_arr = []
        for d,m_v in data_mod_ver.items():
            for m,version in m_v:#zip(model, versions)
                filename=file_pre + d + file_post + f"_{m}_{version}_*.csv"
                log.info(f"***********Getting files from {filename}*************")
                for f in gl.glob(filename): 
                    log.info(f"Current File {f}")
                    df = (pd.read_csv(f"{f}",header="infer" if header else header)
                            .assign(size=size, subfile_id = f.split(str(version)+"_")[1].removesuffix(".csv"))  
                            .assign(dataset=d)
                            .assign(model=m)
                            #.assign(dims = lambda df:df.params.str.split("[", expand=True)[1].str.removesuffix("]")) 
                            #.sort_values(["subfile_id", "part_id", "items"])
                            .assign(part_id = lambda df: df.part_id if "part_id" in df.columns else df.index)
                            .assign(id = lambda df: df.part_id.astype(int) + ((df.subfile_id.astype(int) * (batch_size)) if df.subfile_id.astype(int).max() > 0 else 0))
                            .reset_index(drop=True)
                            )
                    df_arr.append(df)

        return pd.concat(df_arr).reset_index(drop=True)

def collect_response_from_model_output(folder, data_mod_ver, batch_size=0):
    def make_dataframe(arr, indicator, folder, dataset, model, version):

        return (pd.DataFrame(np.atleast_1d(arr.squeeze()))
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
    for dataset, m_v in data_mod_ver.items():
        for model, version in m_v:

            keys_dict = {key: [] for key in keys}

            pkl_file = f"{folder}/mcmc_samples_{dataset}_{model}_{version}_*.pkl"
            pkl_files = gl.glob(pkl_file)

            for file in pkl_files: 
                with open(file, "rb") as pkl:
                        model_out = pickle.load(pkl)
                        for key in keys:
                            if key in keys_process:
                                keys_process_dict[key].append(make_dataframe(model_out[key], file, folder, dataset, model, version))
                            else:
                                keys_dict[key].append(model_out[key])
            dataset_dict[model+"_"+dataset] = keys_dict
    df_observed_rt = pd.concat(keys_process_dict["RT"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + ((df.subfile_id.astype(int) * (batch_size)) if df.subfile_id.astype(int).max() > 0 else 0)).drop(["part_id", "subfile_id"], axis=1)
    df_observed_ra = pd.concat(keys_process_dict["X"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + ((df.subfile_id.astype(int) * (batch_size)) if df.subfile_id.astype(int).max() > 0 else 0)).drop(["part_id", "subfile_id"], axis=1)      
    df_mean_init_conf = pd.concat(keys_process_dict["mean_init_conf"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + ((df.subfile_id.astype(int) * (batch_size)) if df.subfile_id.astype(int).max() > 0 else 0)).drop(["part_id", "subfile_id"], axis=1)
    df_mean_final_conf = pd.concat(keys_process_dict["mean_final_conf"]).reset_index(names="part_id").assign(id = lambda df: df.part_id + ((df.subfile_id.astype(int) * (batch_size)) if df.subfile_id.astype(int).max() > 0 else 0)).drop(["part_id", "subfile_id"], axis=1)
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

def get_conf_traj(condition, df_observed_rt, df_observed_ra, dataset_dict, delta, n_states_dict={}, response_width_dict={}):
    print("starting: ", condition)
    model_type = condition.split('_', maxsplit=1)[0]
    dataset = condition.split('_', maxsplit=1)[1]
    
    RT = df_observed_rt.query("model == @model_type and dataset == @dataset").dropna(axis=1).drop(["dataset", "model", "file", "id"], axis=1).values
    X = df_observed_ra.query("model == @model_type and dataset == @dataset").dropna(axis=1).drop(["dataset", "model", "file", "id"], axis=1).values

    n_states = n_states_dict[model_type] #51 if "Markov" in condition else 21
    response_width = response_width_dict[model_type] #5 if "Markov" in condition else 2
    post_samples = dataset_dict[condition]["post_samples"]

    drift_rate_est = post_samples[0]["mu"].mean(axis=0)
    diffusion_rate_est = post_samples[0]["sigma_final"].mean(axis=0)
    phi_0_est = post_samples[0]["phi_0"].mean(axis=0) #posterior mean

    #for t in RT.mean(axis=1):
    rt = np.tile(np.arange(delta,RT.max(),delta),(RT.shape[0],1))

    #for model_type in ["Markov", "Quantum"]:
    #    for mu, sigma, phi_0 in zip(drift_rate_est, diffusion_rate_est,phi_0_est):

    Mc, Mw, Mn = ca._get_measurement_matrix(n_states, response_width, prob=0.2, model_type = model_type)

    if model_type == "Markov":
        intensity_matrix = dd._buildK(n_states, drift_rate_est, diffusion_rate_est)
    elif model_type == "Quantum":
        intensity_matrix = qd._buildH(n_states, drift_rate_est, diffusion_rate_est)

    ideal_conf_traj_t = ca.get_mean_confidence(n_states=n_states, intensity_matrix=intensity_matrix,phi_0=phi_0_est,
                        delta= delta, Mc = Mc, Mw=Mw, Mn=Mn, t=rt,x=None, conf_scale=None,
                        model_type=model_type, transition_type="TIMESTEP", likelihood_type="SINGLE")
    #conf_traj[condition] = conf_traj_t
    #RT_mean[condition] = RT.mean(axis=1)
    resp_conf_traj_t = np.where(rt[...,None,None] > RT.mean(axis=1,keepdims=True)[...,None,None], np.nan,ideal_conf_traj_t)
    
    print("ended: ", condition)
    return condition, ideal_conf_traj_t, resp_conf_traj_t, RT.mean(axis=1)


def get_mean_confidence(n_states, response_width, measurement_prob, delta, X, RT, drift_rate_est, diffusion_rate_est, phi_0_est, conf_scale, model_type, transition_type, likelihood_type):
        intensity_matrix = ca.get_intensity_matrix(n_states, drift_rate_est, diffusion_rate_est, model_type)
        Mc, Mw, Mn = ca._get_measurement_matrix(n_states, response_width, prob=measurement_prob, model_type = model_type)

        # if model_type == "Markov":
        #     intensity_matrix = dd._buildK(n_states, drift_rate_est, diffusion_rate_est)
        # elif model_type == "Quantum":
        #     intensity_matrix = qd._buildH(n_states, drift_rate_est, diffusion_rate_est)

        
        mean_init_conf = ca.get_mean_init_confidence(n_states=n_states, phi_0 = phi_0_est, model_type=model_type)
        mean_final_conf = ca.get_mean_confidence(n_states=n_states, intensity_matrix=intensity_matrix,phi_0=phi_0_est,
                            delta= delta, Mc = Mc, Mw=Mw, Mn=Mn, t=RT,x=X, conf_scale=conf_scale,
                            model_type=model_type, transition_type=transition_type, likelihood_type=likelihood_type)
        mean_resp_conf = ca.get_mean_confidence(n_states=n_states, intensity_matrix=intensity_matrix,phi_0=phi_0_est,
                            delta= delta, Mc = Mc, Mw=Mw, Mn=Mn, t=RT,x=X, conf_scale=conf_scale,
                            model_type=model_type, transition_type=transition_type, likelihood_type=likelihood_type,
                            return_type = "ResponseConfidence"
                            )
                            
        return mean_init_conf,mean_final_conf,mean_resp_conf


def get_all_stored_data(export_version, participant_folder, est_folder, data_mod_ver):
    df_all = pd.read_csv(participant_folder, index_col=0)
    df_all = df_all.assign(item = lambda df: df[["item", "difficulty"]].groupby("difficulty", sort=False).rank(method="dense", axis=0)["item"])
    df_all = df_all.assign(difficulty = lambda df: df.difficulty.map({"far":"Easy", "close":"Hard"}))
    df_all = df_all \
    .assign(confidence_pre = lambda df: np.where(((df.ra_pre==0) & (df.confidence_pre!=50)), -df.confidence_pre, df.confidence_pre)) 
    

    df_observed_rt, _, _, _, dataset_dict = ut.collect_response_from_model_output(f"{est_folder}", data_mod_ver, 50)
    
    conditions = list(dataset_dict.keys()) #[k.split("_", maxsplit=1) for k in dataset_dict.keys()]
    df_part_id_e = pd.read_csv(f"{est_folder}/participants_id_init_far_{list(data_mod_ver.values())[0][0][0]}_{export_version}_0.csv").rename(columns={"Unnamed: 0":"part_id", "0":"id"}).assign(difficulty="Easy")
    df_part_id_h = pd.read_csv(f"{est_folder}/participants_id_init_close_{list(data_mod_ver.values())[0][0][0]}_{export_version}_0.csv").rename(columns={"Unnamed: 0":"part_id", "0":"id"}).assign(difficulty="Hard")

    df_part_id = pd.concat([df_part_id_e, df_part_id_h])
    part_n = {"far":df_part_id_e.shape[0], "close":df_part_id_h.shape[0]}

    return (df_all, df_part_id, part_n, dataset_dict, df_observed_rt, conditions)

def get_admin_model_conf(n_states, administered_conf_levels, modeled_states_per_obs, modeled_states):
    global administered_conf_levels_center
    obs_conf_levels = np.hstack([np.flip(np.repeat(-np.asarray(administered_conf_levels), modeled_states_per_obs)),
    np.array(administered_conf_levels_center),
    np.repeat(np.asarray(administered_conf_levels), modeled_states_per_obs)])

    modeled_conf_levels = list(reversed([-i for i in range(1,modeled_states+1)])) + [i for i in range(0,modeled_states+1)]
    np.asarray(modeled_conf_levels).shape

    obs_modeled_map = dict(zip(modeled_conf_levels, obs_conf_levels))

    M_c = np.zeros(n_states)
    #M_c[modeled_states+1:] = 1
    M_c[modeled_states:] = 1
    M_w = np.zeros(n_states)
    #M_w[:modeled_states] = 1
    M_w[:modeled_states+1] = 1
    
    return obs_modeled_map, M_c, M_w

def get_conf_for_each_resp(n_states, n_items,conditions, center_state, df_all, obs_modeled_map, dataset_dict, df_part_id, df_pdf, part_n, M_c, M_w):

    df_conf_resp_state = []
    for c_l in conditions:
        diff = c_l.split("_init_")[1]
        mod = c_l.split("_init_")[0]
        phi_t = np.asarray(dataset_dict[c_l]["phi_t"]).squeeze()
        #n_states = phi_t.shape[-1]
        df_t = pd.DataFrame(phi_t.reshape(-1,n_states)) \
        .assign(part_id = np.repeat(np.arange(0, part_n[diff]), n_items), 
                item_id = np.tile(np.arange(1, 31),  part_n[diff])) \
        .melt(id_vars=["part_id", "item_id"], var_name="state_id", value_name="state_prob") \
        .assign(model = mod, difficulty = "Easy" if diff == "far" else "Hard")
        if mod == "Quantum":
            df_t = df_t.assign(state_prob = lambda df:np.abs(df.state_prob)**2)
        df_conf_resp_state.append(df_t)
    df_conf_resp_state = pd.concat(df_conf_resp_state)

    df_conf_resp_state = df_conf_resp_state.merge(df_part_id, left_on=["part_id", "difficulty"], right_on = ["part_id", "difficulty"])
    df_conf_resp_state = df_conf_resp_state.merge(df_all, left_on=["id", "item_id", "difficulty"], right_on=["id", "item", "difficulty"]).drop(columns="item")
    df_conf_resp_state = df_conf_resp_state.assign(state_id = lambda df:df.state_id - center_state,
                                    state_id_scaled = lambda df:df.state_id.map(obs_modeled_map))

    conf_resp_arr = []
    for k, df in df_conf_resp_state.groupby(df_conf_resp_state.columns.drop(["state_id", "state_id_scaled", "state_prob"]).tolist()):
        df.sort_values("state_id", ascending=True)["state_id"]
        df = df.assign(conf_meas_mat = M_c if df.ra_pre.sum() > 0 else M_w) 
        # uncomment this if need confidence given choice
        df = df.assign(state_prob = df.state_prob*df.conf_meas_mat)
        # -------
        df = df.assign(state_prob = (df.state_prob)/(df.state_prob).sum())
        
        conf_resp_arr.append(df)
    df_conf_resp_state = pd.concat(conf_resp_arr)

    return df_conf_resp_state