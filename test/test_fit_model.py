import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cme.decision_models import qdiffusion as qd
from cme.utils import fit_model as fm


def test_add_prior_to_arviz_data(tmp_path):
    # Previous matching posterior/prior draw counts retained for reference:
    # predictive_n, I, J = 2, 2, 3
    posterior_n, predictive_n, I, J = 3, 2, 2, 3
    # Previous sequential participant coordinates retained for reference:
    # coords = {"part_id":np.arange(I)}
    coords = {"part_id":np.array([101, 205])}
    dims = {
            "mu":["part_id"],
            "sigma_final":["part_id"],
            "RT":["part_id"]
            }

    # Previous ArViZ 0.x keyword-group construction retained for reference:
    # arviz_data = az.from_dict(posterior={"mu":np.zeros((1,2,I,1))}, coords=coords, dims=dims)
    arviz_data = az.from_dict({"posterior":{"mu":np.zeros((1,posterior_n,I,1))}}, sample_dims=["chain", "draw"], coords=coords, dims=dims) # 1 x posterior_n x I x 1 (chain x draw x participant x parameter dimension)
    prior_samples = {
                    "mu":np.arange(predictive_n*I).reshape((predictive_n,I,1)), # predictive_n x I x 1
                    "sigma_final":np.ones((predictive_n,I,1)) # predictive_n x I x 1
                    }
    prior_pd_samples = [
                        {"Samples":pd.DataFrame({"RT":np.arange(I*J) + sample_id*I*J})}
                        for sample_id in range(predictive_n)
                        ]
    RT = np.zeros((I,J)) # I x J

    arviz_data = fm._add_prior_to_arviz_data(arviz_data, prior_samples, prior_pd_samples, RT, coords, dims)

    # Previous InferenceData group checks retained for reference:
    # assert "prior" in arviz_data.groups(), "Prior samples missing from ArviZ data"
    # assert "prior_predictive" in arviz_data.groups(), "Prior predictive samples missing from ArViZ data"
    assert isinstance(arviz_data, xr.DataTree), "ArviZ 1.x should return an xarray DataTree"
    assert "prior" in arviz_data.children, "Prior samples missing from ArviZ data"
    assert "prior_predictive" in arviz_data.children, "Prior predictive samples missing from ArviZ data"
    assert arviz_data["posterior"]["mu"].shape == (1,posterior_n,I,1), "Posterior samples changed while adding prior groups"
    assert arviz_data["prior"]["mu"].shape == (1,predictive_n,I,1), "Prior parameter shape not as expected"
    assert arviz_data["prior_predictive"]["RT"].shape == (1,predictive_n,I,J), "Prior predictive shape not as expected"
    assert arviz_data["prior"]["mu"].dims == ("chain", "draw", "part_id", "mu_dim_0")
    assert arviz_data["prior_predictive"]["RT"].dims == ("chain", "draw", "part_id", "RT_dim_0")
    np.testing.assert_array_equal(arviz_data["prior"].coords["part_id"].values, coords["part_id"])
    np.testing.assert_array_equal(arviz_data["prior"]["mu"].values[0], prior_samples["mu"])
    np.testing.assert_array_equal(arviz_data["prior"]["sigma_final"].values[0], prior_samples["sigma_final"])
    np.testing.assert_array_equal(arviz_data["prior_predictive"]["RT"].values[0], np.arange(predictive_n*I*J).reshape((predictive_n,I,J)))

    export_file = tmp_path / "arviz_inferencedata.nc"
    arviz_data.to_netcdf(export_file)
    arviz_export = az.from_netcdf(export_file)

    # Previous InferenceData export checks retained for reference:
    # assert "prior" in arviz_export.groups(), "Prior samples missing from ArviZ export"
    # assert "prior_predictive" in arviz_export.groups(), "Prior predictive samples missing from ArViZ export"
    assert isinstance(arviz_export, xr.DataTree), "ArviZ 1.x export should reload as an xarray DataTree"
    assert "prior" in arviz_export.children, "Prior samples missing from ArviZ export"
    assert "prior_predictive" in arviz_export.children, "Prior predictive samples missing from ArviZ export"
    assert arviz_export["posterior"]["mu"].shape == (1,posterior_n,I,1), "Exported posterior shape not as expected"
    assert arviz_export["prior_predictive"]["RT"].shape == (1,predictive_n,I,J), "Exported prior predictive shape not as expected"
    np.testing.assert_array_equal(arviz_export["prior_predictive"]["RT"].values, arviz_data["prior_predictive"]["RT"].values)


def test_loo_post_processing_uses_loo_labels():
    rng = np.random.default_rng(0)
    arviz_data = az.from_dict(
        {
            "posterior":{"theta":rng.normal(size=(2,100,2))}, # 2 x 100 x 2 (chain x draw x parameter)
            "log_likelihood":{"obs":rng.normal(-1, 0.1, size=(2,100,3))}, # 2 x 100 x 3 (chain x draw x observation)
        },
        sample_dims=["chain", "draw"],
        dims={"theta":["parameter"], "obs":["observation"]},
    )

    summary = qd.post_process_posterior(arviz_data, method="LOO", var_name="obs")

    assert {"elpd_loo", "elpd_se", "p_loo"} <= set(summary.columns)
    assert not {"elpd_waic", "p_waic"} & set(summary.columns)
    assert np.issubdtype(summary["elpd_loo"].dtype, np.number)
    with pytest.raises(NotImplementedError, match="removed WAIC"):
        qd.post_process_posterior(arviz_data, method="WAIC", var_name="obs")
    with pytest.raises(ValueError, match="method must be"):
        qd.post_process_posterior(arviz_data, method="INVALID")
