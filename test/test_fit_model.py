import arviz as az
import numpy as np
import pandas as pd

from cme.utils import fit_model as fm


def test_add_prior_to_arviz_data(tmp_path):
    predictive_n, I, J = 2, 2, 3
    coords = {"part_id":np.arange(I)}
    dims = {
            "mu":["part_id"],
            "sigma_final":["part_id"],
            "RT":["part_id"]
            }

    arviz_data = az.from_dict(posterior={"mu":np.zeros((1,2,I,1))}, coords=coords, dims=dims)
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

    assert "prior" in arviz_data.groups(), "Prior samples missing from ArviZ data"
    assert "prior_predictive" in arviz_data.groups(), "Prior predictive samples missing from ArviZ data"
    assert arviz_data.prior["mu"].shape == (1,predictive_n,I,1), "Prior parameter shape not as expected"
    assert arviz_data.prior_predictive["RT"].shape == (1,predictive_n,I,J), "Prior predictive shape not as expected"

    export_file = tmp_path / "arviz_inferencedata.nc"
    arviz_data.to_netcdf(export_file)
    arviz_export = az.from_netcdf(export_file)

    assert "prior" in arviz_export.groups(), "Prior samples missing from ArviZ export"
    assert "prior_predictive" in arviz_export.groups(), "Prior predictive samples missing from ArviZ export"
    assert arviz_export.prior_predictive["RT"].shape == (1,predictive_n,I,J), "Exported prior predictive shape not as expected"
