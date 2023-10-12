import scipy.sparse
from sklearn.preprocessing import QuantileTransformer
import logging

omicLogger = logging.getLogger("OmicLogger")


def standardize_data(data):
    """
    Standardize the input X using Standard Scaler
    """
    omicLogger.debug("Applying Standard scaling to given data...")

    if scipy.sparse.issparse(data):
        data = data.todense()
    else:
        data = data

    # SS = StandardScaler()
    SS = QuantileTransformer(n_quantiles=max(20, data.shape[0] // 20), output_distribution="normal")
    data = SS.fit_transform(data)
    return data, SS
