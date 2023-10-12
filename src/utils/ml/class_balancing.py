import imblearn
import logging

omicLogger = logging.getLogger("OmicLogger")


def oversample_data(x_train, y_train, seed):
    """
    Given the training set it has a class imbalance problem, this will over sample the training data to balance out the
    classes
    """
    omicLogger.debug("Oversampling data...")
    # define oversampling strategy
    oversample = imblearn.over_sampling.RandomOverSampler(random_state=seed, sampling_strategy="not majority")
    # fit and apply the transform
    x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)
    print(f"X train data after oversampling shape: {x_train.shape}")
    print(f"y train data after oversampling shape: {y_train.shape}")

    return x_resampled, y_resampled, oversample.sample_indices_


def undersample_data(x_train, y_train, seed):
    """
    Given the training set it has a class imbalance problem, this will over sample the training data to balance out the
    classes
    """
    omicLogger.debug("Undersampling data...")
    # define undersampling strategy
    oversample = imblearn.under_sampling.RandomUnderSampler(random_state=seed, sampling_strategy="not minority")
    # fit and apply the transform
    x_resampled, y_resampled = oversample.fit_resample(x_train, y_train)
    print(f"X train data after undersampling shape: {x_train.shape}")
    print(f"y train data after undersampling shape: {y_train.shape}")

    return x_resampled, y_resampled, oversample.sample_indices_
