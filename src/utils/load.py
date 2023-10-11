import json
from models.custom_model import CustomModel


import joblib


def load_model(model_name, model_path):
    """
    Load a previously saved and trained model. Uses joblib's version of pickle.
    """
    print("Model path: ")
    print(model_path)
    print("Model ")
    print()

    if model_name in CustomModel.custom_aliases:
        # Remove .pkl here, it will be handled later
        model_path = model_path.replace(".pkl", "")

        try:
            model = CustomModel.custom_aliases[model_name].load_model(model_path)
        except Exception as e:
            print("The trained model " + model_name + " is not present")
            raise e
    else:
        # Load a previously saved model (using joblib's pickle)
        with open(model_path, "rb") as f:
            model = joblib.load(f)
    return model


def load_config(config_path):
    """
    Load a JSON file (general function, but we use it for configs)
    """
    with open(config_path) as json_file:
        config_dict = json.load(json_file)
    return config_dict
