from nni.experiment import Experiment
import signal

search_space = {
    "hidden_size": {"_type": "quniform", "_value": [128, 1024, 128]},
    "embd_size": {"_type": "quniform", "_value": [128, 1024, 128]},
    # "num_layers": {"_type": "quniform", "_value": [1, 10, 1]},
    "batch_size": {"_type": "choice", "_value": [32, 64, 128]},
    "learning_rate": {"_type": "loguniform", "_value": [0.0001, 1]},
    "epochs": {"_type": "choice", "_value": [200]},
    "max_length": {"_type": "choice", "_value": [24]},
}

# Configure experiment
experiment = Experiment("local")
experiment.config.trial_command = "python go.py"
experiment.config.trial_code_directory = "."
experiment.config.search_space = search_space
experiment.config.max_trial_number = 500
experiment.config.trial_concurrency = 1
experiment.config.tuner.name = "Anneal"  # Anneal, Evolution, TPE
experiment.config.tuner.class_args["optimize_mode"] = "minimize"
# experiment.config.tuner.class_args["population_size"] = 16

# Run it!
experiment.run(port=8848)

print("Experiment is running. Press Ctrl-C to quit.")
signal.pause()
