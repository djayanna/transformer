from pathlib import Path

def get_config():
    return {
        'num_epochs': 20,
        'max_length': 512,
        'batch_size': 8,
        'learning_rate': 10**-4,
        'd_model': 512,
        'lang_source': 'en',
        'lang_target': 'es',
        'preload': None,
        'tokenizer_file': "tokenizer_{0}.json",
        'seq_len': 512,
        'experiment_name': 'runs/tmodel',
        'model_folder': 'weights',
        'model_basename': 'tmodel_',
    }

def get_weights_file_path(config, epoch: str):
    
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pth"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])