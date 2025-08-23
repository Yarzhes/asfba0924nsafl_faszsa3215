import yaml

def merge_walk_forward_configs():
    """
    Merges settings.yaml and walk_forward_config.yaml into a temporary
    config file for walk-forward analysis.
    """
    with open('settings.yaml', 'r') as f:
        settings = yaml.safe_load(f)

    with open('ultra_signals/backtest/walk_forward_config.yaml', 'r') as f:
        wf_config = yaml.safe_load(f)

    # Add the walkforward section from the walk_forward_config
    settings['walkforward'] = wf_config.get('walk_forward_config')
    # Update symbols
    if 'symbols' in wf_config:
        settings['runtime']['symbols'] = wf_config['symbols']

    with open('temp_wf_config.yaml', 'w') as f:
        yaml.dump(settings, f)

if __name__ == '__main__':
    merge_walk_forward_configs()