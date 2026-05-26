import json
import os

FAILURE_METADATA = {
    'high_lr':  {'failure': 'GRADIENT_EXPLOSION',   'cause': 'LR too high (lr=1.0)',      'fix': 'lower LR or clip gradients'},
    'low_lr':   {'failure': 'STALLED_TRAINING',     'cause': 'LR too low (lr=1e-7)',      'fix': 'increase learning rate'},
    'bad_init': {'failure': 'ACTIVATION_EXPLOSION', 'cause': 'init std too large (1.0)',  'fix': 'use std=0.02 scaled init'},
    'no_clip':  {'failure': 'GRADIENT_INSTABILITY', 'cause': 'no gradient clipping',      'fix': 'clip gradients at 1.0'},
    'baseline': {'failure': 'NONE',                 'cause': 'N/A',                       'fix': 'N/A'},
}


def generate_postmortem(experiment_name, logs_dir='logs'):
    exp_dir = os.path.join(logs_dir, experiment_name)
    warnings_path = os.path.join(exp_dir, 'warnings.jsonl')
    metrics_path = os.path.join(exp_dir, 'metrics.jsonl')

    meta = FAILURE_METADATA.get(experiment_name, {'failure': 'UNKNOWN', 'cause': 'unknown', 'fix': 'unknown'})

    first_signal = None
    if os.path.exists(warnings_path):
        with open(warnings_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                w = json.loads(line)
                first_signal = f"{w.get('monitor', w.get('message', 'unknown'))} at step {w.get('step', '?')}"
                break

    if first_signal is None and os.path.exists(metrics_path):
        first_signal = 'no warning signals recorded'

    print(f"Failure: {meta['failure']}")
    if first_signal:
        print(f"First signal: {first_signal}")
    print(f"Cause: {meta['cause']}")
    print(f"Fix: {meta['fix']}")


if __name__ == '__main__':
    logs_dir = 'logs'
    if os.path.isdir(logs_dir):
        for name in sorted(os.listdir(logs_dir)):
            if os.path.isdir(os.path.join(logs_dir, name)):
                print(f"\n=== {name} ===")
                generate_postmortem(name, logs_dir=logs_dir)
