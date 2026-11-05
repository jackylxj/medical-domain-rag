"""Run cloud LLaMA-Factory, preserving configuration and provenance."""
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from scripts.validate_data import sha256, validate

ROOT = Path(__file__).resolve().parents[1]


def capture(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, errors='replace')
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    except OSError as exc:
        return {'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/train_smoke.yaml')
    parser.add_argument('--factory-dir', type=Path, default=Path('/mnt/workspace/LLaMA-Factory'))
    parser.add_argument('--dry-run', action='store_true', help='Resolve config only; no model/GPU loading or training')
    args = parser.parse_args()
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    for key in ('dataset_dir', 'output_dir', 'model_name_or_path'):
        if sys.platform == 'win32' and PurePosixPath(config[key]).is_absolute():
            # Keep Linux cloud paths intact when previewing on Windows.
            continue
        value = Path(config[key])
        config[key] = str(value.resolve() if value.is_absolute() else (ROOT / value).resolve())
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        print('DRY RUN: no files written, no model loaded, no training launched.')
        return
    executable = shutil.which('llamafactory-cli')
    if not executable:
        raise RuntimeError('llamafactory-cli not found; activate the cloud training environment')
    model = Path(config['model_name_or_path'])
    if not (model / 'config.json').is_file():
        raise FileNotFoundError(f'Model config not found: {model}')
    data_dir = Path(config['dataset_dir'])
    info = json.loads((data_dir / 'dataset_info.json').read_text(encoding='utf-8'))
    data = data_dir / info[config['dataset']]['file_name']
    data_summary = validate(data)
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    expected = manifest.get('sample_sha256') or manifest['files'][data.name]['sha256']
    if data_summary['sha256'] != expected:
        raise ValueError('Sample hash differs from manifest; regenerate in a new directory')
    if config.get('eval_dataset'):
        eval_data = data_dir / info[config['eval_dataset']]['file_name']
        eval_summary = validate(eval_data)
        if eval_summary['sha256'] != manifest['files'][eval_data.name]['sha256']:
            raise ValueError('Validation data hash differs from manifest')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
    if config.get('bf16') and not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 is unavailable')
    output = Path(config['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output is nonempty; use a new output_dir: {output}')
    output.mkdir(parents=True, exist_ok=True)
    resolved = output / 'run_config.yaml'
    resolved.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    shutil.copy2(data_dir / 'manifest.json', output / 'data_manifest.json')
    environment = {'started_utc': datetime.now(timezone.utc).isoformat(),
                   'python': sys.version, 'torch': torch.__version__,
                   'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
                   'cli': executable, 'dataset': data_summary,
                   'config_sha256': sha256(resolved),
                   'factory_commit': capture(['git', '-C', str(args.factory_dir), 'rev-parse', 'HEAD']),
                   'project_commit': capture(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                   'factory_version': capture([executable, 'version']),
                   'nvidia_smi': capture(['nvidia-smi']),
                   'model_config_sha256': sha256(model / 'config.json')}
    (output / 'environment.json').write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding='utf-8')
    frozen = capture([sys.executable, '-m', 'pip', 'freeze'])
    (output / 'pip-freeze.txt').write_text(frozen.get('stdout', str(frozen)), encoding='utf-8')
    command = [executable, 'train', str(resolved)]
    with (output / 'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            code = 130
    result = {'returncode': code, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'adapter_config_exists': (output / 'adapter_config.json').is_file(),
              'adapter_weights_exist': (output / 'adapter_model.safetensors').is_file()}
    (output / 'run_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if code:
        raise SystemExit(code)
    if not all((result['adapter_config_exists'], result['adapter_weights_exist'])):
        raise RuntimeError('Training exited successfully but expected adapter files are missing')
    print('Training exited successfully and adapter files exist. Reload/inference still needs verification.')


if __name__ == '__main__':
    main()
