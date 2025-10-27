from abc import ABC, abstractmethod
import argparse
import json
import logging
import pathlib
import subprocess
from typing import List, Dict, Any, Tuple, Union
import uuid
import time
import os

from config import load_config
from job_generator import generate_job_script

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Value(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def next(self) -> Tuple[float, bool]:
        pass

    @abstractmethod
    def get_current(self) -> float:
        pass


class StaticValue(Value):
    def __init__(self, name: str, value: float):
        super().__init__(name)
        self.value = value

    def next(self) -> Tuple[float, bool]:
        return self.value, True

    def get_current(self) -> float:
        return self.value


class LooperValue(Value):
    def __init__(self, name: str, values: List[float]):
        super().__init__(name)
        self.values = values
        self.current = 0

    def reset(self):
        self.current = 0

    def next(self) -> Tuple[float, bool]:
        v = self.values[self.current]
        r = False
        self.current += 1
        if self.current >= len(self.values):
            self.current = 0
            r = True
        return v, r

    def get_current(self) -> float:
        return self.values[self.current]


def generate_grid_job(
        job_name: str, train_lang: Union[str, List[str]], eval_lang: Union[str, List[str]], model_type: str,
        configs: List[pathlib.Path], hyperparameters: Dict[str, Any],
        template: pathlib.Path,
        output_dir: pathlib.Path, config_dir: pathlib.Path,
        job_timeout: str = '01:00:00',
):
    unique_id = str(uuid.uuid4())
    logger.info(f'Generating grid job {unique_id} with parameters {hyperparameters}')

    hyperparameters['uuid'] = unique_id
    hyperparameters['result_file'] = str(output_dir.parent / 'results' / f'{unique_id}.json')

    hyperparameter_config_file = config_dir / f'{unique_id}.json'
    logger.info(f'Writing grid job parameters to {hyperparameter_config_file}')
    with open(hyperparameter_config_file, 'w+') as fp:
        body = {
            "hyperparameters": hyperparameters,
        }
        json.dump(body, fp)

    generated_job_name = f'{unique_id}-{job_name}'
    job_script_file = output_dir / f'{generated_job_name}.sh'
    logger.info(f'Writing grid job script to {job_script_file}')
    config_string = ' '.join(map(str, configs))
    if isinstance(train_lang, list):
        train_lang = ' '.join(train_lang)
    if isinstance(eval_lang, list):
        eval_lang = ' '.join(eval_lang)
    generated_args_string = f'{config_string} {hyperparameter_config_file} --train-langs {train_lang} --eval-langs {eval_lang} --model-type {model_type}'
    generate_job_script(
        template, job_script_file,
        generated_job_name, generated_args_string,
        timeout=job_timeout,
        extra_args={
            'temp_config_name': hyperparameter_config_file,
        }
    )

    logger.info(f'Queueing job {generated_job_name}')
    subprocess.run(['sbatch', job_script_file], check=True)


def collect_results(config_dir: pathlib.Path, results_dir: pathlib.Path) -> Dict[str, Any]:
    """Collect results from all completed jobs and return best configuration."""
    results = []
    
    for result_file in results_dir.glob('*.json'):
        try:
            with open(result_file, 'r') as fp:
                data = json.load(fp)
                if 'final_metric' in data and 'uuid' in data:
                    results.append(data)
        except:
            continue
    
    if not results:
        logger.warning('No results found')
        return {}
    
    # Sort by metric (assumes higher is better; change to reverse=False if lower is better)
    best = max(results, key=lambda x: x['final_metric'])
    
    logger.info(f'Best result: UUID={best["uuid"]}, Metric={best["final_metric"]}')
    logger.info(f'Best hyperparameters: {best.get("hyperparameters", {})}')
    
    return best


def monitor_jobs(config_dir: pathlib.Path, results_dir: pathlib.Path, 
                 check_interval: int = 300, early_stop_threshold: float = 0.1):
    """Monitor running jobs and cancel those performing poorly."""
    logger.info('Starting job monitoring for early stopping')
    
    checked_jobs = set()
    
    while True:
        time.sleep(check_interval)
        
        # Check for intermediate results
        for result_file in results_dir.glob('*_intermediate.json'):
            uuid_str = result_file.stem.replace('_intermediate', '')
            
            if uuid_str in checked_jobs:
                continue
                
            try:
                with open(result_file, 'r') as fp:
                    data = json.load(fp)
                    
                # Cancel if metric is below threshold after some epochs
                if data.get('epoch', 0) >= 2 and data.get('metric', 1.0) < early_stop_threshold:
                    job_name = f'{uuid_str}-*'
                    logger.info(f'Cancelling poor job {uuid_str} (metric={data.get("metric")})')
                    subprocess.run(['scancel', '--name', job_name], check=False)
                    checked_jobs.add(uuid_str)
            except:
                continue
        
        # Break if all jobs done (optional - you can remove this to run indefinitely)
        running = subprocess.run(['squeue', '-u', os.environ.get('USER', '')], 
                                capture_output=True, text=True)
        if 'gridsearch' not in running.stdout:
            break
    
    logger.info('Job monitoring complete')


def load_grid_ranges(grid_config: dict) -> List[Value]:
    ranges = []
    for k in grid_config.keys():
        entity = grid_config[k]
        if isinstance(entity, list):
            logger.info(f'Loading {k} as a loop of values: {entity}')
            ranges.append(LooperValue(
                k,
                entity,
            ))
        else:
            logger.info(f'Loading {k} as a static value: {entity}')
            ranges.append(StaticValue(k, entity))

    return ranges


def perform_ripple(ranges: List[Value]) -> Tuple[Dict[str, Any], bool]:
    result = {}
    resets = []
    needs_next = True
    for v in ranges:
        reset = False
        if needs_next:
            value, reset = v.next()
        else:
            value = v.get_current()
        result[v.name] = value
        resets.append(reset)
        needs_next = reset
    return result, all(resets)


def grid_search_loop(
        cfg: dict,
        configs: List[pathlib.Path], grid_config: pathlib.Path,
        template: pathlib.Path,
        output_dir: pathlib.Path, config_dir: pathlib.Path,
        job_timeout: str = '01:00:00',
):
    logger.info(f'Loading grid ranges from {grid_config}')
    with open(grid_config, 'r') as fp:
        r_config = json.load(fp)

    ranges = load_grid_ranges(r_config['parameters'])

    train_lang = r_config['languages']['train']
    val_lang = r_config['languages']['eval']
    if train_lang == 'all' or val_lang == 'all':
        languages = list(cfg['datasets'].keys())
        if train_lang == 'all':
            train_lang = languages
        if val_lang == 'all':
            val_lang = languages

    train_lang_str = '-'.join(train_lang) if isinstance(train_lang, list) else train_lang
    val_lang_str = '-'.join(val_lang) if isinstance(val_lang, list) else val_lang
    job_name = f'{cfg["wandb"]["project"]}-gridsearch-{train_lang_str}-{val_lang_str}'

    while True:
        current_config, finished = perform_ripple(ranges)

        for mt in args.model_type:
            generate_grid_job(
                job_name, train_lang, val_lang, mt,
                configs, current_config,
                template, output_dir, config_dir,
                job_timeout,
            )

        if finished:
            break

    logger.info(f'Finished generating grid search for {job_name}')
    
    # Create results directory
    results_dir = output_dir.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Monitor jobs and collect results
    monitor_jobs(config_dir, results_dir, check_interval=300, early_stop_threshold=0.5)
    best_config = collect_results(config_dir, results_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('config', type=pathlib.Path, nargs='+', help='paths to config files')
    ap.add_argument(
        '--grid-config', type=pathlib.Path,
        default=pathlib.Path('config/grid-search.json'),
        help='Path to the configuration file for the grid search ranges'
    )
    ap.add_argument(
        "-t", "--template",
        type=pathlib.Path,
        default=pathlib.Path("jobs/templates/grid-search-fine-tune-template.sh"),
        help="Path to the template file"
    )
    ap.add_argument(
        '--output-dir', type=pathlib.Path,
        default=pathlib.Path('./jobs'),
        help="Path to the output directory for the generated jobs"
    )
    ap.add_argument(
        '--temp-config-dir', type=pathlib.Path,
        default=pathlib.Path('config'),
        help="Path to the temporary configuration directory for generate config files"
    )
    ap.add_argument('--model-type', type=str, nargs='+', default=['normal', 'ipa'], help='Type of model to use')
    ap.add_argument('--timeout', type=str, default='01:00:00', help='Timeout in for the slurm jobs')
    args = ap.parse_args()
    cfg, _ = load_config(args.config)

    grid_search_loop(cfg, args.config, args.grid_config, args.template, args.output_dir, args.temp_config_dir, args.timeout)