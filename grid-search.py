from abc import ABC, abstractmethod
import argparse
import json
import logging
import pathlib
import subprocess
from typing import List, Dict, Any, Tuple
import uuid

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
        job_name: str, train_lang: str, eval_lang: str, model_type: str,
        configs: List[pathlib.Path], hyperparameters: Dict[str, Any],
        template: pathlib.Path,
        output_dir: pathlib.Path, config_dir: pathlib.Path,
        job_timeout: str = '01:00:00',
):
    unique_id = str(uuid.uuid4())
    logger.info(f'Generating grid job {unique_id} with parameters {hyperparameters}')

    hyperparameters['uuid'] = unique_id

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


def load_grid_ranges(grid_config: pathlib.Path) -> List[Value]:
    logger.info(f'Loading grid ranges from {grid_config}')
    with open(grid_config) as fp:
        r_config = json.load(fp)

    ranges = []
    for k in r_config.keys():
        entity = r_config[k]
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
    ranges = load_grid_ranges(grid_config)

    languages = list(cfg["datasets"].keys())
    if len(languages) != 2:
        raise ValueError(f'expected exactly 2 languages, but found {len(languages)}')
    train_lang = languages[0]
    val_lang = languages[1]

    job_name = f'{cfg["wandb"]["project"]}-gridsearch-{train_lang}-{val_lang}'

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
