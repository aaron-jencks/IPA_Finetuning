import argparse
import logging
import pathlib
import subprocess
import uuid
from typing import Union, List

from config import load_config
from job_generator import generate_job_script

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Generates and submits jobs for a given config')
    ap.add_argument('config', type=pathlib.Path, nargs='+', help='paths to config files')
    ap.add_argument(
        '--template', type=pathlib.Path,
        default=pathlib.Path('jobs/templates/fine-tune-template.sh'),
        help='template directory'
    )
    ap.add_argument(
        '--output-dir', type=pathlib.Path,
        default=pathlib.Path('./jobs'),
        help="Path to the output directory for the generated jobs"
    )
    ap.add_argument('--timeout', type=str, default='01:00:00', help='Timeout in for the slurm jobs')
    return ap.parse_args()


def generate_finetune_job(
        job_name: str, train_lang: Union[str, List[str]], eval_lang: Union[str, List[str]], model_type: str,
        configs: List[pathlib.Path],
        template: pathlib.Path,
        output_dir: pathlib.Path,
        job_timeout: str = '01:00:00',
):
    unique_id = str(uuid.uuid4())
    logger.info(f'Generating finetune job {unique_id}')

    generated_job_name = f'{unique_id}-{job_name}'
    job_script_file = output_dir / f'{generated_job_name}.sh'
    logger.info(f'Writing finetune job script to {job_script_file}')
    config_string = ' '.join(map(str, configs))
    if isinstance(train_lang, list):
        train_lang = ' '.join(train_lang)
    if isinstance(eval_lang, list):
        eval_lang = ' '.join(eval_lang)
    generated_args_string = f'{config_string} --train-langs {train_lang} --eval-langs {eval_lang} --model-type {model_type}'
    generate_job_script(
        template, job_script_file,
        generated_job_name, generated_args_string,
        timeout=job_timeout,
    )

    logger.info(f'Queueing job {generated_job_name}')
    subprocess.run(['sbatch', job_script_file], check=True)


def main():
    args = parse_args()
    cfg, _ = load_config(args.config)
    languages = list(cfg['datasets'].keys())

    for model_type in ['normal', 'ipa']:
        job_name = f'{cfg["wandb"]["project"]}-gridsearch-all'
        generate_finetune_job(
            job_name,
            languages, languages, model_type,
            args.config,
            args.template, args.output_dir, args.timeout
        )
        for lang in languages:
            job_name = f'{cfg["wandb"]["project"]}-gridsearch-{lang}'
            generate_finetune_job(
                job_name,
                lang, languages, model_type,
                args.config,
                args.template, args.output_dir, args.timeout
            )


if __name__ == "__main__":
    main()