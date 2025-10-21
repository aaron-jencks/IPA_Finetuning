import argparse
import logging
import pathlib
from typing import Dict, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_job_script(
        template: pathlib.Path, output: pathlib.Path,
        name: str, args: str,
        cpus: int = 16, gpus: int = 1, timeout: str = '03:00:00',
        extra_args: Optional[Dict[str, Any]] = None
):
    logger.info(f'Generating job {name} from template {template}')
    with open(template, "r") as template_fp:
        template_body = template_fp.read()

    logger.info(f'Using cli args: "{args}"')
    format_data = extra_args if extra_args is not None else {}
    format_data['job_name'] = name
    format_data['args'] = args
    format_data['cpus'] = cpus
    format_data['gpus'] = gpus
    format_data['timeout'] = timeout
    body = template_body.format(**format_data)

    with open(output, "w+") as output_fp:
        output_fp.write(body)

    logger.info(f'Output written to {output}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate slurm jobs based on config")
    ap.add_argument(
        "-t", "--template",
        type=pathlib.Path,
        default=pathlib.Path("jobs/templates/fine-tune-template.sh"),
        help="Path to the template file"
    )
    ap.add_argument("job_name", type=str, help="Name of the job")
    ap.add_argument("output", type=pathlib.Path, help="Path to the output file")
    ap.add_argument("args", type=str, help="arg string to be passed to main script")
    ap.add_argument('--timeout', type=str, default='03:00:00', help='the time limit for the job')
    ap.add_argument('--cpus', type=int, default=16, help='the number of cpus to request')
    ap.add_argument('--gpus', type=int, default=1, help='the number of gpus to request')
    args = ap.parse_args()

    generate_job_script(
        args.template, args.output,
        args.job_name, args.args,
        args.cpus, args.gpus, args.timeout
    )