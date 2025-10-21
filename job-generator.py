import argparse
import logging
import pathlib
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_job_script(template: pathlib.Path, output: pathlib.Path, name: str, config: List[str]):
    logger.info(f'Generating job {name} from template {template}')
    with open(template, "r") as template_fp:
        template_body = template_fp.read()

    logger.info(f'Using configs: {config}')
    body = template_body.format(
        job_name=name,
        config_names=' '.join(config),
    )

    with open(output, "w+") as output_fp:
        output_fp.write(body)

    logger.info(f'Output written to {output}')


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate slurm jobs based on config")
    ap.add_argument(
        "-t", "--template",
        type=pathlib.Path,
        default=pathlib.Path("./jobs/fine-tune-template.sh"),
        help="Path to the template file"
    )
    ap.add_argument("job_name", type=str, help="Name of the job")
    ap.add_argument("output", type=pathlib.Path, help="Path to the output file")
    ap.add_argument("configs", type=pathlib.Path, nargs='+', help="Paths to the config files")
    args = ap.parse_args()

    generate_job_script(args.template, args.output, args.job_name, list(map(str, args.configs)))