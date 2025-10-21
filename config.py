import json
import pathlib
from typing import List

from cascade_config import CascadeConfig


def load_config(
        configs: List[pathlib.Path],
        default_config: pathlib.Path = pathlib.Path('config/default.json'),
        dataset_database: pathlib.Path = pathlib.Path('config/language-database.json'),
):
    """
    Parses cascading json config files
    :param configs: the cascading json config files
    :param default_config: the default config file (config/default.json)
    :param dataset_database: the dataset database (config/language-database.json)
    :return: tuple containing the parsed config and the database
    """
    config = CascadeConfig()
    config.add_json(default_config)
    for config_path in configs:
        config.add_json(config_path)
    result = config.parse()
    with open(dataset_database, 'r') as fp:
        dataset_database = json.load(fp)
    return result, dataset_database
