# Finetuning in the IPA GPT Project

Alright, so lets get right into it.

To finetune the model that you just pretrained you'll want to use this library, it's responsible for, well, fine-tuning.
It makes a few assumptions:

## Assumptions

The config files are structured in such a way that it assumes the following:
1. Your model checkpoints are called `ckpt.pt` and exist within a folder structure as follows (this is the default structure that nanogpt uses: `checkpoint_prefix/model_prefix/ckpt.pt`)
2. Your model aligns directly with the nanogpt model class, this can be found in [model.py](./model.py) and [hf_wrapper.py](./hf_wrapper.py).
3. The finetuning you're trying to do is classification, seq2seq is not yet supported.

## Config files

Recently everything has been moved over to this system of using cascading json config files.
The way this works is that most of the programs accept a `config` argument, this contains a list of config files to use.
Each later config file overwrites any settings in the previous ones. There is a default.json that is included regardless.
This means, for example, that if you wanted to run the finetune-exp file, you'd call with something like this:
```
python finetune-exp.py config/finetune-rus-pol-sentiment.json ...
```
For the most part [default.json](./config/default.json) defines directory locations, while child configs, like [finetune-rus-pol-sentiment.json](./config/finetune-rus-pol-sentiment.json)
define hyperparameters, model names, tokenizer names, datasets, etc...

### Language Database

There exists another file, the [language-database.json](./config/language-database.json). It defines languages, and their datasets, as well as the tasks associated with them.
The general syntax for an entry is:
```json
"language name": {
    "task name": {
      "dataset name": {
        "dataset": "dataset path",
        "train_features": [
          "one or more features for training"
        ],
        "eval_feature": "the label feature",
        "splits": [
          "names of the splits used for train",
          "and evaluation"
        ],
        "task_type": "classification",  // This is not optional
        "classes": 3  // the number of classes for this task
      }
    }
  }
```

### Grid Search Config

The last type of config, so far... Is the grid search config, it defines the ranges of values to search for during grid search.
```json
{
  // parameters to loop over
  "parameters" : {
    "learning_rate": [1e-3, 5e-4, 1e-4],
    "batch_size": [16, 32, 64],
    "warmup_ratio": [0.0, 0.1, 0.2],
    "epochs": 3  // can also be a single static value
  },
  // defines which languages to train on, and which to evaluate
  // defining one or the other can help with efficiency
  "languages": {
    "train": "all", // can either be all, or the name of the language to use
    "eval": "all"
  }
}
```

## Job Templates

There are scripts to generate and submit jobs for you, they use templates. 
The template is loaded from file at runtime, and format is called on it to populate the fields.
There are a few required fields for finetuning templates:
* `job_name`: job name for the job
* `cpus`: cpus for the job
* `gpus`: gpus for the job
* `timeout`: timeout for the job
* `args`: the arguments to be passed to the script

You can also have additional parameters, you can see [grid-search-fine-tune-template.sh](jobs/templates/grid-search-fine-tune-template.sh) for en example of this.
You define the fields, just like you would with format `{field_name}`.

# API

Okay, now that, that's out of the way, lets start with grid-search.

## Grid-Search

```
usage: grid-search.py [-h] [--grid-config GRID_CONFIG] [-t TEMPLATE] [--output-dir OUTPUT_DIR] [--temp-config-dir TEMP_CONFIG_DIR] [--model-type MODEL_TYPE [MODEL_TYPE ...]] [--timeout TIMEOUT] config [config ...]

positional arguments:
  config                paths to config files

options:
  -h, --help            show this help message and exit
  --grid-config GRID_CONFIG
                        Path to the configuration file for the grid search ranges
  -t TEMPLATE, --template TEMPLATE
                        Path to the template file
  --output-dir OUTPUT_DIR
                        Path to the output directory for the generated jobs
  --temp-config-dir TEMP_CONFIG_DIR
                        Path to the temporary configuration directory for generate config files
  --model-type MODEL_TYPE [MODEL_TYPE ...]
                        Type of model to use
  --timeout TIMEOUT     Timeout in for the slurm jobs
```

**Note**: Model Type is either `normal` or `ipa` or a combination of the two, the rest should be self-explanatory.

The grid-search program is in [grid-search.py](./grid-search.py). 
It reads and loops through the grid configuration file provided and generates jobs for every permutation of hyperparameters. **BE CAREFUL THERE CAN BE A LOT**

You don't need to make a slurm job for this, it will create it's own slurm jobs for you, just make sure to run it in an interactive session or it will hog the cpu.
But it only takes a few seconds to run.

## Fine-Tune

There's another program that will launch all of your finetuning jobs. It's in [finetuning-generator.py](./finetuning-generator.py).
It reads languages from the config file supplied, then loops through the following configurations: (assume 2 languages, A and B)
* A+B -> A
* A+B -> B
* A -> A
* A -> B
* B -> A
* B -> B

And it does this for both normal and ipa model types.

```
usage: finetuning-generator.py [-h] [--template TEMPLATE] [--output-dir OUTPUT_DIR] [--timeout TIMEOUT] config [config ...]

Generates and submits jobs for a given config

positional arguments:
  config                paths to config files

options:
  -h, --help            show this help message and exit
  --template TEMPLATE   template directory
  --output-dir OUTPUT_DIR
                        Path to the output directory for the generated jobs
  --timeout TIMEOUT     Timeout in for the slurm jobs
```

This one's a bit simpler than the other.