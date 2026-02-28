import pyexp
from pyexp import Config
import tqdm


@pyexp.experiment
def exp(config: Config, out): ...


def pretrain_config(dataset): ...


def finetune_config(datset): ...


@exp.configs
def configs(datasets):

    cfgs = []
    for dataset in datasets:

        ptc = pretrain_config(dataset)
        ftc = finetune_config(dataset)
        ftc["depends_on"] = [ptc]

        cfgs.append(ptc)
        cfgs.append(ftc)

    return cfgs
