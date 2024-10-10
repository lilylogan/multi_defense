from .trainer import Trainer
from .ep_trainer import EPTrainer
from .sos_trainer import SOSTrainer
from .lm_trainer import LMTrainer
from .neuba_trainer import NeuBATrainer
from .por_trainer import PORTrainer
from .lwp_trainer import LWPTrainer
from .lws_trainer import LWSTrainer
from .ripples_trainer import RIPPLESTrainer
from .visual_trainer import VisualTrainer

TRAINERS = {
    "base": Trainer,
    "ep": EPTrainer,
    "sos": SOSTrainer,
    "lm": LMTrainer,
    "neuba": NeuBATrainer,
    "por": PORTrainer,
    'lwp': LWPTrainer,
    'lws': LWSTrainer,
    'ripples': RIPPLESTrainer,
    'visual': VisualTrainer
}



def load_trainer(config):
    print("Loaded trainer ", config["name"])
    return TRAINERS[config["name"].lower()](**config)
