from .defender import Defender
from .strip_defender import STRIPDefender
from .rap_defender import RAPDefender
from .onion_defender import ONIONDefender
from .bki_defender import BKIDefender
from .cube_defender import CUBEDefender
from .react_defender import REACTDefender
# from .fabe_defender import FABEDefender
from .fabe_beam_defender import FABEDefender
from .badacts_defender import BadActs_Defender

DEFENDERS = {
    "base": Defender,
    'strip': STRIPDefender,
    'rap': RAPDefender,
    'onion': ONIONDefender,
    'bki':  BKIDefender,
    'cube': CUBEDefender,
    'react': REACTDefender,
    'fabe': FABEDefender,
    'badacts': BadActs_Defender
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
