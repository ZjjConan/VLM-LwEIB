from .caltech101 import CALTECH101_TEMPLATES
from .oxford_pets import OXFORD_PETS_TEMPLATES
from .stanford_cars import STANFORD_CARS_TEMPLATES
from .oxford_flowers import OXFORD_FLOWERS_TEMPLATES
from .food101 import FOOD101_TEMPLATES
from .fgvc_aircraft import FGVC_AIRCRAFT_TEMPLATES
from .sun397 import SUN397_TEMPLATES
from .dtd import DTD_TEMPLATES
from .eurosat import EUROSAT_TEMPLATES
from .ucf101 import UCF101_TEMPLATES
from .imagenet import IMAGENET_TEMPLATES

def load_CuPL_templates(dataset_name):
    dname = dataset_name.lower()
    if dname == "caltech101":
        return CALTECH101_TEMPLATES
    elif dname == "oxfordpets":
        return OXFORD_PETS_TEMPLATES
    elif dname == "stanfordcars":
        return STANFORD_CARS_TEMPLATES
    elif dname == "oxfordflowers":
        return OXFORD_FLOWERS_TEMPLATES
    elif dname == "food101":
        return FOOD101_TEMPLATES
    elif dname == "fgvcaircraft":
        return FGVC_AIRCRAFT_TEMPLATES
    elif dname == "describabletextures":
        return DTD_TEMPLATES
    elif dname == "eurosat":
        return EUROSAT_TEMPLATES
    elif dname == "sun397":
        return SUN397_TEMPLATES
    elif dname == "ucf101":
        return UCF101_TEMPLATES
    elif "imagenet" in dname:
        return IMAGENET_TEMPLATES