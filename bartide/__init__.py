from .paired_extractor import BarcodeExtractor
from .sequence_corrector import SeqCorrect
from .batch_analyzer import BarcodeAnalyzer
from .utils import *
from importlib_metadata import version

try:
    __version__ = version("bartide")
except ImportError:
    print("Bartide is not installed", flush=True)
