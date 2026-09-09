import os
import sys
from .vitwarp_v8 import ViTWarpV8

def fetch_model(args):
    if args.algorithm == 'vitwarp':
        model = ViTWarpV8(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model