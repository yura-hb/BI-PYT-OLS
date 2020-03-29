#!/usr/bin/env python3

"""
Napište funkci, která vyrobí v aktuální adresářy podadresáře, jejichž jména dostane jako argument.
"""

import os
from argparse import ArgumentParser
 
parser = ArgumentParser()
parser.add_argument("adresare", nargs="*", help="adresáře k vytvoření")
args = parser.parse_args()

for d in args.adresare:
    try:
        os.mkdir(d)
    except OSError:
        print('Nepodařilo se vytvořit adresář:', d)
