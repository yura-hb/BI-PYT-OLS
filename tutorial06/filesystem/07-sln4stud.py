#!/usr/bin/env python3

"""
Pokuste se smazat v aktuálním adresáři podadresáře, jejichž jména zadá uživatel jako parametry skriptu.
"""

import os
from argparse import ArgumentParser
 
parser = ArgumentParser()
parser.add_argument("adresare", nargs="*", help="adresáře ke smazání")
args = parser.parse_args()

for d in args.adresare:
    try:
        os.rmdir(d)
    except OSError:
        print('Nepodařilo se smazat adresář:', d)
