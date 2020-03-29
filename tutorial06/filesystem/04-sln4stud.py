#!/usr/bin/env python3

"""
4. Vypište všechny soubory a adresáře v místě zadaném uživatelem (jako řetězec na příkazové řádce).

**Hint**: V podstatě tedy musíte upravit předchozí příklad především tak, abyste správně načetli a znormalizovali cestu zadanou uživatelem. A jelikož `os.listdir()` vrací seznam souborů/adresářů lokálně pro zadanou cestu, musíte kompletní cestu získat složením získaného jména a zadané cesty, nejlépe asi pomocí metody `os.path.join()`.
"""

import os
from argparse import ArgumentParser

def print_content(cesta):
    if not os.path.exists(cesta):
        print("Zadali jste neexistující cestu, zkuste to prosím znovu.")
        exit()
    elif os.path.isfile(cesta):
        print("Zadali jste cestu k souboru, zkuste to prosím znovu.")
        exit()

    cesty = [os.path.join(cesta, x) for x in os.listdir(cesta)]

    print("Adresáře:")
    for d in [x for x in cesty if os.path.isdir(x)]:
        print('  ', d)

    print("Soubory:")
    for f in [x for x in cesty if os.path.isfile(x)]:
        print('  ', f)
        
parser = ArgumentParser()
parser.add_argument("cesta", help="cesta ke zpracování")
args = parser.parse_args()
 
cesta = os.path.normpath(args.cesta)      
print_content(cesta)