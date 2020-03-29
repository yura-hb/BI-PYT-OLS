#!/usr/bin/env python3

""" Napište program, který vypíše řádku po řádce soubor, cestu k němuž bude brát jako svůj vstupní argument. Přitom na začátek každé řádky přidá její pořadové číslo v souboru plus počet slov a počet znaků v příslušné řádce. """

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("soubor", help="soubor ke zpracování")
args = parser.parse_args()

with open(args.soubor, mode='r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        slov  = len( line.split() )
        znaku = len( line )
        print(i, slov, znaku, line, end='')
