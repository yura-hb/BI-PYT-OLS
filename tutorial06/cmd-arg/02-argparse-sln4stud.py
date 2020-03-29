#!/usr/bin/env python3

"""
Napište program, který bude brát dva vstupní argumenty. První bude cesta k souboru, druhý řeťezec. Program bude vypíše ty řádky vstupního souboru, které obsahují řetězec. Přitom na začátek každé řádky přidá její pořadové číslo v souboru plus počet slov a počet znaků v příslušné řádce.
"""

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("soubor", help="soubor ke zpracování")
parser.add_argument("text", help="hledaný řetězec")
args = parser.parse_args()

with open(args.soubor, mode='r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if args.text in line:
            slov  = len( line.split() )
            znaku = len( line )
            print(i, slov, znaku, line, end='')
