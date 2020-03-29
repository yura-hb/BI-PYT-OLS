#!/usr/bin/env python3

"""
Napište program, který bude brát tři vstupní argumenty. První bude cesta k souboru se vstupem, druhý řetězec a třetí cesta k výstupnímu souboru. Program bude do výstupního souboru zapíše ty řádky vstupního souboru, které obsahují řetězec. Přitom na začátek každé řádky přidá její pořadové číslo v souboru. Počet slov v příslušné řádce bude ve výstupu vytištěn před každou řádkou.
"""

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("soubor_in", help="soubor ke zpracování")
parser.add_argument("text", help="hledaný řetězec")
parser.add_argument("soubor_out", help="soubor pro uložení výstupu")
args = parser.parse_args()

lines = []
with open(args.soubor_in, mode='r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if args.text in line:
            slov  = len( line.split() )
            znaku = len( line )
            lines.append( (i, slov, znaku, line) )

with open(args.soubor_out, mode='w', encoding='utf-8') as f:
    for line in lines:
        f.write( str(line[1]) + '\n' )
        f.write( ' '.join( [str(i) for i in line] ) )
