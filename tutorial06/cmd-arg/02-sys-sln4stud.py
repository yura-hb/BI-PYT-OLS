#!/usr/bin/env python3

"""
Napište program, který bude brát dva vstupní argumenty. První bude cesta k souboru, druhý řeťezec. Program bude vypíše ty řádky vstupního souboru, které obsahují řetězec. Přitom na začátek každé řádky přidá její pořadové číslo v souboru plus počet slov a počet znaků v příslušné řádce.
"""

import sys

if len(sys.argv) != 3:
    #print( "Usage: python3 %s SOUBOR TEXT " % sys.argv[0] )
    print( "Usage: python3 {} SOUBOR TEXT ".format( sys.argv[0] ) )
    sys.exit()
soubor = sys.argv[1]
text   = sys.argv[2]

with open(soubor, mode='r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if text in line:
            slov  = len( line.split() )
            znaku = len( line )
            print(i, slov, znaku, line, end='')
