#!/usr/bin/env python3

""" Napište program, který vypíše řádku po řádce soubor, cestu k němuž bude brát jako svůj vstupní argument. Přitom na začátek každé řádky přidá její pořadové číslo v souboru plus počet slov a počet znaků v příslušné řádce. """

import sys

if len(sys.argv) != 2:
    #print( "Usage: python3 %s SOUBOR " % sys.argv[0] )
    print( "Usage: python3 {} SOUBOR ".format( sys.argv[0] ) )
    sys.exit()
soubor = sys.argv[1]

with open(soubor, mode='r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        slov  = len( line.split() )
        znaku = len( line )
        print(i, slov, znaku, line, end='')
