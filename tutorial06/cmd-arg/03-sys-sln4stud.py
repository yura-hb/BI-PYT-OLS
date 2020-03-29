#!/usr/bin/env python3

"""
Napište program, který bude brát tři vstupní argumenty. První bude cesta k souboru se vstupem, druhý řetězec a třetí cesta k výstupnímu souboru. Program bude do výstupního souboru zapíše ty řádky vstupního souboru, které obsahují řetězec. Přitom na začátek každé řádky přidá její pořadové číslo v souboru. Počet slov v příslušné řádce bude ve výstupu vytištěn před každou řádkou.
"""

import sys

if len(sys.argv) != 4:
    #print( "Usage: python3 %s SouborIN TEXT SouborOUT " % sys.argv[0] )
    print( "Usage: python3 {} SouborIN TEXT SouborOUT ".format( sys.argv[0] ) )
    sys.exit()
soubor_in  = sys.argv[1]
text       = sys.argv[2]
soubor_out = sys.argv[3]

lines = []
with open(soubor_in, mode='r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if text in line:
            slov  = len( line.split() )
            znaku = len( line )
            lines.append( (i, slov, znaku, line) )

with open(soubor_out, mode='w', encoding='utf-8') as f:
    for line in lines:
        f.write( str(line[1]) + '\n' )
        f.write( ' '.join( [str(i) for i in line] ) )
