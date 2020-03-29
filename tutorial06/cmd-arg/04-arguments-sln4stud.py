#!/usr/bin/env python3

"""
### 4. Napište program, který za pomoci sys.argv vyhodnotí své vstupní argumenty následujícím způsobem:

* samostatné přepínače -x uloží v podobě n-tice (x, ... );
* přepínače --xarg=hodnota uloží v podobě slovníku { 'xarg': hodnota, ... };
* vlastní argumenty uloží do seznamu.

(Tj. zjednodušeně předpokládáme, že zkrácené přepínače nemají žádnou hodnotu a dlouhé přepínače vždy obsahují rovnítko, abychom od sebe všechny tři kategorie argumentů mohli snadno rozlišit.)
"""

import sys

argumenty = []
prepinace_kratke = []
prepinace_dlouhe = []

for arg in sys.argv[1:]:
    if arg.startswith('--'):
        k, v = arg[2:].split('=')
        prepinace_dlouhe.append( (k, v) )
    elif arg.startswith('-'):
        k = arg[1:]
        prepinace_kratke.append(k)
    else:
        argumenty.append(arg)

argumenty = list(argumenty)
prepinace_kratke = tuple(prepinace_kratke)
prepinace_dlouhe = dict(prepinace_dlouhe)

print(argumenty)
print(prepinace_kratke)
print(prepinace_dlouhe)
