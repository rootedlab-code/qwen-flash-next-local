#!/usr/bin/env python3
"""Score the numeric self-consistency of a binary-search explanation.

Reads the answer on stdin, prints JSON on stdout. The JSON keys stay in Italian
because bench-coherence.sh and the saved result files use them.
  array_distinti : how many *different* sequences of >=4 numbers appear (1 = consistent)
  asserzioni     : how many "array[i] = v" claims the text makes
  errate         : how many of those contradict the first array the answer declared

The prompt this scores is Italian, so the patterns below match Italian phrasing
("array[i] e' ..."); port the patterns together with the prompt.
"""
import json
import re
import sys

MIN_ELEMENTI = 4

BRACKET = re.compile(r'\[\s*(\d+(?:\s*,\s*\d+){%d,})\s*\]' % (MIN_ELEMENTI - 1))
TABELLA = re.compile(r'^\s*Valor[ei]\s*:?\s*((?:\d+[ \t]+){%d,}\d+)\s*$' % (MIN_ELEMENTI - 1),
                     re.MULTILINE | re.IGNORECASE)
ASSERZIONE = re.compile(r'(?:array|arr|vettore)\s*\[\s*(\d+)\s*\]\s*(?:=|==|è|e\')\s*(\d+)',
                        re.IGNORECASE)


def sequenze(testo):
    trovate = []
    for corpo in BRACKET.findall(testo):
        trovate.append(tuple(int(x) for x in re.split(r'\s*,\s*', corpo)))
    for corpo in TABELLA.findall(testo):
        trovate.append(tuple(int(x) for x in corpo.split()))
    return trovate


def e_sottosequenza(seq, canonico):
    """True if seq is a contiguous run of canonical.

    A binary search legitimately shows the surviving interval after each step:
    [16, 23, 38, 56] inside [2, 5, 8, 12, 16, 23, 38, 56] is not a second array,
    it is the first one halved. Counting it as a second array would report the
    correct answers as defects.
    """
    n = len(seq)
    return any(canonico[i:i + n] == seq for i in range(len(canonico) - n + 1))


def main():
    testo = sys.stdin.read()
    trovate = sequenze(testo)
    canonico = trovate[0] if trovate else ()
    trovate = [s for s in trovate if not e_sottosequenza(s, canonico)] or [canonico]
    errate = 0
    asserzioni = ASSERZIONE.findall(testo)
    for indice, valore in asserzioni:
        i = int(indice)
        if i < len(canonico) and canonico[i] != int(valore):
            errate += 1
    json.dump({
        'array_distinti': len(set(trovate)),
        'asserzioni': len(asserzioni),
        'errate': errate,
        'array': list(canonico),
    }, sys.stdout)


if __name__ == '__main__':
    main()
