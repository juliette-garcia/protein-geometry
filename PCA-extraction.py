# this script references the PCA idea to extract the helical axis from the paper below:
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0257318#sec001
import numpy as np
from sklearn.decomposition import PCA


PDB_FILE = '1mjc.pdb'

# get the helices from the file
# formatting: https://www.wwpdb.org/documentation/file-format-content/format33/sect5.html

helices = []
with open(PDB_FILE) as f:
    for line in f:
        if not line.startswith("HELIX "):
            continue
        helices.append({
            'serNum'      : int(line[7:10].strip()),
            'helixID'     : line[11:14].strip(),
            'initResName' : line[15:18].strip(),
            'initChainID' : line[19].strip(),
            'initSeqNum'  : int(line[21:25].strip()),
            'initICode'   : line[25].strip() or None,
            'endResName'  : line[27:30].strip(),
            'endChainID'  : line[31].strip(),
            'endSeqNum'   : int(line[33:37].strip()),
            'endICode'    : line[37].strip() or None,
            'helixClass'  : int(line[38:40].strip()),
            'comment'     : line[40:70].rstrip(),
            'length'      : int(line[71:76].strip()),
        })


if not helices:
    print("No HELIX records found.")
else:
    print(helices)

