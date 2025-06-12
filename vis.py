import py3Dmol
import time
from helix_from_pdb import helices_from_pdb


pdb_file_name = "1al1.pdb"

# Returns a list of helix_worlds, each a list of (x, y, z) coordinates
helix_worlds = helices_from_pdb(pdb_file_name, return_params=False)

# Flatten all helix coords into one long PDB-style string
pdb_model_lines = []
atom_counter = 1
for helix in helix_worlds:
    for x, y, z in helix: # notice that, for out vis purposes, the only thing that has to be accurate is the coords
        pdb_model_lines.append(
            f"ATOM  {atom_counter:5d}  CA  ALA A{atom_counter:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
        )
        atom_counter += 1
pdb_model_lines.append("END\n")
pdb_model = "".join(pdb_model_lines)

# viewer
view = py3Dmol.view(width=800, height=600)

# Add original structure as model 0
with open(pdb_file_name) as f:
    pdb_orig = f.read()
view.addModel(pdb_orig, 'pdb')
view.setStyle({'model': 0}, {'cartoon': {'color': 'spectrum'}})

# Add modeled helices as model 1
view.addModel(pdb_model, 'pdb')
view.setStyle({'model': 1}, {'sphere': {'radius': 0.3, 'color': 'magenta'}})

# zoom and export
view.zoomTo()
filename = f'helix_with_overlay.html'
with open(filename, 'w') as out:
    out.write(view._make_html())
print(f"Visualization saved to {filename}")
