import py3Dmol
import numpy as np
from helix_from_pdb import get_helices, get_helices_atoms_coords, fit_helix

pdb_file_name = '1mjc.pdb'

helices = get_helices(pdb_file_name)
coords = get_helices_atoms_coords(pdb_file_name, helices)
r_fit, omega_fit, phi0_fit, helix_world = fit_helix(coords)

# turn modeled-helix coords into a mini-PDB string 
# (note that everything except the coords dont matter for our visualization purposes)
pdb_model = []
for i, (x,y,z) in enumerate(helix_world, start=1):
    pdb_model.append(f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
pdb_model.append("END\n")
pdb_model = "".join(pdb_model)

# build the py3Dmol view with two models
view = py3Dmol.view(width=800, height=600)
# add original as model 0
with open(pdb_file_name) as f:
    pdb_orig = f.read()
view.addModel(pdb_orig, 'pdb')
view.setStyle({'model': 0}, {'cartoon': {'color':'spectrum'}})
# add modeled helix as model 1
view.addModel(pdb_model, 'pdb')
view.setStyle({'model': 1}, {'sphere': {'radius':0.3}})

# zoom to show both
view.zoomTo()
view.addLabel("Fitted Helix Overlay", {'position': {'x':0,'y':0,'z':0}, 'backgroundColor':'white'})

# export to HTML
html = view._make_html()
with open('helix_with_overlay.html','w') as out:
    out.write(html)
