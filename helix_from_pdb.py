# this script references the PCA idea to extract the helical axis from the paper below:
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0257318#sec001
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from math import atan2
from collections import defaultdict
import os
import requests

# fetch PDB File
def fetch_pdb(pdb_id):
    """Download PDB file from RCSB database"""
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"PDB ID {pdb_id} not found")
    
    filename = f"{pdb_id.lower()}.pdb"
    with open(filename, 'w') as f:
        f.write(response.text)
    return filename

def calculate_helical_properties(r, omega):
    # Pitch parameter c from MathWorld: https://mathworld.wolfram.com/Helix.html 
    c = 2 * np.pi / abs(omega)
    # Actual pitch (distance per full turn)
    pitch = 2 * np.pi / abs(omega)
    # Curvature (κ)
    curvature = r / (r**2 + (1/omega)**2)
    # Torsion (τ)
    torsion = (1/omega) / (r**2 + (1/omega)**2)
    return c, pitch, curvature, torsion

# get all the atoms belonging to any helix from the pdb file
# formatting: https://www.wwpdb.org/documentation/file-format-content/format33/sect5.html
def get_helix_atoms(pdb_file):
    helix_atoms = []
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("HELIX "):
                continue
            helix_atoms.append({
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
    if not helix_atoms:
        print("No HELIX records found.")
    return helix_atoms

# get all atom coords from the file that are part of ANY helix
def get_helix_atoms_coords(pdb_file, helix_atoms):
    coords = []
    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM  "):
                continue
            ranges = [(helix_atom['initSeqNum'], helix_atom['endSeqNum']) for helix_atom in helix_atoms]
            resSeq = int(line[22:26].strip())
            for start, end in ranges:
                if start <= resSeq <= end:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
    return np.array(coords)

# https://www.geometrictools.com/Documentation/HelixFitting.pdf
# 1. use PCA to find the helix's main axis and centroid
# 2. now that the as the main axis, turn the helix into a standard helix (s.t. make its main axis the z-axis)
# 3. unwrap phi vs. z, fit φ = ω z + φ0, r = mean radius
# 4. build fitted helix in rotated frame
# 5. rotate back to its original axis
# 6. return helix params and the helix itself (in 3D points)
def get_helix_model(helix_atoms_coords, n_points = 200, return_params = False):
    """
    Parameters
    ----------
    coords : 3D coordinates of the atoms of one helix
    n_points : number of samples along the fitted helix curve (resolution)
    return_params : whether to return the parameters of the helix

    Returns
    -------
    (r, omega, phi0, helix_world) : tuple of floats that are the fitted helix's parameters
        - r = radius of circular cross‐section (only returned if return_params = True)
        - omega = angular frequency per unit z (only returned if return_params = True)
        - phi0 = phase offset (only returned if return_params = True)
        - helix_world = set of 3D coordinates of fitted helix in the original PDB coordinate frame
    """

    # first get the centroid of the point cloud so that we can center the weight of it to the z-axis
    centroid = helix_atoms_coords.mean(axis=0)
    centered = helix_atoms_coords - centroid

    # use PCA to get to get the main axis of the helix 
    pca = PCA(n_components=3)
    pca.fit(centered)
    axis = pca.components_[0]
    axis = axis / np.linalg.norm(axis) # make axis's length 1 (to simplify any trnasformations of it later)

    # get the rotation matrix R that makes the halicoid's axis be parallel to the z-axis
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(axis, z_axis) # v is perpendicular to og axis and z-axis. so, its the axis we need to rotate around
    c = np.dot(axis, z_axis) #  cosine of the angle between the og axis and z-axis
    if np.linalg.norm(v) < 1e-8: # if v is (near) zero, axis is already parallel to Z: no rotation needed
        R = np.eye(3) # thus our rotation matrix can just be the identity matrix
    else: # toherwise, if indeed the axis needs to be rotated, then the rotation matrix is as follows:
        K = np.array([ 
            [    0, -v[2],  v[1]],
            [ v[2],     0, -v[0]],
            [-v[1],  v[0],     0]
        ])
        R = np.eye(3) + K + (K @ K) * ((1 - c) / (np.linalg.norm(v)**2))
        # https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_rotation_matrix_to_axis%E2%80%93angle

    # rotate all points using this rotation matrix. now we have the points our our "standard helix"
    rotated = centered @ R.T
    xs = rotated[:, 0]
    ys = rotated[:, 1]
    zs = rotated[:, 2]

    # unwrap phase φ_i = atan2(y_rot, x_rot) vs z_rot
    raw_phis  = np.array([atan2(y, x) for x, y in zip(xs, ys)])
    # For each point (x_i, y_i ), atan2(y_i, x_i) returns an 
    # angle ϕ_i ​ in the range (−π,+π]. i.e. geometrically, it’s the polar angle of 
    # the point in the xy-plane (measured from the positive x-axis)
    unwrapped = np.unwrap(raw_phis)
    # scans through the raw_phis sequence and whenever it sees a jump of more
    # than π in magnitude between consecutive entries, it adds or subtracts 2π 
    # to that and all next entries to make the sequence change smoothly

    # linear fit φ ≈ ω z + φ_0 (above the partial derivative on page 3 of the paper)
    omega, phi0 = np.polyfit(zs, unwrapped, 1)

    # compute the radius of the helix (the average radial distance of each point from the Z–axis)
    r = np.mean(np.sqrt(xs**2 + ys**2))

    # build fitted helix in rotated frame
    z_fit = np.linspace(zs.min(), zs.max(), n_points)
    phi_fit = omega * z_fit + phi0 # from the equation φ ≈ ω z + φ_0
    x_fit = r * np.cos(phi_fit)
    y_fit = r * np.sin(phi_fit)
    helix_rotated = np.vstack([x_fit, y_fit, z_fit]).T 

    # rotate the fitted helix back into world frame and re‐center
    helix_world = (helix_rotated @ np.linalg.inv(R).T) + centroid

    # return fitted param
    if return_params:
        return r, omega, phi0, helix_world
    else:
        return helix_world


def helices_from_pdb(pdb_file_name, return_params):
    """
    Parameters
    ----------
    pdb_file_name : name of PDB file you want to extract the helix from
    return_params : whether to return the parameters of each helix

    Returns
    -------
    [(r, omega, phi0, helix_worlds)] : a list of tuples of floats that are the fitted helix's parameters for each helix
        - r = radius of circular cross‐section (only returned if return_params = True)
        - omega = angular frequency per unit z (only returned if return_params = True)
        - phi0 = phase offset (only returned if return_params = True)
        - helix_worlds = list of sets of 3D coordinates of fitted helix in the original PDB coordinate frame
    """
    # if the pdb file is not given, try to fetch it from the RCSB database
    if not os.path.exists(pdb_file_name):
        pdb_id = os.path.splitext(os.path.basename(pdb_file_name))[0]
        pdb_file_name = fetch_pdb(pdb_id)

    # get all the atoms that belong to any helix in the pdb file
    all_helix_atoms = get_helix_atoms(pdb_file_name)

    # split into a list of atom dicts (by helix id)
    grouped_helix_atoms = defaultdict(list)
    for atom_dict in all_helix_atoms:
        grouped_helix_atoms[atom_dict['helixID']].append(atom_dict)

    # get helix and params for each group of helix atoms
    output = []
    for grouped_atoms in grouped_helix_atoms.values():
        atoms_coords = get_helix_atoms_coords(pdb_file_name, grouped_atoms)
        if return_params:
            r, omega, phi0, helix_world = get_helix_model(atoms_coords, return_params = True)
            output.append((r, omega, phi0, helix_world))
            c, pitch, curvature, torsion = calculate_helical_properties(r, omega)
            print(f"Helical properties: \n c = {c:.5f} Å\n pitch = {pitch:.5f} Å\n curvature = {curvature:.5f} Å⁻¹\n torsion = {torsion:.5f} Å⁻¹")
        else:
            helix_world = get_helix_model(atoms_coords)
            output.append(helix_world)
    return output