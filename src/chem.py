import numpy as np
# RDKit
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdCoordGen
from rdkit.Chem.Draw import rdMolDraw2D

# 
def get_highlightInfo(mol, bit_list, bitInfo, colors, radius=0.3):
    """ Get highlighted atoms information
    input
    --------------------
    mol (mol)
    bit_list (list)
    bitInfo (turple)
    colors (list)
    radius (float)

    output
    --------------------
    atoms (list)
    atoms_colors (turple)
    radii (turple)
    bonds (list)
    bonds_colors (turple)
    """
    atoms, atoms_colors, radii, bonds, bonds_colors = [], {}, {}, [], {}

    # bonds & bonds_colors
    for bit, color in zip(bit_list, colors):
      try:
        sub = bitInfo[bit]
        center, rad = sub[0][0], sub[0][1]
        env = Chem.FindAtomEnvironmentOfRadiusN(mol,rad,center)
        for e in env: bonds_colors[e] = color
      except: continue

    # atoms & atoms_colors
    bonds = list(bonds_colors.keys())
    for bond in mol.GetBonds():
        idx = bond.GetIdx()
        if idx in bonds:
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()

            atoms.extend([atom1,atom2])
            atoms_colors[atom1] = bonds_colors[idx]
            atoms_colors[atom2] = bonds_colors[idx]
            radii[atom1] = radius
            radii[atom2] = radius

    return atoms, atoms_colors, radii, bonds, bonds_colors


# 
def vis_attentionedBit(smiles, weight, name='', t_w=0.1, ecfp_rad=2, ecfp_size=2048, figsize=(300,300)):
    """ Visualize compound structure with important bit substructure highlighted.
    input
    --------------------
    smiles (str) : canonical SMILES
    weight (list) : importance (from permutation importance)
    name (str) : molecular name
    t_w (float) : threshold of importance. Important bits above threshold will be reflected on visualization.
    ecfp_rad (int)
    ecfp_size (int)
    figsize (turple)

    output
    --------------------
    view (MolDraw2D)
    """

    # Get bitInfo
    bitInfo = {}
    mol = Chem.MolFromSmiles(smiles)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, ecfp_rad, ecfp_size, bitInfo=bitInfo)

    # Get weighted bit (assending order) : weight, bit, color
    bit_list = np.where(weight>t_w)[0]
    _weight = [weight[i] for i in bit_list]
    _weight, bit_list = zip(*sorted(zip(_weight, bit_list)))
    colors = [(214/256,126/256,126/256)]*len(_weight)

    # Set coloring options
    atoms, atoms_colors, radii, bonds, bonds_colors = get_highlightInfo(mol, bit_list, bitInfo, colors)

    # View
    mol.RemoveAllConformers()
    rdCoordGen.AddCoords(mol)
    view = rdMolDraw2D.MolDraw2DSVG(figsize[0],figsize[1])
    tm = rdMolDraw2D.PrepareMolForDrawing(mol)
    view.DrawMolecule(tm,
                      highlightAtoms = atoms,
                      highlightAtomColors = atoms_colors,
                      highlightAtomRadii = radii,
                      highlightBonds = bonds,
                      highlightBondColors = bonds_colors,
                      legend = name)
    view.FinishDrawing()
    return view.GetDrawingText()