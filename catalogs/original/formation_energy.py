"""Executable crystal descriptors for formation energy per atom."""

import itertools
import math
import warnings

import networkx as nx
import numpy as np

from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.local_env import CrystalNN, MinimumDistanceNN, VoronoiNN
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from pymatgen.core.periodic_table import CovalentRadius
except ImportError:
    CovalentRadius = None

# R01
def rule_01_crystals_a_high_global_unit(structure):
    try:
        total_hard_sphere_volume = 0.0
        for site in structure.sites:
            symbol = site.specie.symbol
            try:
                radius = Element(symbol).atomic_radius
            except Exception:
                radius = None
            if radius is None:
                return 0.0
            atom_volume = (4.0/3.0) * np.pi * (radius ** 3)
            total_hard_sphere_volume += atom_volume
        cell_volume = structure.volume
        if cell_volume <= 0:
            return 0.0
        packing_fraction = total_hard_sphere_volume / cell_volume
        if not (np.isfinite(packing_fraction) and packing_fraction > 0):
            return 0.0
        return float(packing_fraction)
    except Exception:
        return 0.0

# R02
def rule_02_assign_common_oxidation_states_to(structure):
    common_oxidation_state_ranges = {
        'H': [-1, 1], 'He': [0], 'Li': [1], 'Be': [2], 'B': [3], 'C': [-4, 4], 'N': [-3, 3, 5], 'O': [-2],
        'F': [-1], 'Ne': [0], 'Na': [1], 'Mg': [2], 'Al': [3], 'Si': [-4, 4], 'P': [-3, 3, 5], 'S': [-2, 4, 6],
        'Cl': [-1, 1, 5, 7], 'Ar': [0], 'K': [1], 'Ca': [2], 'Sc': [3], 'Ti': [3, 4], 'V': [3, 4, 5],
        'Cr': [2, 3, 6], 'Mn': [2, 4, 7], 'Fe': [2, 3], 'Co': [2, 3], 'Ni': [2, 3], 'Cu': [1, 2],
        'Zn': [2], 'Ga': [3], 'Ge': [2, 4], 'As': [-3, 3, 5], 'Se': [-2, 4, 6], 'Br': [-1, 1, 5],
        'Kr': [0, 2], 'Rb': [1], 'Sr': [2], 'Y': [3], 'Zr': [4], 'Nb': [5], 'Mo': [4, 6], 'Tc': [7],
        'Ru': [3, 4], 'Rh': [3], 'Pd': [2, 4], 'Ag': [1], 'Cd': [2], 'In': [3], 'Sn': [2, 4],
        'Sb': [-3, 3, 5], 'Te': [-2, 4, 6], 'I': [-1, 1, 5, 7], 'Xe': [0, 2, 4, 6, 8], 'Cs': [1],
        'Ba': [2], 'La': [3], 'Ce': [3, 4], 'Pr': [3, 4], 'Nd': [3], 'Pm': [3], 'Sm': [2, 3],
        'Eu': [2, 3], 'Gd': [3], 'Tb': [3, 4], 'Dy': [3], 'Ho': [3], 'Er': [3], 'Tm': [2, 3],
        'Yb': [2, 3], 'Lu': [3], 'Hf': [4], 'Ta': [5], 'W': [4, 6], 'Re': [4, 6, 7],
        'Os': [4, 6], 'Ir': [3, 4], 'Pt': [2, 4], 'Au': [1, 3], 'Hg': [1, 2], 'Tl': [1, 3],
        'Pb': [2, 4], 'Bi': [3, 5], 'Po': [-2, 2, 4, 6], 'At': [-1, 1, 3, 5, 7], 'Rn': [0, 2, 4],
        'Fr': [1], 'Ra': [2], 'Ac': [3], 'Th': [4], 'Pa': [5], 'U': [4, 6], 'Np': [3, 4, 5, 6, 7],
        'Pu': [3, 4, 5, 6], 'Am': [3, 4, 5, 6], 'Cm': [3]
    }
    try:
        formula_elements = list(structure.composition.elements)
        element_symbols = [el.symbol for el in formula_elements]
        stoichs = [structure.composition.get_atomic_fraction(el) * structure.composition.num_atoms for el in formula_elements]
        stoichs = [float(round(s)) for s in stoichs]
        oxidation_state_lists = []
        for el in formula_elements:
            try:
                states = common_oxidation_state_ranges.get(el.symbol, list(el.common_oxidation_states))
                if len(states) == 0:
                    states = [0]
            except Exception:
                states = [0]
            oxidation_state_lists.append(states)
        min_abs_mean = None
        best_assign = None
        for state_tuple in itertools.product(*oxidation_state_lists):
            total_charge = sum(z * n for z, n in zip(state_tuple, stoichs))
            if abs(total_charge) < 1e-3:
                abs_mean = sum(abs(z) * n for z, n in zip(state_tuple, stoichs)) / sum(stoichs)
                if (min_abs_mean is None) or (abs_mean > min_abs_mean):
                    min_abs_mean = abs_mean
                    best_assign = state_tuple
        if best_assign is not None and np.isfinite(min_abs_mean):
            return float(min_abs_mean)
        else:
            return 0.0
    except Exception:
        return 0.0

# R03
def rule_03_minimize_composition_weighted_dispersion_of(structure):
    try:
        composition = structure.composition
        element_types = list(composition.elements)
        occupancies = np.array([composition.get_atomic_fraction(el) for el in element_types])
        radii = []
        for el in element_types:
            r = None
            try:
                r = Element(el.symbol).average_ionic_radius
            except Exception:
                r = None
            if r is None:
                r = Element(el.symbol).atomic_radius
            if r is None or np.isnan(r):
                r = 0.0
            radii.append(r)
        radii = np.array(radii, dtype=float)
        if np.any(radii <= 0):
            return 0.0
        weighted_mean = np.sum(radii * occupancies)
        weighted_squared = np.sum((radii ** 2) * occupancies)
        weighted_var = weighted_squared - weighted_mean ** 2
        weighted_std = np.sqrt(max(weighted_var, 0.0))
        if weighted_mean <= 0:
            return 0.0
        return float(weighted_std / weighted_mean)
    except Exception:
        return 0.0

# R04
def rule_04_use_atomic_fraction_of_metallic(structure):
    try:
        comp = structure.composition
        total_atoms = comp.num_atoms
        if total_atoms == 0:
            return 0.0
        metallic_count = 0.0
        for el, amt in comp.get_el_amt_dict().items():
            try:
                element = Element(el)
                if hasattr(element, 'is_metal') and element.is_metal:
                    metallic_count += amt
            except Exception:
                continue
        return float(metallic_count / total_atoms)
    except Exception:
        return 0.0

# R05
def rule_05_use_composition_weighted_mean_first(structure):
    try:
        comp = structure.composition
        total_atoms = comp.num_atoms
        if total_atoms <= 0:
            return 0.0
        elem_fracs = comp.fractional_composition.items()
        s = 0.0
        for el, frac in elem_fracs:
            E = Element(str(el))
            ionization_energies = getattr(E, 'ionization_energies', None)
            if ionization_energies and len(ionization_energies) > 0 and ionization_energies[0] is not None:
                s += frac * float(ionization_energies[0])
            else:
                return 0.0
        return float(s)
    except Exception:
        return 0.0

# R06
def rule_06_use_multiplicity_weighted_variance_of(structure):
    try:
        sga = SpacegroupAnalyzer(structure)
        symm_struct = sga.get_symmetrized_structure()
        coord_numbers = []
        multiplicities = []
        vnn = VoronoiNN()
        for eq_sites in symm_struct.equivalent_sites:
            site = eq_sites[0]
            try:
                n = len(vnn.get_nn_info(structure, structure.sites.index(site)))
            except Exception:
                n = 0
            coord_numbers.append(float(n))
            multiplicities.append(len(eq_sites))
        if len(coord_numbers) == 0 or any(np.isnan(cn) for cn in coord_numbers):
            return 0.0
        multiplicities = np.array(multiplicities, dtype=float)
        coord_numbers = np.array(coord_numbers, dtype=float)
        weights = multiplicities / multiplicities.sum()
        avg = np.dot(coord_numbers, weights)
        var = np.dot(weights, (coord_numbers - avg) ** 2)
        if not np.isfinite(var):
            return 0.0
        return float(var)
    except Exception:
        return 0.0

# R07
def rule_07_use_site_occupancy_weighted_mean(structure):
    try:
        total_weight = 0.0
        weighted_sum = 0.0
        cnn = CrystalNN()
        for site_idx, site in enumerate(structure.sites):
            for sp, occ in site.species.items():
                r = None
                oxi = getattr(sp, 'oxi_state', None)
                if oxi is not None:
                    try:
                        cn = cnn.get_cn(structure, site_idx)
                        try:
                            str_cn = {4: 'IV', 5: 'V', 6: 'VI', 8: 'VIII'}.get(int(round(cn)), 'VI')
                        except Exception:
                            str_cn = 'VI'
                        r = getattr(sp, 'ionic_radii', {}).get((oxi, str_cn), None)
                        if r is None:
                            r = getattr(sp, 'ionic_radius', None)
                    except Exception:
                        r = getattr(sp, 'ionic_radius', None)
                if r is None or (hasattr(r, '__le__') and r <= 0):
                    try:
                        r = getattr(sp, 'covalent_radius', None)
                    except Exception:
                        r = None
                if r is None or (hasattr(r, '__le__') and r <= 0):
                    continue
                weighted_sum += occ * r
                total_weight += occ
        if total_weight > 0.0 and np.isfinite(weighted_sum / total_weight):
            return float(weighted_sum / total_weight)
        else:
            return 0.0
    except Exception:
        return 0.0

# R08
def rule_08_high_coordination_weighted_mean_electronegativity(structure):
    try:
        vnn = VoronoiNN()
        n_sites = len(structure.sites)
        bond_diffs = []
        for idx, site in enumerate(structure.sites):
            try:
                neighbors = vnn.get_nn_info(structure, idx)
            except Exception:
                continue
            site_elem = site.specie.element if hasattr(site.specie, "element") else Element(str(site.specie))
            xi = site_elem.X
            site_weights = []
            for nn in neighbors:
                w = nn.get('weight', 1.0)
                nn_elem = nn['site'].specie.element if hasattr(nn['site'].specie, "element") else Element(str(nn['site'].specie))
                xj = nn_elem.X
                diff = abs(xi - xj)
                site_weights.append(w * diff)
            if site_weights:
                bond_diffs.append(np.sum(site_weights))
        if bond_diffs and n_sites > 0:
            mean = float(np.sum(bond_diffs) / float(n_sites))
            if np.isfinite(mean):
                return mean
        return 0.0
    except Exception:
        return 0.0

# R09
def rule_09_multiplicity_weighted_mean_coordination_number(structure):
    # Use available VoronoiNN and SpacegroupAnalyzer from module level names
    site_cn = []
    multiplicities = []
    vnn = VoronoiNN(allow_pathological=True)
    try:
        sga = SpacegroupAnalyzer(structure)
        equiv_atoms = sga.get_symmetry_dataset()["equivalent_atoms"]
    except Exception:
        equiv_atoms = [i for i in range(len(structure.sites))]
    for i, site in enumerate(structure.sites):
        try:
            cn = vnn.get_cn(structure, i)
        except Exception:
            cn = 0.0
        site_cn.append(cn)
        # Try getting the multiplicity
        if hasattr(site, 'properties') and site.properties is not None and 'multiplicity' in site.properties:
            multiplicity = site.properties['multiplicity']
        else:
            mult = sum(1 for eqi in equiv_atoms if eqi == i)
            multiplicity = mult if mult > 0 else 1
        multiplicities.append(multiplicity)
    weighted_sum = float(np.dot(site_cn, multiplicities))
    total_mult = float(np.sum(multiplicities))
    if total_mult > 0.0 and np.isfinite(weighted_sum / total_mult):
        return weighted_sum / total_mult
    return 0.0

# R10
def rule_10_structures_a_low_mean_absolute(structure):
    struct = structure.copy()
    # Try to assign oxidation states using BVAnalyzer as fallback if add_oxidation_state_by_guess fails
    try:
        struct.add_oxidation_state_by_guess()
        oxi_state_assigned = True
    except Exception:
        oxi_state_assigned = False
        try:
            struct = BVAnalyzer().get_oxi_state_decorated_structure(struct)
            oxi_state_assigned = True
        except Exception:
            oxi_state_assigned = False
    # Try CrystalNN, fallback to VoronoiNN for neighbor list
    for nn_class in [CrystalNN, VoronoiNN]:
        try:
            nn_finder = nn_class()
            nlist = [nn_finder.get_nn_info(struct, i) for i in range(len(struct))]
            break
        except Exception:
            nlist = None
            continue
    if nlist is None:
        return 0.0
    bond_fracs = []
    for i, site in enumerate(struct):
        site_elem = site.specie.symbol
        if oxi_state_assigned and hasattr(site.specie, 'oxi_state'):
            # Try Shannon radius with oxi state
            elem_i = Element(site_elem)
            oxi_i = int(round(float(getattr(site.specie, 'oxi_state', 0))))
            r_i = elem_i.ionic_radii.get(oxi_i, elem_i.average_ionic_radius)
        else:
            r_i = Element(site_elem).atomic_radius
        if r_i is None:
            r_i = Element(site_elem).atomic_radius
        for ninfo in nlist[i]:
            neigh_spec = ninfo['site'].specie
            neigh_elem = neigh_spec.symbol
            if oxi_state_assigned and hasattr(neigh_spec, 'oxi_state'):
                elem_j = Element(neigh_elem)
                oxi_j = int(round(float(getattr(neigh_spec, 'oxi_state', 0))))
                r_j = elem_j.ionic_radii.get(oxi_j, elem_j.average_ionic_radius)
            else:
                r_j = Element(neigh_elem).atomic_radius
            if r_j is None:
                r_j = Element(neigh_elem).atomic_radius
            if r_i is None or r_j is None:
                continue
            d_ij = site.distance(ninfo['site'])
            r_sum = r_i + r_j
            if r_sum <= 0:
                continue
            frac = abs(d_ij - r_sum) / r_sum
            bond_fracs.append(frac)
    bond_fracs = [f for f in bond_fracs if np.isfinite(f)]
    if not bond_fracs:
        return 0.0
    return float(np.mean(bond_fracs))

# R11
def rule_11_composition_weighted_mean_valence_electrons(structure):
    try:
        comp = structure.composition
        total_atoms = comp.num_atoms
        if total_atoms == 0:
            return 0.0
        total_valence_e = 0.0
        for el, amt in comp.items():
            try:
                elem = Element(el)
                val_e = elem.full_electronic_structure
                # Find the sum of electrons in the highest principal quantum number shell (valence shell)
                max_n = max([t[0] for t in val_e])
                val_electrons = sum(t[2] for t in val_e if t[0] == max_n)
                total_valence_e += val_electrons * float(amt)
            except Exception:
                group = None
                try:
                    group = Element(el).group
                except Exception:
                    return 0.0
                if group is None:
                    return 0.0
                if group <= 2:
                    val_electrons = group
                elif group <= 12:
                    val_electrons = (group - 2) + 2
                elif group <= 18:
                    val_electrons = group - 10
                else:
                    val_electrons = 0
                total_valence_e += val_electrons * float(amt)
        mean_valence = total_valence_e / total_atoms
        if not (np.isfinite(mean_valence) and mean_valence >= 0.0):
            return 0.0
        return float(mean_valence)
    except Exception:
        return 0.0

# R12
def rule_12_materials_a_higher_fraction_of(structure):
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.01)
        symm_struct = sga.get_symmetrized_structure()
        dataset = sga.get_symmetry_dataset()
        wyckoff_info = dataset['wyckoffs'] if 'wyckoffs' in dataset else []
        equivalent_atoms = dataset['equivalent_atoms'] if 'equivalent_atoms' in dataset else []
        site_symmetry_symbols = dataset['site_symmetry_symbols'] if 'site_symmetry_symbols' in dataset else []
        site_multiplicities = [getattr(site, 'wyckoff_multiplicity', 1) for site in getattr(symm_struct, 'sites', [])]
        site_letters = getattr(symm_struct, 'wyckoff_letters', [None]*len(site_multiplicities))
        # Fallback general multiplicity determination
        if site_multiplicities:
            try:
                general_mult = max(site_multiplicities)
            except Exception:
                general_mult = 1
        else:
            general_mult = 1
        special_flags = []
        for idx, (site, letter, mult) in enumerate(zip(getattr(symm_struct, 'sites', []), site_letters, site_multiplicities)):
            # If multiplicity lower than general position
            if mult < general_mult:
                special_flags.append(1)
            else:
                # Check if site symmetry > 1
                sym = site_symmetry_symbols[idx] if idx < len(site_symmetry_symbols) else ''
                order = 1
                for c in str(sym):
                    if c.isdigit():
                        order = max(order, int(c))
                if order > 1:
                    special_flags.append(1)
                else:
                    special_flags.append(0)
        total = float(len(special_flags))
        if total > 0:
            f_special = float(sum(special_flags)) / total
            if np.isfinite(f_special):
                return f_special
            else:
                return 0.0
        else:
            return 0.0
    except Exception:
        return 0.0

# R13
def rule_13_at_fixed_composition_a_larger(structure):
    try:
        comp = structure.composition
        total = comp.num_atoms
        if total == 0:
            return 0.0
        chi_list = []
        weights = []
        for el, amt in comp.items():
            try:
                ele = Element(el)
                chi = ele.X
                if chi is None:
                    return 0.0
            except Exception:
                return 0.0
            chi_list.append(chi)
            weights.append(amt)
        weights = np.array(weights)
        chi_array = np.array(chi_list)
        if weights.sum() == 0:
            return 0.0
        mean = np.average(chi_array, weights=weights)
        variance = np.average((chi_array - mean) ** 2, weights=weights)
        stddev = math.sqrt(variance)
        if not np.isfinite(stddev):
            return 0.0
        return float(stddev)
    except Exception:
        return 0.0

# R14
def rule_14_compounds_a_composition_weighted_mean(structure):
    try:
        comp = structure.composition
        total = comp.num_atoms
        if total == 0:
            return 0.0
        weighted_sum = 0.0
        for el, amt in comp.items():
            try:
                en = Element(str(el)).X
            except Exception:
                return 0.0
            if en is None or not np.isfinite(en):
                return 0.0
            weighted_sum += amt * en
        mean_en = weighted_sum / total
        if not np.isfinite(mean_en):
            return 0.0
        return float(mean_en)
    except Exception:
        return 0.0

# R15
def rule_15_use_composition_weighted_mean_atomic(structure):
    try:
        comp = structure.composition
        total = comp.num_atoms
        if total <= 0:
            return 0.0
        mean_Z = 0.0
        for el, amt in comp.items():
            try:
                Z = Element(str(el)).Z
                mean_Z += (amt / total) * Z
            except Exception:
                return 0.0
        if not (np.isfinite(mean_Z) and mean_Z > 0.0):
            return 0.0
        return float(mean_Z)
    except Exception:
        return 0.0

# R16
def rule_16_use_composition_weighted_mean_pairwise(structure):
    try:
        comp = structure.composition.fractional_composition
        el_amt = comp.get_el_amt_dict()
        elements = sorted(el_amt.keys())
        fractions = [comp[el] for el in elements]
        chis = []
        for el in elements:
            try:
                chis.append(Element(el).X)
            except Exception:
                return 0.0
        n = len(elements)
        if n < 2:
            return 0.0
        delta_chi = 0.0
        for i in range(n - 1):
            for j in range(i + 1, n):
                xixj = fractions[i] * fractions[j]
                dchi = abs(chis[i] - chis[j])
                delta_chi += xixj * dchi
        val = 2 * delta_chi
        return float(val) if math.isfinite(val) else 0.0
    except Exception:
        return 0.0

# R17
def rule_17_compositional_configurational_entropy_stabilizes_multicomponent(structure):
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.01)
        sym_struct = sga.get_symmetrized_structure()
        total_sites = len(structure.sites)
        # Fallback: check if equivalent_sites exists and is non-empty
        if not hasattr(sym_struct, 'equivalent_sites') or not sym_struct.equivalent_sites:
            comp = structure.composition.get_el_amt_dict()
            total = sum(comp.values())
            x_i = [amt / total for amt in comp.values() if amt > 0]
            entropy = -sum(x * np.log(x) for x in x_i) if all(x_i) else 0.0
            return float(entropy)
        total_entropy = 0.0
        total_mult = 0
        for sites in sym_struct.equivalent_sites:
            occ_dict = {}
            for site in sites:
                for sp, occ in site.species.items():
                    el = str(sp)
                    occ_dict[el] = occ_dict.get(el, 0.0) + occ
            num_sites = len(sites)
            fracs = [occ/num_sites for occ in occ_dict.values() if occ > 0]
            if len(fracs) > 1:
                s = -sum(x * np.log(x) for x in fracs)
            else:
                s = 0.0
            total_entropy += num_sites * s
            total_mult += num_sites
        if total_mult == 0:
            return 0.0
        return float(total_entropy / total_mult)
    except Exception:
        return 0.0

# R18
def rule_18_use_composition_weighted_mean_of(structure):
    # Cohesive energies (in eV/atom) from Kittel or literature for main elements
    cohesive_energies = {
        'H': 2.31,
        'He': 0.00,
        'Li': 1.63,
        'Be': 3.32,
        'B': 5.81,
        'C': 7.37,
        'N': 0.00,
        'O': 0.00,
        'F': 0.00,
        'Ne': 0.00,
        'Na': 1.13,
        'Mg': 1.51,
        'Al': 3.39,
        'Si': 4.63,
        'P': 3.17,
        'S': 2.87,
        'Cl': 0.00,
        'Ar': 0.00,
        'K': 0.93,
        'Ca': 1.84,
        'Sc': 3.90,
        'Ti': 4.85,
        'V': 5.31,
        'Cr': 4.10,
        'Mn': 2.92,
        'Fe': 4.28,
        'Co': 4.39,
        'Ni': 4.44,
        'Cu': 3.49,
        'Zn': 1.35,
        'Ga': 2.81,
        'Ge': 3.85,
        'As': 2.95,
        'Se': 2.35,
        'Br': 0.00,
        'Kr': 0.00,
        'Rb': 0.82,
        'Sr': 1.76,
        'Y': 4.05,
        'Zr': 6.25,
        'Nb': 7.57,
        'Mo': 6.82,
        'Tc': 7.20,
        'Ru': 6.74,
        'Rh': 5.75,
        'Pd': 3.89,
        'Ag': 2.95,
        'Cd': 1.16,
        'In': 2.52,
        'Sn': 3.14,
        'Sb': 2.74,
        'Te': 2.51,
        'I': 1.11,
        'Xe': 0.00,
        'Cs': 0.81,
        'Ba': 1.88,
        'La': 4.24,
        'Ce': 4.10,
        'Pr': 3.67,
        'Nd': 3.38,
        'Pm': 2.90,
        'Sm': 2.17,
        'Eu': 1.76,
        'Gd': 3.11,
        'Tb': 3.66,
        'Dy': 2.45,
        'Ho': 2.48,
        'Er': 2.70,
        'Tm': 2.44,
        'Yb': 1.60,
        'Lu': 4.18,
        'Hf': 6.44,
        'Ta': 8.10,
        'W': 8.90,
        'Re': 8.03,
        'Os': 8.17,
        'Ir': 6.94,
        'Pt': 5.84,
        'Au': 3.81,
        'Hg': 0.67,
        'Tl': 2.00,
        'Pb': 2.03,
        'Bi': 2.16,
        'Po': 2.0,
        'At': 2.0,
        'Rn': 0.0
    }
    try:
        comp = structure.composition
        total = comp.num_atoms
        if total == 0:
            return 0.0
        mean = 0.0
        for el, amt in comp.items():
            ce = cohesive_energies.get(el.symbol, None)
            if ce is None:
                return 0.0
            mean += ce * (amt / total)
        if not np.isfinite(mean):
            return 0.0
        return float(mean)
    except Exception:
        return 0.0

# R19
def rule_19_larger_composition_weighted_pairwise_atomic(structure):
    try:
        comp = structure.composition.get_el_amt_dict()
        total = sum(comp.values())
        elements = list(comp.keys())
        xs = [comp[el]/total for el in elements]
        Zs = [Element(el).Z for el in elements]
        n = len(elements)
        if n < 2:
            return 0.0
        mean_delta_Z = 0.0
        for i in range(n):
            for j in range(i+1, n):
                mean_delta_Z += xs[i] * xs[j] * abs(Zs[i] - Zs[j])
        return float(mean_delta_Z)
    except Exception:
        return 0.0

# R20
def rule_20_quantify_geometric_volume_per_atom(structure):
    try:
        v_cell = structure.volume
        n_atoms = len(structure.sites)
        if n_atoms <= 0:
            return 0.0
        v_atom = v_cell / n_atoms
        if not (np.isfinite(v_atom) and v_atom > 0):
            return 0.0
        return float(v_atom)
    except Exception:
        return 0.0

# R21
def rule_21_use_composition_weighted_mean_elemental(structure):
    try:
        comp = structure.composition
        elements = comp.elements
        fractions = [comp.get_atomic_fraction(el) for el in elements]
        melting_points = []
        for el in elements:
            try:
                mp = Element(el.symbol).melting_point
                if mp is None or not np.isfinite(mp):
                    return 0.0
                melting_points.append(mp)
            except Exception:
                return 0.0
        mean = 0.0
        for f, mp in zip(fractions, melting_points):
            mean += f * mp
        if not np.isfinite(mean):
            return 0.0
        return mean
    except Exception:
        return 0.0

# R22
def rule_22_a_higher_bond_count_weighted(structure):
    try:
        cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0, porous_adjustment=False)
        graph = cnn.get_bonded_structure(structure)
        total_bonds = 0.0
        ionic_like_bonds = 0.0
        for idx, site in enumerate(graph.structure.sites):
            neighbors = graph.get_connected_sites(idx)
            for neigh in neighbors:
                j = neigh.index
                if idx < j:
                    specie_i = site.specie
                    specie_j = structure[j].specie
                    try:
                        chi1 = Element(str(specie_i)).X
                        chi2 = Element(str(specie_j)).X
                        if chi1 is None or chi2 is None:
                            continue
                        dchi = abs(chi1 - chi2)
                    except Exception:
                        continue
                    weight = neigh.weight if hasattr(neigh, "weight") else 1.0
                    total_bonds += weight
                    if dchi >= 1.7:
                        ionic_like_bonds += weight
        if total_bonds == 0.0:
            return 0.0
        return float(ionic_like_bonds / total_bonds)
    except Exception:
        return 0.0

# R23
def rule_23_compounds_smaller_composition_site_occupancy(structure):
    try:
        species_mass_occ = []
        comp = structure.composition.get_el_amt_dict()
        total_atoms = sum(comp.values())
        for site in structure.sites:
            occ_sum = 0.0
            mass = 0.0
            for sp, occ in site.species.items():
                # Use Element for atomic mass if needed
                if hasattr(sp, 'atomic_mass'):
                    m = sp.atomic_mass
                else:
                    try:
                        m = Element(sp.symbol).atomic_mass
                    except Exception:
                        m = 0.0
                mass += m * occ
                occ_sum += occ
            # 'site.occupancy' is not a standard attribute, estimate from sum of fractional occupancies
            estimated_occ = occ_sum
            if occ_sum > 0:
                species_mass_occ.append((mass / occ_sum, estimated_occ))
        if not species_mass_occ:
            return 0.0
        masses = np.array([x[0] for x in species_mass_occ])
        occs = np.array([x[1] for x in species_mass_occ])
        occ_sum_total = np.sum(occs)
        if occ_sum_total == 0:
            return 0.0
        weights = occs / occ_sum_total
        mean_mass = np.sum(masses * weights)
        variance = np.sum(weights * (masses - mean_mass) ** 2)
        sigma_m = math.sqrt(variance)
        if math.isfinite(sigma_m):
            return float(sigma_m)
        else:
            return 0.0
    except Exception:
        return 0.0

# R24
def rule_24_a_higher_atomic_fraction_of(structure):
    try:
        composition = structure.composition
        total_atoms = composition.num_atoms
        if total_atoms == 0:
            return 0.0
        df_block_atoms = 0
        for el, amt in composition.get_el_amt_dict().items():
            try:
                elem = Element(el)
                Z = elem.Z
                group = elem.group
                period = elem.row
                is_d_block = 3 <= period <= 7 and 3 <= group <= 12
                is_f_block = (period == 6 or period == 7) and ((57 <= Z <= 71) or (89 <= Z <= 103))
                if is_d_block or is_f_block:
                    df_block_atoms += amt
            except Exception:
                continue
        return float(df_block_atoms) / float(total_atoms)
    except Exception:
        return 0.0

# R25
def rule_25_use_composition_weighted_mean_mendeleev(structure):
    mendeleev_numbers = {
        'H': 1,  'He': 2,  'Li': 3,  'Be': 4,  'B': 5,  'C': 6,  'N': 7,  'O': 8,  'F': 9,  'Ne': 10,
        'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
        'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27,
        'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
        'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46,
        'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56,
        'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
        'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76,
        'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86,
        'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
        'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 
        'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Fl': 114, 'Lv': 116
    }
    try:
        comp = structure.composition
        total = 0.0
        m_sum = 0.0
        for el, amt in comp.items():
            sym = el.symbol
            mn = mendeleev_numbers.get(sym)
            if mn is None or not np.isfinite(mn):
                return 0.0
            m_sum += amt * mn
            total += amt
        if total <= 0:
            return 0.0
        m_av = float(m_sum) / total
        if not np.isfinite(m_av):
            return 0.0
        return m_av
    except Exception:
        return 0.0
