import os
import numpy as np
from functools import lru_cache
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.core.composition import Composition
from pymatgen.util.coord import pbc_shortest_vectors
from pymatgen.analysis.elasticity.elastic import ElasticTensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester
from pymatgen.core import Structure

API_KEY = os.environ.get("MP_API_KEY")


@lru_cache(maxsize=None)
def _summary_doc(mp_id):
    with MPRester(API_KEY) as mpr:
        docs = mpr.materials.summary.search(
            material_ids=[mp_id],
            fields=["material_id", "energy_above_hull", "band_gap", "formation_energy_per_atom"]
        )
    if not docs:
        return {}
    doc = docs[0]
    if hasattr(doc, "model_dump"):
        return doc.model_dump()
    if hasattr(doc, "dict"):
        return doc.dict()
    return dict(doc)


@lru_cache(maxsize=None)
def _prototype_structures(elements_key):
    elements = list(elements_key)
    try:
        with MPRester(API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                elements=elements,
                fields=["structure"],
                chunk_size=10,
                num_chunks=1
            )
        return [doc.structure for doc in docs if getattr(doc, "structure", None) is not None]
    except Exception as e:
        print(f"[prototype-cache] DB fetch failed: {e}")
        return []


@lru_cache(maxsize=None)
def _chemsys_count(chemsys):
    try:
        with MPRester(API_KEY) as mpr:
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                fields=["material_id"]
            )
        return float(len(docs))
    except Exception as e:
        print(f"[chemsys-cache] API fail: {e}")
        return 0.0

def rule_01_thermodynamic_stability(structure, mp_id=None):
    if mp_id is None:
        return 0.0
    doc_data = _summary_doc(mp_id)
    if doc_data:
        if "energy_above_hull" in doc_data and doc_data["energy_above_hull"] is not None:
            return float(doc_data["energy_above_hull"])
        print(f"[WARN] 'energy_above_hull' not found or is None for {mp_id}")
    else:
        print(f"[WARN] No summary found for {mp_id}")

    return 0.0


def rule_02_phonon_stability_proxy(structure):
    vnn = VoronoiNN()
    mean_cn = np.mean([vnn.get_cn(structure, i) for i in range(len(structure))])
    mass = sum(site.specie.atomic_mass for site in structure.sites)
    volume = structure.lattice.volume
    density = mass / volume
    return float(mean_cn * density)

def rule_03_avg_coordination_number(structure):
    try:
        cnn = CrystalNN()
        cn_list = [cnn.get_cn(structure, i) for i in range(len(structure))]
        return float(np.mean(cn_list))
    except:
        return 0.0

def rule_04_space_group_symmetry(structure):
    try:
        return float(SpacegroupAnalyzer(structure).get_space_group_number())
    except:
        return 0.0

def rule_05_packing_density(structure):
    return float(structure.density)

def rule_06_band_gap(structure, mp_id=None):
    if mp_id is None:
        return 0.0
    doc_data = _summary_doc(mp_id)
    if doc_data and doc_data.get("band_gap") is not None:
        return float(doc_data["band_gap"])
    return 0.0

def rule_07_bond_length_angle_deviation(structure):
    cnn = CrystalNN()
    bond_lengths = []
    bond_angles = []
    for i, site in enumerate(structure):
        try:
            neighs = cnn.get_nn_info(structure, i)
        except:
            continue
        coords = [n['site'].coords for n in neighs]
        if not coords:
            continue
        for c in coords:
            bond_lengths.append(np.linalg.norm(site.coords - c))
        if len(coords) >= 2:
            for j in range(len(coords)):
                for k in range(j+1, len(coords)):
                    v1 = coords[j] - site.coords
                    v2 = coords[k] - site.coords
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                    bond_angles.append(angle)
    typical_bond_length = 2.0
    typical_bond_angle = 109.5
    if bond_lengths and bond_angles:
        length_dev = np.mean([abs(l - typical_bond_length) for l in bond_lengths])
        angle_dev = np.mean([abs(a - typical_bond_angle) for a in bond_angles])
        return float(length_dev + 0.1 * angle_dev)
    else:
        return 0.0

def rule_08_oxidation_state_consistency(structure):
    comp = structure.composition
    try:
        oxi_states = BVAnalyzer().get_oxi_state_decorated_structure(structure)
        total_charge = sum(site.specie.oxi_state for site in oxi_states.sites)
        return abs(float(total_charge))
    except:
        try:
            total_charge = 0.0
            for el, amt in comp.items():
                common_oxis = Element(el).oxidation_states
                if common_oxis:
                    oxi = max(set(common_oxis), key=common_oxis.count)
                    total_charge += oxi * amt
            return abs(float(total_charge))
        except:
            return 0.0

def rule_09_lattice_strain(structure):
    a, b, c = structure.lattice.abc
    mean_abc = np.mean([a, b, c])
    strain = np.sqrt(((a - mean_abc) ** 2 + (b - mean_abc) ** 2 + (c - mean_abc) ** 2) / 3) / mean_abc
    return float(strain)

def rule_10_prototype_similarity_score(structure):
    """
    Fast, continuous prototype similarity score.
    Only compares to a small sample of database-known structures for speed.
    """
    elements = tuple(sorted(el.symbol for el in structure.composition.elements))
    db_structs = _prototype_structures(elements)

    if not db_structs:
        return 0.0

    matcher = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5)
    min_dist = np.inf
    for db_struct in db_structs:
        try:
            if len(structure) == len(db_struct):
                dists = [
                    np.linalg.norm(pbc_shortest_vectors(
                        structure.lattice, [structure.frac_coords[i]], [db_struct.frac_coords[i]])[0][0])
                    for i in range(len(structure))
                ]
                mean_dist = np.mean(dists)
                if mean_dist < min_dist:
                    min_dist = mean_dist
        except Exception:
            continue

    if np.isfinite(min_dist):
        return float(np.exp(-min_dist))
    else:
        return 0.0


def rule_11_elemental_abundance_score(structure):
    clarke_abundance = {
        "O": 4.6e5, "Si": 2.8e5, "Al": 8.2e4, "Fe": 5.6e4, "Ca": 4.1e4, "Na": 2.3e4, "K": 2.1e4, "Mg": 2.3e4,
        "Ti": 5.6e3, "H": 1.4e4, "P": 1.1e3, "Mn": 1e3, "F": 5.8e2, "Ba": 4.3e2, "S": 3.5e2, "C": 2e2, "Sr": 3.7e2,
        "Zr": 1.6e2, "V": 1.2e2, "Cr": 1.0e2, "Ni": 8.4e1, "Cu": 6.0e1, "Zn": 7.9e1, "Rb": 9.0e0, "Pb": 1.4e1,
        "Li": 2.0e1, "B": 1.0e1, "Sn": 2.2e1, "Co": 2.5e1, "Ce": 6.6e1, "Y": 2.8e1, "Nb": 2.0e1, "La": 3.9e1,
        "Nd": 2.0e1, "U": 2.7e0, "Th": 9.6e0, "Ag": 7.5e-1, "Cd": 1.1e0, "In": 2.5e-1, "Ga": 1.9e0, "Ge": 1.5e0,
        "Mo": 1.2e0, "W": 1.3e0, "Sb": 2.0e0, "Bi": 1.2e0, "Au": 4.0e-2, "Hg": 8.5e-1, "Pt": 3.7e-2, "Ta": 2.0e-1,
        "Re": 4.0e-2, "Os": 5.0e-2, "Ir": 1.0e-2, "Rh": 7.0e-2, "Pd": 1.5e-1, "Sc": 2.2e1, "Tl": 6.0e-1, "Te": 1.0e0,
        "Se": 5.0e-1, "As": 1.8e1, "Br": 2.3e1, "Cl": 1.3e2, "I": 4.5e0, "Cs": 3.0e0
    }
    comp = structure.composition.fractional_composition
    scores = []
    for el, amt in comp.items():
        ab = clarke_abundance.get(el.symbol, 1e-3)
        scores.append(np.log10(ab) * amt)
    return float(np.sum(scores))

def rule_12_volume_per_atom(structure):
    return structure.volume / structure.num_sites

def rule_13_charge_neutrality_deviation(structure):
    comp = structure.composition
    try:
        oxi_struct = BVAnalyzer().get_oxi_state_decorated_structure(structure)
        total_charge = sum(site.specie.oxi_state for site in oxi_struct.sites)
        return abs(float(total_charge))
    except Exception:
        # Estimate net charge from tabulated common oxidation states.
        oxi_table = {"O": -2, "N": -3, "F": -1, "Cl": -1, "Br": -1, "I": -1, "S": -2, "Se": -2,
                     "Li": 1, "Na": 1, "K": 1, "Mg": 2, "Ca": 2, "Al": 3, "Si": 4, "P": 5, "Fe": 3, "Cu": 2, "Zn": 2}
        charge = 0.0
        for el, amt in comp.items():
            charge += oxi_table.get(el.symbol, 0) * amt
        return abs(float(charge))


def rule_14_bond_valence_sum_deviation(structure):
    try:
        bv_struct = BVAnalyzer().get_oxi_state_decorated_structure(structure)
        deviations = []
        for site in bv_struct.sites:
            expected = site.specie.oxi_state
            try:
                actual = BVAnalyzer().get_bv_sum(site, bv_struct)
                deviations.append(abs(actual - expected))
            except Exception:
                continue
        return float(np.mean(deviations)) if deviations else 0.0
    except Exception:
        return 0.0


def rule_15_max_bond_angle_strain(structure):
    cnn = CrystalNN()
    angle_strains = []
    for i, site in enumerate(structure.sites):
        try:
            neighbors = cnn.get_nn_info(structure, i)
            if len(neighbors) < 2:
                continue
            neighbor_coords = [n['site'].coords for n in neighbors]
            center = site.coords
            for j in range(len(neighbor_coords)):
                for k in range(j + 1, len(neighbor_coords)):
                    v1 = neighbor_coords[j] - center
                    v2 = neighbor_coords[k] - center
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    ideal_angles = [90, 109.5, 120, 180]
                    min_strain = min(abs(angle - ia) for ia in ideal_angles)
                    angle_strains.append(min_strain)
        except:
            continue
    return float(max(angle_strains)) if angle_strains else 0.0

def rule_16_mechanical_stability_proxy(structure):
    comp = structure.composition
    atomic_numbers = [el.Z for el in comp.elements]
    avg_atomic_number = sum(atomic_numbers[i] * comp[el] for i, el in enumerate(comp.elements)) / comp.num_atoms
    density = structure.density
    return float(density / avg_atomic_number)

def rule_17_elemental_toxicity_volatility(structure):
    toxic_elements = {"Hg", "Cd", "Pb", "As", "Tl", "Se", "Sb", "Be", "Cr", "F", "Cl", "Br", "I", "U", "Th", "Ra"}
    comp = structure.composition.get_el_amt_dict()
    total = sum(comp.values())
    toxic_amt = sum(amt for el, amt in comp.items() if el in toxic_elements)
    return toxic_amt / total if total > 0 else 0.0

def rule_18_synthesis_temperature_estimate(structure):
    mp_melting_points = {
        "Li": 453.65, "Na": 370.87, "K": 336.53, "Rb": 312.46, "Cs": 301.59,
        "Mg": 923, "Ca": 1115, "Sr": 1050, "Ba": 1000,
        "Al": 933.47, "Ga": 302.91, "In": 429.75, "Tl": 577,
        "Si": 1687, "Ge": 1211.4, "Sn": 505.08, "Pb": 600.61,
        "B": 2349, "C": 3823, "N": 63.15, "O": 54.36, "F": 53.53, "P": 317.3, "S": 388.36, "Cl": 171.6, "Br": 265.8, "I": 386.85,
        "Ti": 1941, "V": 2183, "Cr": 2180, "Mn": 1519, "Fe": 1811, "Co": 1768, "Ni": 1728, "Cu": 1357.77, "Zn": 692.68,
        "Zr": 2128, "Nb": 2750, "Mo": 2896, "Ru": 2607, "Rh": 2237, "Pd": 1828, "Ag": 1234.93, "Cd": 594.22, "Hf": 2506, "Ta": 3290, "W": 3695, "Re": 3459, "Os": 3306, "Ir": 2719, "Pt": 2041.4, "Au": 1337.33, "Hg": 234.32,
        "La": 1193, "Ce": 1068, "Pr": 1208, "Nd": 1297, "Sm": 1345, "Eu": 1095, "Gd": 1585, "Tb": 1629, "Dy": 1680, "Ho": 1747, "Er": 1802, "Tm": 1818, "Yb": 1092, "Lu": 1936,
        "Y": 1799, "Sc": 1814,
        "U": 1405.3, "Th": 2023, "Pu": 913, "Am": 1449
    }
    comp = structure.composition
    mp_list = []
    for el, amt in comp.get_el_amt_dict().items():
        mp = mp_melting_points.get(el, None)
        if mp:
            mp_list.extend([mp] * int(amt))
    if not mp_list:
        return 0.0
    return float(np.mean(mp_list))

def rule_19_phase_stability_window(structure, mp_id=None):
    try:
        if mp_id is None:
            return 0.0
        doc_data = _summary_doc(mp_id)
        eah = doc_data.get("energy_above_hull")
        if eah is not None:
            if eah < 0.1:
                return 0.0
            elif eah < 0.2:
                return 0.5
            else:
                return 1.0
    except Exception as e:
        print(f"[rule_19] API fail: {e}")
    return 0.0


def rule_20_atomic_radii_mismatch(structure):
    comp = structure.composition.get_el_amt_dict()
    radii = []
    for el, amt in comp.items():
        try:
            r = Element(el).atomic_radius
            if r is not None:
                radii.extend([r] * int(amt))
        except:
            continue
    if not radii or len(radii) < 2:
        return 0.0
    return float(np.std(radii) / np.mean(radii))

def rule_21_atoms_per_unit_cell(structure):
    return float(len(structure.sites))

def rule_22_electronegativity_difference(structure):
    comp = structure.composition.get_el_amt_dict()
    en_values = [Element(el).X for el in comp if Element(el).X is not None]
    if not en_values:
        return 0.0
    return float(max(en_values) - min(en_values))

def rule_23_synthesis_analogue_count(structure):
    chemsys = "-".join(sorted([el.symbol for el in structure.composition.elements]))
    return _chemsys_count(chemsys)


def rule_24_fraction_fully_occupied_sites(structure):
    total_sites = len(structure.sites)
    fully_occupied = sum(1 for site in structure.sites if all(oc == 1.0 for _, oc in site.species.items()))
    return float(fully_occupied) / total_sites if total_sites > 0 else 0.0

def rule_25_void_fraction(structure):
    try:
        total_volume = structure.lattice.volume
        atom_volumes = 0.0
        for site in structure:
            r = Element(site.specie.symbol).atomic_radius
            if r:
                atom_volumes += (4/3) * np.pi * r**3
        void_fraction = 1.0 - (atom_volumes / total_volume)
        return float(max(0.0, void_fraction))
    except:
        return 0.0

def rule_26_polyhedral_connectivity(structure):
    cnn = CrystalNN()
    poly_connectivities = []
    for i, site in enumerate(structure.sites):
        try:
            el = Element(site.specie.symbol)
        except:
            continue
        if el.is_transition_metal or el.is_lanthanoid or el.is_actinoid or el.symbol in ["Al", "Si", "P", "B"]:
            try:
                neighbors = cnn.get_nn_info(structure, i)
                poly_connectivities.append(len(neighbors))
            except:
                continue
    return float(np.mean(poly_connectivities)) if poly_connectivities else 0.0


def rule_27_absence_of_mixed_valence(structure):
    comp = structure.composition
    elements = comp.elements
    analyzer = BVAnalyzer()
    try:
        valences = analyzer.get_valences(structure)
    except:
        return 0.0
    valence_dict = {}
    for el, val in zip(structure.species, valences):
        sym = el.symbol
        if sym not in valence_dict:
            valence_dict[sym] = set()
        valence_dict[sym].add(round(val, 2))
    mixed_count = sum(1 for vset in valence_dict.values() if len(vset) > 1)
    return 1.0 - mixed_count / len(valence_dict) if valence_dict else 1.0

def rule_28_min_interatomic_distance(structure):
    try:
        dists = structure.distance_matrix
        dists[dists == 0] = np.inf
        return float(np.min(dists))
    except:
        return 0.0

def rule_29_absence_unstable_motifs(structure):
    cn = CrystalNN()
    highly_unstable_motifs = 0
    for site in structure.sites:
        # Use the site's Element or Species object directly.
        el = site.specie
        if Element(el.symbol).is_metal and el.symbol not in [
            "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Al", "Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta",
            "Cr", "Mo", "W", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "In", "Sn", "Pb", "Sb", "Bi"
        ]:
            highly_unstable_motifs += 1
        try:
            cn_num = cn.get_cn(structure, site)
            if el.symbol == "O" and cn_num < 2:
                highly_unstable_motifs += 1
            if el.symbol == "N" and cn_num < 2:
                highly_unstable_motifs += 1
            if el.symbol == "F" and cn_num < 1:
                highly_unstable_motifs += 1
            if el.symbol == "Cl" and cn_num < 1:
                highly_unstable_motifs += 1
        except:
            continue
    return float(highly_unstable_motifs) / len(structure.sites) if structure.sites else 0.0


def rule_30_elastic_anisotropy(structure):
    a, b, c = structure.lattice.abc
    tol = 1e-2
    if abs(a - b) < tol and abs(b - c) < tol:
        vnn = VoronoiNN()
        bond_lengths = []
        bond_angles = []
        for i, site in enumerate(structure):
            neighs = vnn.get_nn_info(structure, i)
            for n in neighs:
                bond_lengths.append(n['weight'])
            coords = [n['site'].coords for n in neighs]
            center = site.coords
            for j in range(len(coords)):
                for k in range(j+1, len(coords)):
                    v1 = coords[j] - center
                    v2 = coords[k] - center
                    norm1 = np.linalg.norm(v1)
                    norm2 = np.linalg.norm(v2)
                    if norm1 > 1e-3 and norm2 > 1e-3:
                        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        bond_angles.append(np.degrees(angle))
        if bond_lengths and bond_angles:
            c11 = 1.0 / np.mean(bond_lengths)
            c44 = 1.0 / np.std(bond_angles)
            c12 = c11 * 0.5
            if c11 - c12 != 0:
                zener = 2 * c44 / (c11 - c12)
                return float(zener)
    return float(max(a, b, c) / min(a, b, c))

def rule_31_defect_tolerance_indicator(structure):
    vnn = VoronoiNN()
    site_fractions = []
    for i, site in enumerate(structure.sites):
        try:
            nns = vnn.get_nn_info(structure, i)
            site_fractions.append(len(nns))
        except:
            continue
    if not site_fractions:
        return 0.0
    mean_cn = np.mean(site_fractions)
    std_cn = np.std(site_fractions)
    return float(std_cn / mean_cn) if mean_cn != 0 else 0.0

def rule_32_max_coordination_number(structure):
    cnn = CrystalNN()
    try:
        coord_numbers = [cnn.get_cn(structure, i) for i in range(len(structure))]
        return float(max(coord_numbers))
    except:
        return 0.0

def rule_33_unstable_magnetic_electronic_states_score(structure):
    comp = structure.composition.get_el_amt_dict()
    def is_unstable_magnetic_element(el):
        e = Element(el)
        if e.is_transition_metal or e.is_lanthanoid or e.is_actinoid:
            nval = e.full_electronic_structure
            for orb in nval:
                if len(orb) == 4:
                    if orb[1] == 'd' and 0 < orb[3] < 10:
                        return True
                    if orb[1] == 'f' and 0 < orb[3] < 14:
                        return True
        return False
    unstable_count = sum(amt for el, amt in comp.items() if is_unstable_magnetic_element(el))
    total = sum(comp.values())
    return float(unstable_count) / total if total > 0 else 0.0

def rule_34_absence_unstable_volatile_elements(structure):
    unstable_volatile_elements = {
        "He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og",
        "Tc", "Pm", "Po", "At", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Fl", "Lv", "Ts",
        "Hg", "Cd", "Se", "Br", "I"
    }
    comp = structure.composition
    total = comp.num_atoms
    count = sum(comp[el] for el in comp.elements if el.symbol in unstable_volatile_elements)
    return 1.0 - (count / total) if total > 0 else 1.0

def rule_35_lattice_mismatch_min(structure):
    known_lattices = [
        (3.905, 3.905, 3.905),
        (3.868, 3.868, 3.868),
        (4.213, 4.213, 4.213),
        (5.431, 5.431, 5.431),
        (3.615, 3.615, 3.615),
        (4.758, 4.758, 12.991),
    ]
    a, b, c = sorted(structure.lattice.abc)
    min_mismatch = 1.0
    for klat in known_lattices:
        ka, kb, kc = sorted(klat)
        mismatches = [
            abs(a - ka) / ka,
            abs(b - kb) / kb,
            abs(c - kc) / kc
        ]
        mismatch = min(mismatches)
        if mismatch < min_mismatch:
            min_mismatch = mismatch
    return float(min_mismatch)

def rule_36_polyhedral_distortion_index(structure):
    cnn = CrystalNN()
    distortions = []
    for i, site in enumerate(structure.sites):
        try:
            nns = cnn.get_nn_info(structure, i)
            if not nns:
                continue
            neighbor_coords = [nn['site'].coords for nn in nns]
            center = site.coords
            distances = [np.linalg.norm(center - nc) for nc in neighbor_coords]
            mean_dist = np.mean(distances)
            if mean_dist == 0:
                continue
            distortion = np.mean([abs(d - mean_dist) / mean_dist for d in distances])
            distortions.append(distortion)
        except:
            continue
    return float(np.mean(distortions)) if distortions else 0.0

def rule_37_dimensional_anisotropy(structure):
    lengths = np.array(structure.lattice.abc)
    max_len = np.max(lengths)
    min_len = np.min(lengths)
    return float(max_len / min_len) if min_len > 0 else 0.0

def rule_38_min_site_occupancy(structure):
    try:
        occs = [max([frac for _, frac in site.species.items()]) for site in structure.sites]
        return float(min(occs)) if occs else 0.0
    except:
        return 0.0


def rule_39_fraction_metallic_bonding(structure):
    metallic_count = 0
    total_bonds = 0
    neighbors = structure.get_all_neighbors(r=3.0)
    for i, site in enumerate(structure.sites):
        el1 = Element(site.specie.symbol)
        for neighbor in neighbors[i]:
            # Handle both neighbor wrappers and site objects.
            try:
                el2 = Element(neighbor.site.specie.symbol)
            except AttributeError:
                el2 = Element(neighbor.specie.symbol)
            if el1.is_metal and el2.is_metal:
                metallic_count += 1
            total_bonds += 1
    return metallic_count / total_bonds if total_bonds > 0 else 0.0



def rule_40_interlayer_distance_std(structure):
    coords = np.array([site.frac_coords for site in structure.sites])
    c_axis = structure.lattice.matrix[2] / np.linalg.norm(structure.lattice.matrix[2])
    vnn = VoronoiNN()
    interlayer_distances = []
    for i, site in enumerate(structure.sites):
        neighbors = vnn.get_nn_info(structure, i)
        for n in neighbors:
            j = n['site_index']
            dz = np.dot(coords[j] - coords[i], c_axis)
            dz = dz - np.round(dz)
            abs_dz = abs(dz * np.linalg.norm(structure.lattice.c))
            if abs_dz > 2.0:
                interlayer_distances.append(abs_dz)
    return float(np.std(interlayer_distances)) if interlayer_distances else 0.0

def rule_41_framework_density(structure):
    v = structure.lattice.volume
    n_atoms = len(structure.sites)
    return n_atoms / v if v > 0 else 0.0

def rule_42_stoichiometry_deviation(structure):
    comp = structure.composition
    reduced = comp.reduced_composition
    ratios = [amt / min(reduced.values()) for amt in reduced.values()]
    deviation = np.std(ratios)
    return float(deviation)

def rule_43_charge_balance_deviation(structure):
    try:
        valences = BVAnalyzer().get_valences(structure)
        total_charge = sum(valences)
        n_sites = len(structure.sites)
        return abs(total_charge) / n_sites if n_sites > 0 else 0.0
    except Exception:
        # Estimate net charge from the composition if bond valences are unavailable.
        oxi_table = {"O": -2, "N": -3, "F": -1, "Cl": -1, "Br": -1, "I": -1, "S": -2, "Se": -2,
                     "Li": 1, "Na": 1, "K": 1, "Mg": 2, "Ca": 2, "Al": 3, "Si": 4, "P": 5, "Fe": 3, "Cu": 2, "Zn": 2}
        comp = structure.composition
        charge = 0.0
        for el, amt in comp.items():
            charge += oxi_table.get(el.symbol, 0) * amt
        n_sites = len(structure.sites)
        return abs(float(charge)) / n_sites if n_sites > 0 else 0.0


def rule_44_site_disorder_score(structure):
    disorder_scores = []
    for site in structure.sites:
        occs = list(site.species.values())
        if len(occs) == 1 and occs[0] == 1.0:
            disorder_scores.append(0.0)
        else:
            max_occ = max(occs)
            disorder_scores.append(1.0 - max_occ)
    return float(np.mean(disorder_scores)) if disorder_scores else 0.0


def rule_45_min_antisite_formation_energy(structure):
    elements = list(structure.composition.elements)
    cations = [el for el in elements if Element(el).is_metal]
    if len(cations) < 2:
        return 0.0
    en_diffs = []
    for i, el1 in enumerate(cations):
        for el2 in cations[i+1:]:
            en1 = Element(el1).X
            en2 = Element(el2).X
            en_diffs.append(abs(en1 - en2))
    if not en_diffs:
        return 0.0
    return float(np.min(en_diffs))

def rule_46_surface_energy_proxy(structure):
    vnn = VoronoiNN()
    undercoord_fractions = []
    for site in structure.sites:
        try:
            ideal_cn = site.specie.max_coordinated_neighbors
        except AttributeError:
            ideal_cn = None
        if ideal_cn is None:
            Z = site.specie.Z
            if Z <= 2:
                ideal_cn = 2
            elif Z <= 10:
                ideal_cn = 4
            elif Z <= 18:
                ideal_cn = 4
            elif Z <= 54:
                ideal_cn = 6
            else:
                ideal_cn = 6
        try:
            actual_cn = len(vnn.get_nn_info(structure, site.index))
        except:
            actual_cn = 0
        undercoord = max(0, ideal_cn - actual_cn) / ideal_cn
        undercoord_fractions.append(undercoord)
    return float(np.mean(undercoord_fractions)) if undercoord_fractions else 0.0

def rule_47_synthesis_complexity_proxy(structure):
    elements = [el.symbol for el in structure.composition.elements]
    n_elements = len(elements)
    rare_earths = set([
        "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"
    ])
    n_rare_earths = sum(1 for el in elements if el in rare_earths)
    n_metalloids = sum(1 for el in elements if Element(el).is_metalloid)
    try:
        vnn = VoronoiNN()
        n_unusual_coord = 0
        for site in structure:
            try:
                cn = vnn.get_cn(structure, site)
                if cn < 2 or cn > 8:
                    n_unusual_coord += 1
            except:
                continue
        frac_unusual_coord = n_unusual_coord / len(structure)
    except:
        frac_unusual_coord = 0.0
    return float(0.5 * n_elements + n_rare_earths + n_metalloids + 5 * frac_unusual_coord)

def rule_48_max_thermodynamic_stability_deviation(structure, mp_id=None, api_key=API_KEY):
    e_per_atom = None
    if mp_id is not None:
        doc_data = _summary_doc(mp_id)
        if doc_data and doc_data.get("formation_energy_per_atom") is not None:
            e_per_atom = float(doc_data["formation_energy_per_atom"])
        else:
            print(f"[WARN] 'formation_energy_per_atom' not found or is None for {mp_id}")
    else:
        comp = structure.composition.reduced_formula
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                formula=comp,
                fields=["material_id", "formation_energy_per_atom"]
            )
        if docs:
            doc_data = docs[0].model_dump() if hasattr(docs[0], "model_dump") else docs[0].dict()
            if "formation_energy_per_atom" in doc_data and doc_data["formation_energy_per_atom"] is not None:
                e_per_atom = float(doc_data["formation_energy_per_atom"])
            else:
                print(f"[WARN] 'formation_energy_per_atom' not found or is None for {comp}")
        else:
            print(f"[WARN] No summary found for {comp}")
    if e_per_atom is None:
        return 0.0

    lower, upper = -6.0, 1.0
    if e_per_atom < lower:
        return abs(e_per_atom - lower)
    elif e_per_atom > upper:
        return abs(e_per_atom - upper)
    else:
        return 1.0 / (1.0 + abs(e_per_atom))


def rule_49_min_migration_barrier(structure):
    from pymatgen.core.periodic_table import Element
    try:
        mobile_ions = [el for el in structure.composition.elements if Element(el).is_alkali or Element(el).is_alkaline]
        if not mobile_ions:
            return 0.0
        radii = [Element(el).ionic_radii.get(1, Element(el).atomic_radius) for el in mobile_ions]
        min_radius = min([r for r in radii if r is not None])
        density = structure.density if structure.density > 0 else 1.0
        return float(min_radius / density)
    except:
        return 0.0


def rule_50_env_sensitivity_proxy(structure):
    reactive_elements = {"Li", "Na", "K", "Rb", "Cs", "F", "Cl", "Br", "I", "O", "S", "Se", "Te"}
    comp = structure.composition.get_el_amt_dict()
    total = sum(comp.values())
    if total == 0:
        return 0.0
    reactive_amt = sum(amt for el, amt in comp.items() if el in reactive_elements)
    return 1.0 - (reactive_amt / total)
