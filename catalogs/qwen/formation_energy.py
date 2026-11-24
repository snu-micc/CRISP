import math
import itertools
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
from pymatgen.core import Structure, Composition, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN, VoronoiNN, MinimumDistanceNN
from pymatgen.analysis.bond_valence import BVAnalyzer


def rule_01(structure):

    def cov(sym):
        fixed = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.5, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69}
        if sym not in fixed: raise ValueError("No tabulated covalent radius: "+sym)
        return fixed[sym]

    def parts(s):
        comp=s.composition.element_composition
        return [(el,float(n)/comp.num_atoms) for el,n in comp.items()]

    s=structure
    if not len(s): return float("nan")
    r=[cov(e.symbol) for e,p in parts(s)]
    return max(r)/min(r)


def rule_02(structure):

    s=structure
    if not len(s): return float("nan")
    return len(s.composition.elements)


def rule_03(structure):
    def shell(s):

        n=len(s)
        if not n: raise ValueError("Empty structure")
        cutoff=max(2.0,1.5*(s.volume/n)**(1/3))
        for attempt in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            mins=np.full(n,np.inf)
            use=dd>1e-8
            np.minimum.at(mins,ii[use],dd[use])
            if np.all(1.2*mins+1e-7<=cutoff): break
            cutoff*=2
        else: raise ValueError("Could not enclose complete periodic first shells")
        out=[[] for i in range(n)]
        for i,j,image,d in zip(ii,jj,images,dd):
            if d>1e-8 and d<=1.2*mins[i]+1e-7:
                out[int(i)].append((int(j),tuple(map(int,image)),float(d)))
        return out

    elements = set(structure.composition.elements)
    chi = {}
    for el in elements:
        e = Element(el)
        x = e.X
        if x is None:
            return float('nan')
        try:
            x = float(x)
        except (TypeError, ValueError):
            return float('nan')
        if math.isnan(x):
            return float('nan')
        chi[el] = x

    try:
        cnn = CrystalNN()
        nn_info = cnn.get_all_nn_info(structure)
    except Exception:
        return float('nan')

    if not any(nn_info):
        nn_info = [[{'site_index':j} for j,t,d in row] for row in shell(structure)]

    bond_dchis = []
    seen_pairs = set()
    n_sites = len(structure)

    for i in range(n_sites):
        site_i = structure[i]
        for nn in nn_info[i]:
            j = nn['site_index']

            pair = (min(i, j), max(i, j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            el_a = site_i.specie
            el_b = structure[j].specie
            dchi = abs(chi[el_a] - chi[el_b])
            bond_dchis.append(dchi)

    if not bond_dchis:
        return float('nan')

    n_bonds = len(bond_dchis)
    mean_dchi = sum(bond_dchis) / n_bonds

    if n_bonds > 1:
        var_dchi = sum((d - mean_dchi) ** 2 for d in bond_dchis) / (n_bonds - 1)
        std_dchi = math.sqrt(var_dchi)
    else:
        std_dchi = 0.0

    chi_values = list(chi.values())
    chi_spread = max(chi_values) - min(chi_values)

    scalar = mean_dchi + 0.5 * chi_spread + std_dchi

    return float(scalar)


def rule_04(structure):
    def shell(s):

        n=len(s)
        if not n: raise ValueError("Empty structure")
        cutoff=max(2.0,1.5*(s.volume/n)**(1/3))
        for attempt in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            mins=np.full(n,np.inf)
            use=dd>1e-8
            np.minimum.at(mins,ii[use],dd[use])
            if np.all(1.2*mins+1e-7<=cutoff): break
            cutoff*=2
        else: raise ValueError("Could not enclose complete periodic first shells")
        out=[[] for i in range(n)]
        for i,j,image,d in zip(ii,jj,images,dd):
            if d>1e-8 and d<=1.2*mins[i]+1e-7:
                out[int(i)].append((int(j),tuple(map(int,image)),float(d)))
        return out

    def undirected_cnn(st):
        probe=CrystalNN()
        rows=probe.get_all_nn_info(st)
        geom=None; links={}
        for i,row in enumerate(rows):
            if row: neighbors=[(int(v["site_index"]),tuple(int(round(x)) for x in v["image"])) for v in row]
            else:
                if geom is None: geom=shell(st)
                neighbors=[(j,t) for j,t,d in geom[i]]
            for j,t in neighbors:
                k=min((i,j,t),(j,i,tuple(-x for x in t)))
                if k not in links:
                    links[k]=float(np.linalg.norm(st.lattice.get_cartesian_coords(st[j].frac_coords+np.asarray(t)-st[i].frac_coords)))
        out={i:[] for i in range(len(st))}
        for (i,j,t),d in sorted(links.items()):
            out[i].append({"site_index":j,"image":t,"distance_A":d})
            out[j].append({"site_index":i,"image":tuple(-x for x in t),"distance_A":d})
        return out

    def shannon(el,cn,ox):
        roman={3:"III",4:"IV",5:"V",6:"VI",7:"VII",8:"VIII",12:"XII"}
        opts=Element(el.symbol).data.get("Shannon radii",{}).get(str(int(ox)),{}).get(roman.get(cn),{})
        row=opts.get("")
        if row is None and len(opts)==1: row=list(opts.values())[0]
        return None if row is None else row.get("ionic_radius")
    _cov_table = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.5, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69}
    struct = structure

    try:
        cnn = CrystalNN()
        nn_info = undirected_cnn(struct)
    except Exception:
        return float('nan')

    if not nn_info:
        return float('nan')

    comp = struct.composition
    el_dic = comp.get_el_amt_dict()

    def _assign_ox(el_dic):
        if len(el_dic) == 1:
            return {list(el_dic.keys())[0]: 0}
        fixed, flexible = {}, {}
        for sym, cnt in el_dic.items():
            el = Element(sym)
            Z = el.Z
            if Z == 9:
                fixed[sym] = -1
            elif Z == 8:
                fixed[sym] = -2
            elif Z in (1, 11, 19, 37, 55, 87):
                fixed[sym] = 1
            elif Z in (4, 12, 20, 38, 56, 88):
                fixed[sym] = 2
            elif Z in (13, 31, 49, 81):
                fixed[sym] = 3
            elif Z in (35, 53, 85):
                fixed[sym] = -1
            else:
                flexible[sym] = cnt
        fixed_sum = sum(fixed[s] * el_dic[s] for s in fixed)
        if not flexible:
            return fixed
        if len(flexible) == 1:
            s = list(flexible.keys())[0]
            cnt = el_dic[s]
            return {**fixed, s: round(-fixed_sum / cnt)} if cnt > 0 else fixed

        common = {
            'H': 1, 'C': 4, 'N': -3, 'P': -3, 'As': -3, 'Sb': -3, 'Bi': -3,
            'S': -2, 'Se': -2, 'Te': -2, 'Po': -2,
            'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 3, 'Mn': 3, 'Fe': 3, 'Co': 3,
            'Ni': 2, 'Cu': 2, 'Zn': 2, 'Ga': 3, 'Ge': 4,
            'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 4,
            'Rh': 3, 'Pd': 2, 'Ag': 1, 'Cd': 2, 'In': 3, 'Sn': 4,
            'Hf': 4, 'Ta': 5, 'W': 6, 'Re': 7, 'Os': 4, 'Ir': 4,
            'Pt': 2, 'Au': 3, 'Hg': 2, 'Tl': 3, 'Pb': 2,
            'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Sm': 3, 'Eu': 2, 'Gd': 3,
            'Tb': 3, 'Dy': 3, 'Ho': 3, 'Er': 3, 'Tm': 3, 'Yb': 2, 'Lu': 3,
            'Ac': 3, 'Th': 4, 'Pa': 5, 'U': 6, 'Np': 5, 'Pu': 4,
        }
        result = dict(fixed)
        assigned = 0
        for s in flexible:
            ox = common.get(s, 0)
            result[s] = ox
            assigned += ox * el_dic[s]

        residual = -fixed_sum - assigned
        total_flex = sum(el_dic[s] for s in flexible)
        if total_flex > 0 and abs(residual) > 0.01:
            adj = residual / total_flex
            for s in flexible:
                result[s] = round(result[s] + adj)
        return result

    ox_states = _assign_ox(el_dic)

    coord_nums = {}
    for i, nn_list in nn_info.items():
        coord_nums[i] = len(nn_list)

    valid_shannon_coords = [3, 4, 5, 6, 7, 8, 12]

    def _snap_coord(c):
        if c in valid_shannon_coords:
            return c
        return min(valid_shannon_coords, key=lambda v: abs(v - c))

    deviations = []

    for i, nn_list in nn_info.items():
        site_i = struct[i]
        el_i = site_i.specie
        coord_i = _snap_coord(coord_nums[i])
        ox_i = ox_states.get(el_i.symbol, 0)

        for nn in nn_list:
            j = nn['site_index']
            if j < i or (j == i and tuple(nn['image']) <= (0,0,0)):
                continue
            site_j = struct[j]
            el_j = site_j.specie
            coord_j = _snap_coord(coord_nums[j])
            ox_j = ox_states.get(el_j.symbol, 0)
            d_ij = (float(nn['distance_A']) if 'distance_A' in nn else float(np.linalg.norm(nn['site'].coords - site_i.coords)))

            chi_diff = abs(el_i.X - el_j.X)
            if chi_diff < 0.5:
                r_i = _cov_table.get((el_i).symbol)
                r_j = _cov_table.get((el_j).symbol)
            else:
                try:
                    r_i = shannon(el_i, coord_i, ox_i)
                except Exception:
                    r_i = None
                try:
                    r_j = shannon(el_j, coord_j, ox_j)
                except Exception:
                    r_j = None
                if r_i is None:
                    r_i = _cov_table.get((el_i).symbol)
                if r_j is None:
                    r_j = _cov_table.get((el_j).symbol)

            if r_i is None or r_j is None:
                return float('nan')

            r_sum = r_i + r_j
            if r_sum <= 0.0:
                return float('nan')

            delta = (d_ij - r_sum) / r_sum
            deviations.append(delta)

    if not deviations:
        return float('nan')

    rms = math.sqrt(sum(d * d for d in deviations) / len(deviations))
    return float(rms)


def rule_05(structure):
    try:
        sga = SpacegroupAnalyzer(structure)
        ops = sga.get_symmetry_operations()
        return float(len(ops))
    except Exception:
        return float('nan')


def rule_06(structure):

    def shell(s):

        n=len(s)
        if not n: raise ValueError("Empty structure")
        cutoff=max(2.0,1.5*(s.volume/n)**(1/3))
        for attempt in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            mins=np.full(n,np.inf)
            use=dd>1e-8
            np.minimum.at(mins,ii[use],dd[use])
            if np.all(1.2*mins+1e-7<=cutoff): break
            cutoff*=2
        else: raise ValueError("Could not enclose complete periodic first shells")
        out=[[] for i in range(n)]
        for i,j,image,d in zip(ii,jj,images,dd):
            if d>1e-8 and d<=1.2*mins[i]+1e-7:
                out[int(i)].append((int(j),tuple(map(int,image)),float(d)))
        return out

    s=structure
    if not len(s): return float("nan")
    return np.mean([len(x) for x in shell(s)])


def rule_07(structure):

    def shell(s):

        n=len(s)
        if not n: raise ValueError("Empty structure")
        cutoff=max(2.0,1.5*(s.volume/n)**(1/3))
        for attempt in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            mins=np.full(n,np.inf)
            use=dd>1e-8
            np.minimum.at(mins,ii[use],dd[use])
            if np.all(1.2*mins+1e-7<=cutoff): break
            cutoff*=2
        else: raise ValueError("Could not enclose complete periodic first shells")
        out=[[] for i in range(n)]
        for i,j,image,d in zip(ii,jj,images,dd):
            if d>1e-8 and d<=1.2*mins[i]+1e-7:
                out[int(i)].append((int(j),tuple(map(int,image)),float(d)))
        return out

    def undirected_cnn(st):
        probe=CrystalNN()
        rows=probe.get_all_nn_info(st)
        geom=None; links={}
        for i,row in enumerate(rows):
            if row: neighbors=[(int(v["site_index"]),tuple(int(round(x)) for x in v["image"])) for v in row]
            else:
                if geom is None: geom=shell(st)
                neighbors=[(j,t) for j,t,d in geom[i]]
            for j,t in neighbors:
                k=min((i,j,t),(j,i,tuple(-x for x in t)))
                if k not in links:
                    links[k]=float(np.linalg.norm(st.lattice.get_cartesian_coords(st[j].frac_coords+np.asarray(t)-st[i].frac_coords)))
        out={i:[] for i in range(len(st))}
        for (i,j,t),d in sorted(links.items()):
            out[i].append({"site_index":j,"image":t,"distance_A":d})
            out[j].append({"site_index":i,"image":tuple(-x for x in t),"distance_A":d})
        return out

    struct = structure.copy()

    comp = struct.composition
    elements = list(comp.elements)

    chi = {}
    for el in elements:
        try:
            chi[el] = float(el.X)
        except (AttributeError, TypeError, ValueError):
            return float('nan')

    if len(elements) < 2:
        return 0.0

    try:
        cnn = CrystalNN()
    except Exception:
        return float('nan')

    en_diffs = []
    weights = []

    complete_nn = undirected_cnn(struct)
    n_sites = len(struct)
    for i in range(n_sites):
        try:
            neighbors = [(v['site_index'],v['distance_A']) for v in complete_nn[i]]
        except Exception:
            return float('nan')
        for j, dist in neighbors:
            if i < j:
                el_i = Element(struct[i].specie.symbol)
                el_j = Element(struct[j].specie.symbol)

                if el_i != el_j:
                    d_chi = abs(chi[el_i] - chi[el_j])
                    en_diffs.append(d_chi)

                    w = comp[el_i] * comp[el_j]
                    weights.append(w)

    if not en_diffs:

        return 0.0

    total_w = sum(weights)
    if total_w <= 0.0:
        return float('nan')

    result = sum(d * w for d, w in zip(en_diffs, weights)) / total_w
    return float(result)


def rule_08(structure):

    def symbols(s):

        return [max(site.species.items(),key=lambda x:(x[1],x[0].symbol))[0].symbol for site in s]

    def radial_shell_ratio(s):

        species=np.asarray(symbols(s));mono=len(set(species))==1
        cutoff=max(3.,2.1*(s.volume/len(s))**(1/3))
        for trial in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            ratios=[]
            for i in range(len(s)):
                keep=(ii==i)&(dd>1e-8)
                if not mono: keep=keep&(species[jj]!=species[i])
                ds=np.sort(dd[keep])
                if not len(ds): break
                first=ds[ds<=1.20*ds[0]+1e-7];outer=ds[ds>1.20*ds[0]+1e-7]
                if not len(outer) or cutoff<1.05*outer[0]+1e-7: break
                second=outer[outer<=1.05*outer[0]+1e-7]
                ratios.append(np.mean(second)/np.mean(first))
            if len(ratios)==len(s): return np.mean(ratios)
            cutoff*=2
        raise ValueError("Cannot resolve two periodic radial shells")

    s=structure
    if not len(s): return float("nan")
    return radial_shell_ratio(s)


def rule_09(structure):
    struct = structure.copy()

    if len(struct) == 0:
        return float('nan')

    try:
        sga = SpacegroupAnalyzer(struct)

        sym_dataset = sga.get_symmetry_dataset()
        pg_order = len(sym_dataset.rotations)
        if pg_order <= 0:
            return float('nan')

        labels = list(sym_dataset.wyckoffs)
        equiv = list(map(int,sym_dataset.equivalent_atoms))
        sizes = Counter(equiv)
        wyckoff_symbols = [str(sizes[j])+w for j,w in zip(equiv,labels)]
        if len(wyckoff_symbols) != len(struct):
            return float('nan')

        wyckoff_letters = set()
        multiplicities = []
        for ws in wyckoff_symbols:

            letter = ws[-1]
            mult_str = ws[:-1]
            if not mult_str.isdigit():
                return float('nan')
            wyckoff_letters.add(letter)
            multiplicities.append(int(mult_str))

        n_wyckoff = len(wyckoff_letters)

        max_mult = max(multiplicities)
        n_special = sum(1 for m in multiplicities if m < max_mult)
        n_total = len(multiplicities)
        f_special = n_special / n_total

        n_atoms = len(struct)
        ratio_sites = n_wyckoff / n_atoms

        comp = struct.composition
        n_elements = len(comp.elements)

        complexity = ratio_sites * (1.0 / pg_order) * (1.0 - f_special) * n_elements

        return float(complexity)

    except Exception:
        return float('nan')


def rule_10(structure):

    def shell(s):

        n=len(s)
        if not n: raise ValueError("Empty structure")
        cutoff=max(2.0,1.5*(s.volume/n)**(1/3))
        for attempt in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            mins=np.full(n,np.inf)
            use=dd>1e-8
            np.minimum.at(mins,ii[use],dd[use])
            if np.all(1.2*mins+1e-7<=cutoff): break
            cutoff*=2
        else: raise ValueError("Could not enclose complete periodic first shells")
        out=[[] for i in range(n)]
        for i,j,image,d in zip(ii,jj,images,dd):
            if d>1e-8 and d<=1.2*mins[i]+1e-7:
                out[int(i)].append((int(j),tuple(map(int,image)),float(d)))
        return out

    s=structure
    if not len(s): return float("nan")
    return np.var([len(x) for x in shell(s)])


def rule_11(structure):

    def shell(s):

        n=len(s)
        if not n: raise ValueError("Empty structure")
        cutoff=max(2.0,1.5*(s.volume/n)**(1/3))
        for attempt in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            mins=np.full(n,np.inf)
            use=dd>1e-8
            np.minimum.at(mins,ii[use],dd[use])
            if np.all(1.2*mins+1e-7<=cutoff): break
            cutoff*=2
        else: raise ValueError("Could not enclose complete periodic first shells")
        out=[[] for i in range(n)]
        for i,j,image,d in zip(ii,jj,images,dd):
            if d>1e-8 and d<=1.2*mins[i]+1e-7:
                out[int(i)].append((int(j),tuple(map(int,image)),float(d)))
        return out

    def symbols(s):

        return [max(site.species.items(),key=lambda x:(x[1],x[0].symbol))[0].symbol for site in s]

    def cov(sym):
        fixed = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.5, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69}
        if sym not in fixed: raise ValueError("No tabulated covalent radius: "+sym)
        return fixed[sym]

    def shellratios(s):
        rad=[cov(x) for x in symbols(s)]
        return [d/(rad[i]+rad[j]) for i,row in enumerate(shell(s)) for j,t,d in row]

    s=structure
    if not len(s): return float("nan")
    return np.mean(shellratios(s))


def rule_12(structure):

    def ve(el):
        cfg=el.full_electronic_structure
        n=int(el.row)

        return sum(k for m,orb,k in cfg if (m==n and orb in ("s","p")) or
                   (el.is_transition_metal and m==n-1 and orb=="d"))

    def parts(s):
        comp=s.composition.element_composition
        return [(el,float(n)/comp.num_atoms) for el,n in comp.items()]

    s=structure
    if not len(s): return float("nan")
    x=[(ve(e),p) for e,p in parts(s)]
    avg=sum(v*p for v,p in x)
    return math.sqrt(sum(p*(v-avg)**2 for v,p in x))


def rule_13(structure):

    def vals(sym,mono):
        if mono: return (0,)
        e=Element(sym)
        z=e.common_oxidation_states or e.oxidation_states
        if not z: raise ValueError("No tabulated nominal charge: "+sym)
        return tuple(int(q) for q in z)

    def spectrum(s):

        amounts=s.composition.get_el_amt_dict()
        result={0.:1}
        for sym,n in sorted(amounts.items()):
            nxt=defaultdict(int)
            for prev,mult in result.items():
                for z in vals(sym,len(amounts)==1):
                    nxt[round(prev+n*z,8)]+=mult
            result=nxt
        return result

    s=structure
    if not len(s): return float("nan")
    comp,factor=s.composition.element_composition.get_reduced_composition_and_factor()
    return min(abs(z) for z in spectrum(s))/factor


def rule_14(structure):

    def vals(sym,mono):
        if mono: return (0,)
        e=Element(sym)
        z=e.common_oxidation_states or e.oxidation_states
        if not z: raise ValueError("No tabulated nominal charge: "+sym)
        return tuple(int(q) for q in z)

    def nearest_states(s,b):
        mono=len(s.composition.elements)==1
        ref=[]
        for site,obs in zip(s,b):
            z=0.
            for el,occ in site.species.items():
                q=min(vals(el.symbol,mono),key=lambda x:(abs(obs-x),abs(x),x))
                z+=occ*q
            ref.append(z/sum(site.species.values()))
        return np.asarray(ref,dtype=float)

    def electr(sym):
        x=float(Element(sym).X)
        if not np.isfinite(x): raise ValueError("No Pauling electronegativity: "+sym)
        return x

    def nominal_assignment(s):

        am=s.composition.element_composition.get_el_amt_dict()
        sy=sorted(am)
        if len(sy)==1: return {sy[0]:0}
        xs={k:electr(k) for k in sy}; mean=sum(am[k]*xs[k] for k in sy)/sum(am.values())
        states={0.:(0.,0.,0.,())}
        for k in sy:
            nxt={}
            common=vals(k,False)
            available=tuple(sorted(set(Element(k).oxidation_states)|set(common)))
            for net,(p,u,a,qs) in states.items():
                for q in available:
                    key=round(net+am[k]*q,8)
                    score=(p+am[k]*max(0.,q*(xs[k]-mean)),u+(0. if q in common else am[k]),a+am[k]*abs(q),qs+(q,))
                    if key not in nxt or score<nxt[key]: nxt[key]=score
            states=nxt
        net=min(states,key=lambda k:(abs(k),states[k]))
        return dict(zip(sy,states[net][3]))

    def topological_valence(s):

        assignment=nominal_assignment(s)
        occ=np.asarray([site.species.num_atoms for site in s])
        q=np.asarray([sum(v*assignment[e.symbol] for e,v in site.species.items())/o for site,o in zip(s,occ)])
        incoming=np.zeros(len(s))
        if not (np.any(q>0) and np.any(q<0)): return incoming
        distances=s.distance_matrix
        minima=np.array([np.min(distances[i,np.where(q*qi<0)[0]]) if qi!=0 else 0. for i,qi in enumerate(q)])
        cutoff=1.20*float(max(minima))+1e-6
        ii,jj,images,dd=s.get_neighbor_list(cutoff)
        rows=[[] for site in s]
        for i,j,d in zip(ii,jj,dd):
            if q[i]*q[j]<0 and 1e-8<d<=1.20*minima[i]+1e-7: rows[int(i)].append(int(j))
        for i,row in enumerate(rows):
            if not row:
                if q[i]!=0: raise ValueError("Nominal ionic site has no opposite neighbor")
                continue
            den=sum(occ[j] for j in row)
            for j in row: incoming[j]-=q[i]*occ[i]/den
        return incoming

    def bvs(st):

        params = {'Ag': [1.12, 1.42], 'Al': [1.13, 1.47], 'As': [1.21, 2.23], 'B': [0.79, 1.6], 'Ba': [1.88, 0.97], 'Be': [0.81, 1.47], 'Bi': [1.54, 1.67], 'Br': [1.13, 2.58], 'C': [0.78, 2.0], 'Ca': [1.5, 1.04], 'Cd': [1.28, 1.46], 'Ce': [1.68, 1.08], 'Cl': [0.99, 2.74], 'Co': [1.09, 1.7], 'Cr': [1.16, 1.56], 'Cs': [2.05, 0.86], 'Cu': [0.87, 1.75], 'Dy': [1.54, 1.1], 'Er': [1.51, 1.11], 'Eu': [1.62, 1.01], 'F': [0.58, 3.98], 'Fe': [1.16, 1.64], 'Ga': [1.14, 1.82], 'Gd': [1.58, 1.11], 'Ge': [1.21, 1.51], 'H': [0.38, 0.89], 'Hf': [1.42, 1.23], 'Hg': [1.32, 1.44], 'Ho': [1.53, 1.1], 'I': [1.33, 2.38], 'In': [1.34, 1.49], 'Ir': [1.37, 1.55], 'K': [1.73, 0.91], 'La': [1.71, 1.08], 'Li': [1.0, 0.97], 'Lu': [1.47, 1.14], 'Mg': [1.21, 1.23], 'Mn': [1.17, 1.6], 'Mo': [1.37, 1.3], 'N': [0.72, 2.61], 'Na': [1.36, 1.01], 'Nb': [1.4, 1.23], 'Nd': [1.64, 1.07], 'Ni': [1.04, 1.75], 'O': [0.63, 3.15], 'P': [1.09, 1.96], 'Pb': [1.53, 1.55], 'Pd': [1.11, 1.35], 'Pr': [1.66, 1.07], 'Rb': [1.84, 0.89], 'Re': [1.37, 1.46], 'Rh': [1.18, 1.54], 'Ru': [1.21, 1.42], 'S': [1.03, 2.35], 'Sb': [1.41, 1.72], 'Sc': [1.34, 1.2], 'Se': [1.18, 2.51], 'Si': [1.12, 1.58], 'Sm': [1.61, 1.07], 'Sn': [1.37, 1.72], 'Sr': [1.66, 0.99], 'Ta': [1.39, 1.33], 'Tb': [1.56, 1.1], 'Te': [1.4, 2.72], 'Th': [1.7, 1.11], 'Ti': [1.27, 1.32], 'Tl': [1.62, 1.44], 'Tm': [1.5, 1.11], 'U': [1.59, 1.22], 'V': [1.21, 1.45], 'W': [1.38, 1.4], 'Y': [1.52, 1.11], 'Yb': [1.49, 1.06], 'Zn': [1.07, 1.66], 'Zr': [1.43, 1.22]}
        en = ('As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Sb', 'Se', 'Si', 'Te')
        if any(el.symbol not in params for el in st.composition.element_composition.elements):
            return topological_valence(st)
        totals = []
        for centre in st:
            near = st.get_neighbors(centre, 4)
            total = 0.0
            for sp1, occ1 in centre.species.items():
                a = sp1.symbol
                for nb in near:
                    for sp2, occ2 in nb.species.items():
                        b = sp2.symbol
                        if a == b or (a not in en and b not in en):
                            continue
                        r1, c1 = params[a]
                        r2, c2 = params[b]
                        ref = r1 + r2 - r1*r2*(math.sqrt(c1)-math.sqrt(c2))**2/(c1*r1+c2*r2)
                        sign = 1 if Element(a).X < Element(b).X else -1
                        total += occ1*occ2*math.exp((ref - nb.nn_distance*1.015)/0.31)*sign
            totals.append(float(total))
        return totals

    s=structure
    if not len(s): return float("nan")
    b=bvs(s);z=nearest_states(s,b)
    return np.mean(np.abs(z))


def rule_15(structure):

    comp = structure.composition
    total_atoms = sum(comp.values())

    if total_atoms == 0:
        return float('nan')

    def _valence_electrons(el):

        Z = el.Z

        if 57 <= Z <= 71 or 89 <= Z <= 103:
            return 3.0
        group = el.group
        if group is None:
            return float('nan')

        if 1 <= group <= 2:
            return float(group)

        if 3 <= group <= 12:
            return float(group)

        if 13 <= group <= 18:
            return float(group - 10)
        return float('nan')

    total_valence = 0.0
    for el, count in comp.items():
        ve = _valence_electrons(el)
        if math.isnan(ve):
            return float('nan')
        total_valence += ve * count

    return total_valence / float(total_atoms)


def rule_16(structure):

    def shell(s):

        n=len(s)
        if not n: raise ValueError("Empty structure")
        cutoff=max(2.0,1.5*(s.volume/n)**(1/3))
        for attempt in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            mins=np.full(n,np.inf)
            use=dd>1e-8
            np.minimum.at(mins,ii[use],dd[use])
            if np.all(1.2*mins+1e-7<=cutoff): break
            cutoff*=2
        else: raise ValueError("Could not enclose complete periodic first shells")
        out=[[] for i in range(n)]
        for i,j,image,d in zip(ii,jj,images,dd):
            if d>1e-8 and d<=1.2*mins[i]+1e-7:
                out[int(i)].append((int(j),tuple(map(int,image)),float(d)))
        return out

    s=structure
    if not len(s): return float("nan")
    vals_site=[]
    for i,row in enumerate(shell(s)):
        unit=np.array([s.lattice.get_cartesian_coords(s[j].frac_coords+np.asarray(t)-s[i].frac_coords)/d for j,t,d in row])
        v=np.mean(unit,axis=0)
        vals_site.append(float(np.dot(v,v)))
    return np.mean(vals_site)


def rule_17(structure):

    def vals(sym,mono):
        if mono: return (0,)
        e=Element(sym)
        z=e.common_oxidation_states or e.oxidation_states
        if not z: raise ValueError("No tabulated nominal charge: "+sym)
        return tuple(int(q) for q in z)

    def nearest_states(s,b):
        mono=len(s.composition.elements)==1
        ref=[]
        for site,obs in zip(s,b):
            z=0.
            for el,occ in site.species.items():
                q=min(vals(el.symbol,mono),key=lambda x:(abs(obs-x),abs(x),x))
                z+=occ*q
            ref.append(z/sum(site.species.values()))
        return np.asarray(ref,dtype=float)

    def electr(sym):
        x=float(Element(sym).X)
        if not np.isfinite(x): raise ValueError("No Pauling electronegativity: "+sym)
        return x

    def nominal_assignment(s):

        am=s.composition.element_composition.get_el_amt_dict()
        sy=sorted(am)
        if len(sy)==1: return {sy[0]:0}
        xs={k:electr(k) for k in sy}; mean=sum(am[k]*xs[k] for k in sy)/sum(am.values())
        states={0.:(0.,0.,0.,())}
        for k in sy:
            nxt={}
            common=vals(k,False)
            available=tuple(sorted(set(Element(k).oxidation_states)|set(common)))
            for net,(p,u,a,qs) in states.items():
                for q in available:
                    key=round(net+am[k]*q,8)
                    score=(p+am[k]*max(0.,q*(xs[k]-mean)),u+(0. if q in common else am[k]),a+am[k]*abs(q),qs+(q,))
                    if key not in nxt or score<nxt[key]: nxt[key]=score
            states=nxt
        net=min(states,key=lambda k:(abs(k),states[k]))
        return dict(zip(sy,states[net][3]))

    def topological_valence(s):

        assignment=nominal_assignment(s)
        occ=np.asarray([site.species.num_atoms for site in s])
        q=np.asarray([sum(v*assignment[e.symbol] for e,v in site.species.items())/o for site,o in zip(s,occ)])
        incoming=np.zeros(len(s))
        if not (np.any(q>0) and np.any(q<0)): return incoming
        distances=s.distance_matrix
        minima=np.array([np.min(distances[i,np.where(q*qi<0)[0]]) if qi!=0 else 0. for i,qi in enumerate(q)])
        cutoff=1.20*float(max(minima))+1e-6
        ii,jj,images,dd=s.get_neighbor_list(cutoff)
        rows=[[] for site in s]
        for i,j,d in zip(ii,jj,dd):
            if q[i]*q[j]<0 and 1e-8<d<=1.20*minima[i]+1e-7: rows[int(i)].append(int(j))
        for i,row in enumerate(rows):
            if not row:
                if q[i]!=0: raise ValueError("Nominal ionic site has no opposite neighbor")
                continue
            den=sum(occ[j] for j in row)
            for j in row: incoming[j]-=q[i]*occ[i]/den
        return incoming

    def bvs(st):

        params = {'Ag': [1.12, 1.42], 'Al': [1.13, 1.47], 'As': [1.21, 2.23], 'B': [0.79, 1.6], 'Ba': [1.88, 0.97], 'Be': [0.81, 1.47], 'Bi': [1.54, 1.67], 'Br': [1.13, 2.58], 'C': [0.78, 2.0], 'Ca': [1.5, 1.04], 'Cd': [1.28, 1.46], 'Ce': [1.68, 1.08], 'Cl': [0.99, 2.74], 'Co': [1.09, 1.7], 'Cr': [1.16, 1.56], 'Cs': [2.05, 0.86], 'Cu': [0.87, 1.75], 'Dy': [1.54, 1.1], 'Er': [1.51, 1.11], 'Eu': [1.62, 1.01], 'F': [0.58, 3.98], 'Fe': [1.16, 1.64], 'Ga': [1.14, 1.82], 'Gd': [1.58, 1.11], 'Ge': [1.21, 1.51], 'H': [0.38, 0.89], 'Hf': [1.42, 1.23], 'Hg': [1.32, 1.44], 'Ho': [1.53, 1.1], 'I': [1.33, 2.38], 'In': [1.34, 1.49], 'Ir': [1.37, 1.55], 'K': [1.73, 0.91], 'La': [1.71, 1.08], 'Li': [1.0, 0.97], 'Lu': [1.47, 1.14], 'Mg': [1.21, 1.23], 'Mn': [1.17, 1.6], 'Mo': [1.37, 1.3], 'N': [0.72, 2.61], 'Na': [1.36, 1.01], 'Nb': [1.4, 1.23], 'Nd': [1.64, 1.07], 'Ni': [1.04, 1.75], 'O': [0.63, 3.15], 'P': [1.09, 1.96], 'Pb': [1.53, 1.55], 'Pd': [1.11, 1.35], 'Pr': [1.66, 1.07], 'Rb': [1.84, 0.89], 'Re': [1.37, 1.46], 'Rh': [1.18, 1.54], 'Ru': [1.21, 1.42], 'S': [1.03, 2.35], 'Sb': [1.41, 1.72], 'Sc': [1.34, 1.2], 'Se': [1.18, 2.51], 'Si': [1.12, 1.58], 'Sm': [1.61, 1.07], 'Sn': [1.37, 1.72], 'Sr': [1.66, 0.99], 'Ta': [1.39, 1.33], 'Tb': [1.56, 1.1], 'Te': [1.4, 2.72], 'Th': [1.7, 1.11], 'Ti': [1.27, 1.32], 'Tl': [1.62, 1.44], 'Tm': [1.5, 1.11], 'U': [1.59, 1.22], 'V': [1.21, 1.45], 'W': [1.38, 1.4], 'Y': [1.52, 1.11], 'Yb': [1.49, 1.06], 'Zn': [1.07, 1.66], 'Zr': [1.43, 1.22]}
        en = ('As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Sb', 'Se', 'Si', 'Te')
        if any(el.symbol not in params for el in st.composition.element_composition.elements):
            return topological_valence(st)
        totals = []
        for centre in st:
            near = st.get_neighbors(centre, 4)
            total = 0.0
            for sp1, occ1 in centre.species.items():
                a = sp1.symbol
                for nb in near:
                    for sp2, occ2 in nb.species.items():
                        b = sp2.symbol
                        if a == b or (a not in en and b not in en):
                            continue
                        r1, c1 = params[a]
                        r2, c2 = params[b]
                        ref = r1 + r2 - r1*r2*(math.sqrt(c1)-math.sqrt(c2))**2/(c1*r1+c2*r2)
                        sign = 1 if Element(a).X < Element(b).X else -1
                        total += occ1*occ2*math.exp((ref - nb.nn_distance*1.015)/0.31)*sign
            totals.append(float(total))
        return totals

    s=structure
    if not len(s): return float("nan")
    b=np.asarray(bvs(s));z=nearest_states(s,b)
    return np.sqrt(np.mean((b-z)**2))


def rule_18(structure):

    struct = structure.copy()

    volume = struct.lattice.volume
    if volume is None or volume <= 0.0:
        return float('nan')

    n_atoms = len(struct)
    if n_atoms == 0:
        return float('nan')

    elem_counts = Counter()
    for site in struct:
        elem_counts[site.specie.symbol] += 1

    total_atomic_volume = 0.0
    for elem_symbol, count in elem_counts.items():
        elem = Element(elem_symbol)
        r_cov = elem.atomic_radius
        if r_cov is None or r_cov <= 0.0:
            return float('nan')
        total_atomic_volume += count * (4.0 / 3.0) * math.pi * r_cov ** 3

    if total_atomic_volume <= 0.0:
        return float('nan')

    packing_ratio = volume / total_atomic_volume

    return float(packing_ratio)


def rule_19(structure):

    s=structure
    if not len(s): return float("nan")
    comp,factor=s.composition.element_composition.get_reduced_composition_and_factor()
    out=0.
    for el,n in comp.items():
        if not el.is_transition_metal: continue
        shells=el.full_electronic_structure
        states=[(m,k) for m,o,k in shells if o=="d"]
        if states: out+=n*max(states)[1]
    return out


def rule_20(structure):

    def vals(sym,mono):
        if mono: return (0,)
        e=Element(sym)
        z=e.common_oxidation_states or e.oxidation_states
        if not z: raise ValueError("No tabulated nominal charge: "+sym)
        return tuple(int(q) for q in z)

    def nearest_states(s,b):
        mono=len(s.composition.elements)==1
        ref=[]
        for site,obs in zip(s,b):
            z=0.
            for el,occ in site.species.items():
                q=min(vals(el.symbol,mono),key=lambda x:(abs(obs-x),abs(x),x))
                z+=occ*q
            ref.append(z/sum(site.species.values()))
        return np.asarray(ref,dtype=float)

    def electr(sym):
        x=float(Element(sym).X)
        if not np.isfinite(x): raise ValueError("No Pauling electronegativity: "+sym)
        return x

    def nominal_assignment(s):

        am=s.composition.element_composition.get_el_amt_dict()
        sy=sorted(am)
        if len(sy)==1: return {sy[0]:0}
        xs={k:electr(k) for k in sy}; mean=sum(am[k]*xs[k] for k in sy)/sum(am.values())
        states={0.:(0.,0.,0.,())}
        for k in sy:
            nxt={}
            common=vals(k,False)
            available=tuple(sorted(set(Element(k).oxidation_states)|set(common)))
            for net,(p,u,a,qs) in states.items():
                for q in available:
                    key=round(net+am[k]*q,8)
                    score=(p+am[k]*max(0.,q*(xs[k]-mean)),u+(0. if q in common else am[k]),a+am[k]*abs(q),qs+(q,))
                    if key not in nxt or score<nxt[key]: nxt[key]=score
            states=nxt
        net=min(states,key=lambda k:(abs(k),states[k]))
        return dict(zip(sy,states[net][3]))

    def topological_valence(s):

        assignment=nominal_assignment(s)
        occ=np.asarray([site.species.num_atoms for site in s])
        q=np.asarray([sum(v*assignment[e.symbol] for e,v in site.species.items())/o for site,o in zip(s,occ)])
        incoming=np.zeros(len(s))
        if not (np.any(q>0) and np.any(q<0)): return incoming
        distances=s.distance_matrix
        minima=np.array([np.min(distances[i,np.where(q*qi<0)[0]]) if qi!=0 else 0. for i,qi in enumerate(q)])
        cutoff=1.20*float(max(minima))+1e-6
        ii,jj,images,dd=s.get_neighbor_list(cutoff)
        rows=[[] for site in s]
        for i,j,d in zip(ii,jj,dd):
            if q[i]*q[j]<0 and 1e-8<d<=1.20*minima[i]+1e-7: rows[int(i)].append(int(j))
        for i,row in enumerate(rows):
            if not row:
                if q[i]!=0: raise ValueError("Nominal ionic site has no opposite neighbor")
                continue
            den=sum(occ[j] for j in row)
            for j in row: incoming[j]-=q[i]*occ[i]/den
        return incoming

    def bvs(st):

        params = {'Ag': [1.12, 1.42], 'Al': [1.13, 1.47], 'As': [1.21, 2.23], 'B': [0.79, 1.6], 'Ba': [1.88, 0.97], 'Be': [0.81, 1.47], 'Bi': [1.54, 1.67], 'Br': [1.13, 2.58], 'C': [0.78, 2.0], 'Ca': [1.5, 1.04], 'Cd': [1.28, 1.46], 'Ce': [1.68, 1.08], 'Cl': [0.99, 2.74], 'Co': [1.09, 1.7], 'Cr': [1.16, 1.56], 'Cs': [2.05, 0.86], 'Cu': [0.87, 1.75], 'Dy': [1.54, 1.1], 'Er': [1.51, 1.11], 'Eu': [1.62, 1.01], 'F': [0.58, 3.98], 'Fe': [1.16, 1.64], 'Ga': [1.14, 1.82], 'Gd': [1.58, 1.11], 'Ge': [1.21, 1.51], 'H': [0.38, 0.89], 'Hf': [1.42, 1.23], 'Hg': [1.32, 1.44], 'Ho': [1.53, 1.1], 'I': [1.33, 2.38], 'In': [1.34, 1.49], 'Ir': [1.37, 1.55], 'K': [1.73, 0.91], 'La': [1.71, 1.08], 'Li': [1.0, 0.97], 'Lu': [1.47, 1.14], 'Mg': [1.21, 1.23], 'Mn': [1.17, 1.6], 'Mo': [1.37, 1.3], 'N': [0.72, 2.61], 'Na': [1.36, 1.01], 'Nb': [1.4, 1.23], 'Nd': [1.64, 1.07], 'Ni': [1.04, 1.75], 'O': [0.63, 3.15], 'P': [1.09, 1.96], 'Pb': [1.53, 1.55], 'Pd': [1.11, 1.35], 'Pr': [1.66, 1.07], 'Rb': [1.84, 0.89], 'Re': [1.37, 1.46], 'Rh': [1.18, 1.54], 'Ru': [1.21, 1.42], 'S': [1.03, 2.35], 'Sb': [1.41, 1.72], 'Sc': [1.34, 1.2], 'Se': [1.18, 2.51], 'Si': [1.12, 1.58], 'Sm': [1.61, 1.07], 'Sn': [1.37, 1.72], 'Sr': [1.66, 0.99], 'Ta': [1.39, 1.33], 'Tb': [1.56, 1.1], 'Te': [1.4, 2.72], 'Th': [1.7, 1.11], 'Ti': [1.27, 1.32], 'Tl': [1.62, 1.44], 'Tm': [1.5, 1.11], 'U': [1.59, 1.22], 'V': [1.21, 1.45], 'W': [1.38, 1.4], 'Y': [1.52, 1.11], 'Yb': [1.49, 1.06], 'Zn': [1.07, 1.66], 'Zr': [1.43, 1.22]}
        en = ('As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Sb', 'Se', 'Si', 'Te')
        if any(el.symbol not in params for el in st.composition.element_composition.elements):
            return topological_valence(st)
        totals = []
        for centre in st:
            near = st.get_neighbors(centre, 4)
            total = 0.0
            for sp1, occ1 in centre.species.items():
                a = sp1.symbol
                for nb in near:
                    for sp2, occ2 in nb.species.items():
                        b = sp2.symbol
                        if a == b or (a not in en and b not in en):
                            continue
                        r1, c1 = params[a]
                        r2, c2 = params[b]
                        ref = r1 + r2 - r1*r2*(math.sqrt(c1)-math.sqrt(c2))**2/(c1*r1+c2*r2)
                        sign = 1 if Element(a).X < Element(b).X else -1
                        total += occ1*occ2*math.exp((ref - nb.nn_distance*1.015)/0.31)*sign
            totals.append(float(total))
        return totals

    s=structure
    if not len(s): return float("nan")
    b=np.asarray(bvs(s))
    return np.mean(np.abs(b-nearest_states(s,b)))


def rule_21(structure):

    def shell(s):

        n=len(s)
        if not n: raise ValueError("Empty structure")
        cutoff=max(2.0,1.5*(s.volume/n)**(1/3))
        for attempt in range(7):
            ii,jj,images,dd=s.get_neighbor_list(cutoff)
            mins=np.full(n,np.inf)
            use=dd>1e-8
            np.minimum.at(mins,ii[use],dd[use])
            if np.all(1.2*mins+1e-7<=cutoff): break
            cutoff*=2
        else: raise ValueError("Could not enclose complete periodic first shells")
        out=[[] for i in range(n)]
        for i,j,image,d in zip(ii,jj,images,dd):
            if d>1e-8 and d<=1.2*mins[i]+1e-7:
                out[int(i)].append((int(j),tuple(map(int,image)),float(d)))
        return out

    s=structure
    if not len(s): return float("nan")
    d=[d for row in shell(s) for j,t,d in row]
    return np.std(d)/np.mean(d)


def rule_22(structure):

    def cov(sym):
        fixed = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.5, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69}
        if sym not in fixed: raise ValueError("No tabulated covalent radius: "+sym)
        return fixed[sym]

    def parts(s):
        comp=s.composition.element_composition
        return [(el,float(n)/comp.num_atoms) for el,n in comp.items()]

    s=structure
    if not len(s): return float("nan")
    x=[(cov(e.symbol),p) for e,p in parts(s)]
    avg=sum(v*p for v,p in x)
    return math.sqrt(sum(p*(v-avg)**2 for v,p in x))


def rule_23(structure):

    def symbols(s):

        return [max(site.species.items(),key=lambda x:(x[1],x[0].symbol))[0].symbol for site in s]

    def cov(sym):
        fixed = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.5, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69}
        if sym not in fixed: raise ValueError("No tabulated covalent radius: "+sym)
        return fixed[sym]

    def shortest_translation(s):
        lat=s.lattice
        upper=min(lat.abc)
        pts=lat.get_points_in_sphere([[0,0,0]],[0,0,0],upper+1e-6)
        return min(float(q[1]) for q in pts if q[1]>1e-8)

    def contacts(s,mode):

        sym=symbols(s)
        rad=[cov(x) for x in sym]
        D=s.distance_matrix
        trans=shortest_translation(s)
        values=[]
        for i in range(len(s)):
            if mode=="metal" and not Element(sym[i]).is_metal: continue
            values.append(trans/(2*rad[i]))
            for j in range(i):
                if mode=="same" and sym[i]!=sym[j]: continue
                if mode=="metal" and not Element(sym[j]).is_metal: continue
                values.append(float(D[i,j])/(rad[i]+rad[j]))
        return values

    s=structure
    if not len(s): return float("nan")
    q=contacts(s,"metal")
    if not q: return 0.
    if min(q)<=0.: return float("nan")
    return 1./min(q)


def rule_24(structure):

    s=structure
    if not len(s): return float("nan")
    return s.composition.num_atoms/s.volume


def rule_25(structure):

    def parts(s):
        comp=s.composition.element_composition
        return [(el,float(n)/comp.num_atoms) for el,n in comp.items()]

    s=structure
    if not len(s): return float("nan")
    return sum(e.Z*p for e,p in parts(s))
