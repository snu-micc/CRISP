import math

import numpy as np

from pymatgen.analysis.bond_valence import BVAnalyzer

from pymatgen.analysis.local_env import CrystalNN, VoronoiNN

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatgen.core.periodic_table import Element

_OXIDATION_GUESS_CACHE = {}

def _species_symbol(specie):
    symbol = getattr(specie, "symbol", None)
    if symbol is None:
        symbol = getattr(getattr(specie, "element", None), "symbol", None)
    return str(symbol) if symbol else ""

def _site_fraction(site, symbol):
    return sum(
        float(occupancy)
        for specie, occupancy in site.species.items()
        if _species_symbol(specie) == symbol
    )

def _element_radius(symbol):
    try:
        element = Element(symbol)
    except Exception:
        return 1.0
    for attribute in (
        "average_ionic_radius",
        "atomic_radius",
        "atomic_radius_calculated",
    ):
        try:
            value = getattr(element, attribute)
            if value is not None and float(value) > 0.0:
                return float(value)
        except Exception:
            pass
    return 1.0

def _site_mean_radius(site, excluded_symbol=None):
    weighted_radius = 0.0
    total_weight = 0.0
    for specie, occupancy in site.species.items():
        symbol = _species_symbol(specie)
        if not symbol or symbol == excluded_symbol:
            continue
        weight = max(0.0, float(occupancy))
        weighted_radius += weight * _element_radius(symbol)
        total_weight += weight
    if total_weight <= 0.0:
        return None
    return weighted_radius / total_weight

def _composition_charge_map(structure):
    try:
        composition = structure.composition.element_composition
    except Exception:
        return {}

    cache_key = composition.formula
    if cache_key in _OXIDATION_GUESS_CACHE:
        return dict(_OXIDATION_GUESS_CACHE[cache_key])

    try:
        guesses = composition.oxi_state_guesses(max_sites=40)
    except Exception:
        guesses = ()

    if not guesses:
        _OXIDATION_GUESS_CACHE[cache_key] = {}
        return {}

    charge_map = {}
    for key, value in guesses[0].items():
        symbol = getattr(key, "symbol", str(key))
        try:
            charge_map[str(symbol)] = float(value)
        except Exception:
            pass
    _OXIDATION_GUESS_CACHE[cache_key] = dict(charge_map)
    return charge_map

def _dominant_mobile_symbol(structure, mobile_cutoffs):
    amounts = {symbol: 0.0 for symbol in mobile_cutoffs}
    for site in structure:
        for symbol in mobile_cutoffs:
            amounts[symbol] += _site_fraction(site, symbol)
    present = [(amount, symbol) for symbol, amount in amounts.items() if amount > 1e-8]
    return max(present)[1] if present else None

def rule_mobile_ion_defect_connectivity(structure):
    try:
        cutoffs = {"H": 2.5, "Li": 4.2, "Na": 4.8, "K": 5.3, "Rb": 5.6, "Cs": 6.0, "Cu": 4.0, "Ag": 4.5, "Mg": 4.0, "Ca": 4.8, "Zn": 4.0, "O": 3.5, "F": 3.5, "Cl": 4.5, "Br": 4.8, "I": 5.2}
        best = 0.0
        for symbol, cutoff in cutoffs.items():
            candidates = []
            mobile_occ = {}
            vacancy = {}
            for i, site in enumerate(structure):
                total = 0.0
                mobile = 0.0
                for specie, occ in site.species.items():
                    amount = float(occ)
                    total += amount
                    if getattr(specie, "symbol", str(specie)) == symbol:
                        mobile += amount
                if mobile > 1e-8:
                    candidates.append(i)
                    mobile_occ[i] = min(1.0, max(0.0, mobile))
                    vacancy[i] = min(1.0, max(0.0, 1.0 - total))
            if not candidates or sum(vacancy.values()) <= 1e-8:
                continue
            candidate_set = set(candidates)
            adjacency = {i: set() for i in candidates}
            degree = {i: 0 for i in candidates}
            for i in candidates:
                for neighbor in structure.get_neighbors(structure[i], cutoff):
                    j = getattr(neighbor, "index", None)
                    if j in candidate_set:
                        degree[i] += 1
                        if j != i:
                            adjacency[i].add(j)
                            adjacency[j].add(i)
            unseen = set(candidates)
            while unseen:
                root = unseen.pop()
                component = {root}
                stack = [root]
                while stack:
                    current = stack.pop()
                    for nxt in adjacency[current]:
                        if nxt in unseen:
                            unseen.remove(nxt)
                            component.add(nxt)
                            stack.append(nxt)
                size = float(len(component))
                mean_mobile = sum(mobile_occ[i] for i in component) / size
                mean_vacancy = sum(vacancy[i] for i in component) / size
                defect_balance = min(1.0, 4.0 * mean_mobile * mean_vacancy)
                mean_degree = sum(degree[i] for i in component) / size
                pathway_factor = min(1.0, mean_degree / 2.0)
                coverage = size / float(len(candidates))
                best = max(best, defect_balance * pathway_factor * coverage)
        return float(min(1.0, max(0.0, best)))
    except Exception:
        return 0.0

def rule_r02_soft_anion_framework(structure):
    try:
        polarizability = {"H": 0.67, "N": 1.10, "O": 0.80, "F": 0.56, "P": 3.63, "S": 2.90, "Cl": 2.18, "As": 4.31, "Se": 3.77, "Br": 3.05, "Sb": 6.60, "Te": 5.50, "I": 5.35}
        anion_charge = {"H": 1.0, "N": 3.0, "O": 2.0, "F": 1.0, "P": 3.0, "S": 2.0, "Cl": 1.0, "As": 3.0, "Se": 2.0, "Br": 1.0, "Sb": 3.0, "Te": 2.0, "I": 1.0}
        electronegativity = {"H": 2.20, "N": 3.04, "O": 3.44, "F": 3.98, "P": 2.19, "S": 2.58, "Cl": 3.16, "As": 2.18, "Se": 2.55, "Br": 2.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "B": 2.04, "Si": 1.90, "Ge": 2.01, "Sn": 1.96, "Al": 1.61, "Ga": 1.81, "In": 1.78}
        radii = {"H": 0.31, "Li": 1.28, "Na": 1.66, "K": 2.03, "Rb": 2.20, "Cs": 2.44, "B": 0.84, "N": 0.71, "O": 0.66, "F": 0.57, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Ag": 1.45, "Cu": 1.32}
        mobile = {"Li", "Na", "K", "Rb", "Cs", "Ag"}
        symbols = []
        for site in structure:
            items = list(site.species.items())
            if not items:
                symbols.append("")
            else:
                sp = max(items, key=lambda x: float(x[1]))[0]
                symbols.append(getattr(sp, "symbol", str(sp).split("+")[0].split("-")[0]))
        soft_sum = 0.0
        anion_weight = 0.0
        for site in structure:
            for sp, occ in site.species.items():
                sym = getattr(sp, "symbol", str(sp).split("+")[0].split("-")[0])
                if sym in polarizability:
                    w = float(occ)
                    soft_sum += w * polarizability[sym] / anion_charge[sym]
                    anion_weight += w
        if anion_weight == 0.0:
            return 0.0
        softness = soft_sum / anion_weight
        host = [i for i, s in enumerate(symbols) if s and s not in mobile]
        if not host:
            return 0.0
        host_set = set(host)
        degrees = {i: 0 for i in host}
        bond_weakness = []
        bond_lengths = []
        for i in host:
            try:
                neighbors = structure.get_neighbors(structure[i], 4.0)
            except Exception:
                continue
            for nn in neighbors:
                try:
                    j = int(nn.index)
                    if j not in host_set or j <= i:
                        continue
                    d = float(nn.nn_distance)
                    si, sj = symbols[i], symbols[j]
                    cutoff = radii.get(si, 1.25) + radii.get(sj, 1.25) + 0.55
                    if d > cutoff:
                        continue
                    degrees[i] += 1
                    degrees[j] += 1
                    ai = si in polarizability
                    aj = sj in polarizability
                    if ai != aj:
                        xa = electronegativity.get(si, 2.0)
                        xb = electronegativity.get(sj, 2.0)
                        strength_proxy = abs(xa - xb) / max(d * d, 0.25)
                        bond_weakness.append(1.0 / (1.0 + 2.0 * strength_proxy))
                        bond_lengths.append(d)
                except Exception:
                    continue
        connected_fraction = sum(1 for i in host if degrees[i] > 0) / float(len(host))
        mean_degree = sum(degrees.values()) / float(len(host))
        connectivity = connected_fraction * mean_degree / (mean_degree + 2.0)
        compliance = sum(bond_weakness) / len(bond_weakness) if bond_weakness else 0.5
        if len(bond_lengths) > 1:
            mean_length = sum(bond_lengths) / len(bond_lengths)
            variance = sum((x - mean_length) ** 2 for x in bond_lengths) / len(bond_lengths)
            compliance *= 1.0 + min(0.25, variance ** 0.5 / max(mean_length, 1e-8))
        return float(softness * compliance * connectivity)
    except Exception:
        return 0.0

def rule_accessible_free_volume_connectivity(structure):
    try:
        fallback = {"H": 0.25, "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Cu": 0.77, "Ag": 1.15, "Mg": 0.72, "Ca": 1.00, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20, "O": 1.40, "S": 1.84, "N": 1.46, "P": 1.07, "Si": 1.11, "Al": 1.18}
        def dominant(site):
            return max(site.species.items(), key=lambda x: float(x[1]))[0]
        def symbol(specie):
            return str(getattr(specie, "symbol", getattr(getattr(specie, "element", None), "symbol", "")))
        def radius(specie):
            el = getattr(specie, "element", specie)
            for obj, attr in ((specie, "ionic_radius"), (el, "average_ionic_radius"), (el, "atomic_radius"), (el, "atomic_radius_calculated")):
                try:
                    value = float(getattr(obj, attr))
                    if 0.2 <= value <= 3.0:
                        return value
                except Exception:
                    pass
            return fallback.get(symbol(specie), 1.2)
        present = {symbol(dominant(site)) for site in structure}
        mobile_symbol = next((x for x in ("Li", "Na", "K", "Rb", "Cs", "Ag", "Cu", "H", "Mg", "Ca") if x in present), None)
        if mobile_symbol is None or float(structure.volume) <= 0:
            return 0.0
        mobile = []
        framework = []
        occupied_volume = 0.0
        mobile_radii = []
        for i, site in enumerate(structure):
            sp = dominant(site)
            if symbol(sp) == mobile_symbol:
                mobile.append(i)
                mobile_radii.append(radius(sp))
            else:
                framework.append((site, radius(sp)))
                for species, occ in site.species.items():
                    r = radius(species)
                    occupied_volume += float(occ) * 4.1887902047863905 * r * r * r
        if not mobile or not framework:
            return 0.0
        mobile_radius = sum(mobile_radii) / len(mobile_radii)
        free_fraction = max(0.0, min(1.0, 1.0 - occupied_volume / float(structure.volume)))
        openness_x = max(0.0, min(1.0, (free_fraction - 0.12) / 0.76))
        openness_score = 4.0 * openness_x * (1.0 - openness_x)
        lattice = structure.lattice
        def clearance(frac):
            best = float("inf")
            for fw_site, fw_radius in framework:
                d = float(lattice.get_distance_and_image(frac, fw_site.frac_coords)[0]) - fw_radius
                if d < best:
                    best = d
            return best
        def fit_score(value):
            ratio = value / mobile_radius if mobile_radius > 0 else 0.0
            return max(0.0, min(1.0, (ratio - 0.65) / 0.85))
        local_score = sum(fit_score(clearance(structure[i].frac_coords)) for i in mobile) / len(mobile)
        network_scores = []
        translations = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
        for i in mobile:
            fi = structure[i].frac_coords
            candidates = []
            for j in mobile:
                fj = structure[j].frac_coords
                for t in translations:
                    delta = [float(fj[k]) + t[k] - float(fi[k]) for k in range(3)]
                    cart = lattice.get_cartesian_coords(delta)
                    dist = sum(float(x) * float(x) for x in cart) ** 0.5
                    candidates.append((dist, delta))
            candidates.sort(key=lambda x: x[0])
            best_path = 0.0
            for dist, delta in candidates[:8]:
                bottleneck = float("inf")
                for q in (0.25, 0.5, 0.75):
                    point = [float(fi[k]) + q * delta[k] for k in range(3)]
                    bottleneck = min(bottleneck, clearance(point))
                length_score = max(0.0, min(1.0, (8.0 - dist) / 4.0))
                best_path = max(best_path, fit_score(bottleneck) * length_score)
            network_scores.append(best_path)
        connectivity_score = sum(network_scores) / len(network_scores)
        return float(openness_score * (0.45 * local_score + 0.55 * connectivity_score))
    except Exception:
        return 0.0

def rule_mobile_ion_dynamic_delocalization(structure):
    try:
        hop_cutoffs = {
            "H": 2.2,
            "Li": 3.0,
            "Na": 3.5,
            "K": 4.1,
            "Rb": 4.4,
            "Cs": 4.7,
            "Cu": 3.0,
            "Ag": 3.3,
        }
        mobile = _dominant_mobile_symbol(structure, hop_cutoffs)
        if mobile is None:
            return 0.0

        nodes = []
        partial_values = []
        partial_weights = []
        for index, site in enumerate(structure):
            mobile_occupancy = _site_fraction(site, mobile)
            if mobile_occupancy <= 0.02:
                continue
            nodes.append(index)
            probabilities = [
                max(0.0, float(occupancy))
                for occupancy in site.species.values()
            ]
            total_occupancy = sum(probabilities)
            if total_occupancy < 1.0:
                probabilities.append(1.0 - total_occupancy)
            elif total_occupancy > 1.0:
                probabilities = [value / total_occupancy for value in probabilities]
            disorder = 1.0 - sum(value * value for value in probabilities)
            partial_values.append(min(1.0, max(0.0, 2.0 * disorder)))
            partial_weights.append(min(1.0, mobile_occupancy))

        if not nodes:
            return 0.0

        partial_score = 0.0
        if sum(partial_weights) > 0.0:
            partial_score = sum(
                value * weight
                for value, weight in zip(partial_values, partial_weights)
            ) / sum(partial_weights)

        node_set = set(nodes)
        position = {index: offset for offset, index in enumerate(nodes)}
        fingerprints = {}
        coordination = {}
        for index in nodes:
            framework_neighbors = []
            for neighbor in structure.get_neighbors(structure[index], 4.8):
                if _site_fraction(neighbor, mobile) > 0.02:
                    continue
                distance = float(neighbor.nn_distance)
                if distance > 0.35:
                    framework_neighbors.append(distance)
            framework_neighbors.sort()
            fingerprints[index] = framework_neighbors[:10]
            if framework_neighbors:
                shell_limit = min(4.3, framework_neighbors[0] + 0.85)
                coordination[index] = sum(
                    1 for distance in framework_neighbors if distance <= shell_limit
                )
            else:
                coordination[index] = 0

        adjacency = [set() for _ in nodes]
        path_counts = [0 for _ in nodes]
        split_scores = []
        equivalence_scores = []
        short_hop_scores = []
        cutoff = hop_cutoffs[mobile]

        for index in nodes:
            source = position[index]
            for neighbor in structure.get_neighbors(structure[index], cutoff):
                target_index = getattr(neighbor, "index", None)
                distance = float(neighbor.nn_distance)
                if (
                    target_index is None
                    or target_index not in node_set
                    or distance <= 0.35
                ):
                    continue

                path_counts[source] += 1
                target = position[target_index]
                if target != source:
                    adjacency[source].add(target)
                    adjacency[target].add(source)

                target_distance = 0.72 * cutoff
                width = max(0.35, 0.24 * cutoff)
                short_hop_scores.append(
                    math.exp(-((distance - target_distance) / width) ** 2)
                )

                split_limit = 0.55 * cutoff
                if distance < split_limit:
                    split_scores.append(
                        min(
                            1.0,
                            max(
                                0.0,
                                (split_limit - distance)
                                / max(0.4, split_limit - 0.35),
                            ),
                        )
                    )

                if target_index != index:
                    first = fingerprints[index]
                    second = fingerprints[target_index]
                    count = min(len(first), len(second))
                    if count:
                        rms = math.sqrt(
                            sum(
                                (first[offset] - second[offset]) ** 2
                                for offset in range(count)
                            )
                            / count
                        )
                        count_mismatch = abs(
                            coordination[index] - coordination[target_index]
                        ) / max(
                            1.0,
                            float(
                                max(
                                    coordination[index],
                                    coordination[target_index],
                                )
                            ),
                        )
                        length_penalty = 0.08 * abs(len(first) - len(second))
                        equivalence_scores.append(
                            1.0
                            / (
                                1.0
                                + rms / 0.25
                                + count_mismatch
                                + length_penalty
                            )
                        )

        split_score = (
            sum(split_scores) / len(split_scores) if split_scores else 0.0
        )
        equivalence_score = (
            sum(equivalence_scores) / len(equivalence_scores)
            if equivalence_scores
            else 0.0
        )
        short_hop_score = (
            sum(short_hop_scores) / len(short_hop_scores)
            if short_hop_scores
            else 0.0
        )

        visited = set()
        largest_component = 0
        for start in range(len(nodes)):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component_size = 0
            while stack:
                current = stack.pop()
                component_size += 1
                for target in adjacency[current]:
                    if target not in visited:
                        visited.add(target)
                        stack.append(target)
            largest_component = max(largest_component, component_size)

        active_fraction = sum(count > 0 for count in path_counts) / float(len(nodes))
        connected_fraction = largest_component / float(len(nodes))
        mean_path_count = sum(path_counts) / float(len(nodes))
        degree_support = 1.0 - math.exp(-mean_path_count / 2.5)
        network_score = (
            0.60 * connected_fraction + 0.40 * active_fraction
        ) * degree_support

        score = (
            0.30 * partial_score
            + 0.15 * split_score
            + 0.22 * equivalence_score
            + 0.23 * network_score
            + 0.10 * short_hop_score
        )
        return float(min(1.0, max(0.0, score)))
    except Exception:
        return 0.0

def rule_05_mobile_ion_hop_network(structure):
    try:
        mobile_data = {
            "H": (0.38, 2.0, 3.2, 0.8), "Li": (0.76, 2.8, 4.2, 1.0),
            "Na": (1.02, 3.2, 4.8, 0.95), "K": (1.38, 3.7, 5.5, 0.85),
            "Rb": (1.52, 4.0, 5.8, 0.75), "Cs": (1.67, 4.3, 6.2, 0.65),
            "Ag": (1.15, 3.1, 4.8, 0.85), "Cu": (0.77, 2.8, 4.3, 0.8),
            "Mg": (0.72, 2.9, 4.2, 0.45), "Ca": (1.00, 3.3, 4.8, 0.4)
        }
        def symbol_of(sp):
            return getattr(sp, "symbol", getattr(getattr(sp, "element", None), "symbol", ""))
        amounts = {}
        for site in structure:
            for sp, occ in site.species.items():
                s = symbol_of(sp)
                if s in mobile_data:
                    amounts[s] = amounts.get(s, 0.0) + float(occ)
        if not amounts:
            return 0.0
        mobile = max(amounts, key=lambda s: amounts[s] * mobile_data[s][3])
        radius, optimal, cutoff, mobility = mobile_data[mobile]
        nodes = []
        framework = []
        vacancies = []
        for i, site in enumerate(structure):
            mobile_occ = 0.0
            total_occ = 0.0
            has_other = False
            for sp, occ in site.species.items():
                o = float(occ)
                total_occ += o
                if symbol_of(sp) == mobile:
                    mobile_occ += o
                elif o > 0.0:
                    has_other = True
            if mobile_occ > 1e-8:
                nodes.append((i, mobile_occ))
                vacancies.append(max(0.0, min(1.0, 1.0 - mobile_occ)))
            if has_other or mobile_occ <= 1e-8:
                framework.append(i)
        n = len(nodes)
        if n < 2:
            return 0.0
        def framework_radius(site):
            total = 0.0
            weight = 0.0
            for sp, occ in site.species.items():
                if symbol_of(sp) == mobile:
                    continue
                r = getattr(getattr(sp, "element", sp), "atomic_radius", None)
                try:
                    rv = float(r) if r is not None else 1.0
                except Exception:
                    rv = 1.0
                total += float(occ) * rv
                weight += float(occ)
            return total / weight if weight > 0.0 else 1.0
        framework_radii = {i: framework_radius(structure[i]) for i in framework}
        adjacency = [[] for _ in range(n)]
        edge_scores = []
        nearest_hops = [None for _ in range(n)]
        repulsion = 0.0
        repulsion_pairs = 0
        for a in range(n):
            ia, oa = nodes[a]
            for b in range(a + 1, n):
                ib, ob = nodes[b]
                d = float(structure.get_distance(ia, ib))
                if d < 2.0 * radius:
                    repulsion += oa * ob * (2.0 * radius - d) / max(2.0 * radius, 1e-8)
                    repulsion_pairs += 1
                if d > cutoff:
                    continue
                if nearest_hops[a] is None or d < nearest_hops[a]:
                    nearest_hops[a] = d
                if nearest_hops[b] is None or d < nearest_hops[b]:
                    nearest_hops[b] = d
                width = max(0.7, 0.35 * optimal)
                distance_score = 1.0 / (1.0 + ((d - optimal) / width) ** 2)
                if d < 1.7 * radius:
                    distance_score *= max(0.0, d / max(1.7 * radius, 1e-8)) ** 3
                fa = structure[ia].frac_coords
                fb = structure[ib].frac_coords
                delta = [float(fb[k] - fa[k]) for k in range(3)]
                delta = [x - round(x) for x in delta]
                midpoint = [float(fa[k]) + 0.5 * delta[k] for k in range(3)]
                clearance = 2.0
                if framework:
                    clearance = min(float(structure.lattice.get_distance_and_image(midpoint, structure[j].frac_coords)[0]) - framework_radii[j] - radius for j in framework)
                bottleneck_score = max(0.0, min(1.0, (clearance + 0.15) / 0.85))
                edge = distance_score * bottleneck_score * (oa * ob) ** 0.5
                if edge > 0.03:
                    adjacency[a].append((b, edge))
                    adjacency[b].append((a, edge))
                    edge_scores.append(edge)
        if not edge_scores:
            return 0.0
        seen = set()
        largest = 0
        for start in range(n):
            if start in seen:
                continue
            stack = [start]
            seen.add(start)
            size = 0
            while stack:
                u = stack.pop()
                size += 1
                for v, w in adjacency[u]:
                    if v not in seen:
                        seen.add(v)
                        stack.append(v)
            largest = max(largest, size)
        connectivity = largest / float(n)
        node_quality = sum(max((w for _, w in links), default=0.0) for links in adjacency) / float(n)
        hops = [x for x in nearest_hops if x is not None]
        if hops:
            mean_hop = sum(hops) / len(hops)
            variance = sum((x - mean_hop) ** 2 for x in hops) / len(hops)
            regularity = 1.0 / (1.0 + variance / max(mean_hop * mean_hop, 1e-8) * 16.0)
        else:
            regularity = 0.0
        local_clearances = []
        if framework:
            for i, occ in nodes:
                c = min(float(structure.get_distance(i, j)) - framework_radii[j] - radius for j in framework if j != i)
                local_clearances.append(c)
        if len(local_clearances) > 1:
            m = sum(local_clearances) / len(local_clearances)
            v = sum((x - m) ** 2 for x in local_clearances) / len(local_clearances)
            trap_uniformity = 1.0 / (1.0 + 4.0 * v / max((abs(m) + 0.5) ** 2, 1e-8))
        else:
            trap_uniformity = 1.0
        vacancy_factor = 0.2 + 0.8 * sum(vacancies) / len(vacancies)
        repulsion_factor = 1.0 / (1.0 + (repulsion / max(repulsion_pairs, 1)) * 4.0)
        score = mobility * node_quality * connectivity * regularity * trap_uniformity * vacancy_factor * repulsion_factor
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0

def rule_r06_bottleneck_to_mobile_ion_ratio(structure):
    try:
        mobile_radii = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Ag": 1.15, "Cu": 0.77, "Mg": 0.72, "Ca": 1.00}
        framework_radii = {"H": 0.25, "B": 0.27, "C": 0.30, "N": 1.46, "O": 1.40, "F": 1.33, "Si": 0.40, "P": 0.38, "S": 1.84, "Cl": 1.81, "Br": 1.96, "I": 2.20, "Al": 0.54, "Ga": 0.62, "Ge": 0.53, "As": 0.58, "Se": 1.98, "Te": 2.21, "Ti": 0.61, "Zr": 0.72, "Hf": 0.71, "V": 0.59, "Nb": 0.64, "Ta": 0.64, "Cr": 0.62, "Mo": 0.65, "W": 0.66, "Mn": 0.67, "Fe": 0.65, "Co": 0.65, "Ni": 0.69, "Zn": 0.74, "Cd": 0.95, "La": 1.03, "Ce": 1.01}

        def symbol_of(site):
            try:
                return site.specie.symbol
            except Exception:
                try:
                    return site.specie.element.symbol
                except Exception:
                    try:
                        return max(site.species.items(), key=lambda x: float(x[1]))[0].symbol
                    except Exception:
                        return ""

        present = {}
        for site in structure:
            for sp, occ in site.species.items():
                sym = getattr(sp, "symbol", getattr(getattr(sp, "element", None), "symbol", ""))
                present[sym] = present.get(sym, 0.0) + float(occ)
        candidates = [s for s in ("Li", "Na", "K", "Ag", "Cu", "Rb", "Cs", "Mg", "Ca") if present.get(s, 0.0) > 0.0]
        if not candidates:
            return 0.0
        mobile = min(candidates, key=lambda s: present[s])
        ion_radius = mobile_radii[mobile]
        mobile_sites = [site for site in structure if symbol_of(site) == mobile]
        if not mobile_sites:
            return 0.0

        nearest = None
        neighbor_lists = []
        for site in mobile_sites:
            neighbors = []
            for nn in structure.get_neighbors(site, 9.0):
                if symbol_of(nn) != mobile:
                    continue
                d = float(getattr(nn, "nn_distance", site.distance(nn)))
                if d > 0.5:
                    neighbors.append((nn, d))
                    if nearest is None or d < nearest:
                        nearest = d
            neighbor_lists.append(neighbors)
        if nearest is None:
            return 0.0

        cutoff = min(9.0, nearest * 1.35)
        path_ratios = []
        connected_sites = 0
        for site, neighbors in zip(mobile_sites, neighbor_lists):
            usable = [(nn, d) for nn, d in neighbors if d <= cutoff]
            if len(usable) >= 2:
                connected_sites += 1
            for nn, d in usable:
                p0 = site.coords
                p1 = nn.coords
                clearances = []
                for t in (0.2, 0.35, 0.5, 0.65, 0.8):
                    point = (float(p0[0] + t * (p1[0] - p0[0])), float(p0[1] + t * (p1[1] - p0[1])), float(p0[2] + t * (p1[2] - p0[2])))
                    local_clearance = None
                    for item in structure.get_sites_in_sphere(point, 5.0):
                        obj = item[0] if isinstance(item, (tuple, list)) else item
                        sym = symbol_of(obj)
                        if sym == mobile:
                            continue
                        try:
                            dist = float(item[1]) if isinstance(item, (tuple, list)) else float(obj.nn_distance)
                        except Exception:
                            q = obj.coords
                            dist = ((float(q[0]) - point[0]) ** 2 + (float(q[1]) - point[1]) ** 2 + (float(q[2]) - point[2]) ** 2) ** 0.5
                        radius = framework_radii.get(sym, 1.0)
                        clearance = dist - radius
                        if local_clearance is None or clearance < local_clearance:
                            local_clearance = clearance
                    if local_clearance is not None:
                        clearances.append(local_clearance)
                if clearances:
                    path_ratios.append(max(0.0, min(clearances)) / ion_radius)
        if not path_ratios:
            return 0.0
        mean_ratio = sum(path_ratios) / len(path_ratios)
        variance = sum((x - mean_ratio) ** 2 for x in path_ratios) / len(path_ratios)
        distortion_factor = 1.0 / (1.0 + variance ** 0.5 / max(mean_ratio, 1e-8))
        connectivity_factor = connected_sites / float(len(mobile_sites))
        return float(mean_ratio * distortion_factor * connectivity_factor)
    except Exception:
        return 0.0

def rule_r07_symmetric_mobile_ion_pathways(structure):
    try:
        mobile_order = ("Li", "Na", "K", "Rb", "Cs", "Ag", "Cu")

        def occupancy_of(obj, symbol):
            try:
                return float(sum(occ for sp, occ in obj.species.items() if sp.symbol == symbol))
            except Exception:
                try:
                    return 1.0 if obj.specie.symbol == symbol else 0.0
                except Exception:
                    return 0.0

        chosen = None
        for symbol in mobile_order:
            if sum(occupancy_of(site, symbol) for site in structure) > 0.0:
                chosen = symbol
                break
        if chosen is None:
            return 0.0

        mobile_indices = [i for i, site in enumerate(structure) if occupancy_of(site, chosen) > 0.05]
        if not mobile_indices:
            return 0.0

        analyzer = SpacegroupAnalyzer(structure, symprec=0.1, angle_tolerance=5.0)
        try:
            operation_count = len(analyzer.get_symmetry_operations(cartesian=False))
            symmetry_score = min(1.0, max(0.0, operation_count / 192.0) ** 0.5)
        except Exception:
            symmetry_score = 0.0

        try:
            dataset = analyzer.get_symmetry_dataset()
            equivalent_atoms = getattr(dataset, "equivalent_atoms", None)
            if equivalent_atoms is None:
                equivalent_atoms = dataset["equivalent_atoms"]
            orbit_counts = {}
            for i in mobile_indices:
                orbit = int(equivalent_atoms[i])
                orbit_counts[orbit] = orbit_counts.get(orbit, 0) + 1
            n_mobile = float(len(mobile_indices))
            equivalence_score = sum(v * v for v in orbit_counts.values()) / (n_mobile * n_mobile)
        except Exception:
            equivalence_score = 0.0

        a, b, c = [float(x) for x in structure.lattice.abc]
        lengths = [a, b, c]
        mean_length = sum(lengths) / 3.0
        length_cv = (sum((x - mean_length) ** 2 for x in lengths) / 3.0) ** 0.5 / mean_length if mean_length > 0 else 1.0
        angles = [float(x) for x in structure.lattice.angles]
        angle_distortion = sum(min(abs(x - 60.0), abs(x - 90.0), abs(x - 120.0)) / 30.0 for x in angles) / 3.0
        lattice_score = 1.0 / (1.0 + length_cv + angle_distortion)

        neighbor_lists = []
        all_mobile_distances = []
        framework_shell_means = []
        framework_shell_cvs = []
        for i in mobile_indices:
            neighbors = structure.get_neighbors(structure[i], 8.0)
            mobile_distances = []
            framework_distances = []
            for neighbor in neighbors:
                distance = float(neighbor.nn_distance)
                if distance <= 1e-6:
                    continue
                if occupancy_of(neighbor, chosen) > 0.05:
                    mobile_distances.append(distance)
                    all_mobile_distances.append(distance)
                else:
                    framework_distances.append(distance)
            neighbor_lists.append(mobile_distances)
            if framework_distances:
                nearest = min(framework_distances)
                shell = [d for d in framework_distances if d <= 1.25 * nearest]
                shell_mean = sum(shell) / len(shell)
                shell_cv = (sum((d - shell_mean) ** 2 for d in shell) / len(shell)) ** 0.5 / shell_mean if shell_mean > 0 else 1.0
                framework_shell_means.append(shell_mean)
                framework_shell_cvs.append(shell_cv)

        if all_mobile_distances:
            minimum_hop = min(all_mobile_distances)
            cutoff = 1.30 * minimum_hop
            coordinations = [sum(1 for d in distances if d <= cutoff) for distances in neighbor_lists]
            selected_hops = [d for distances in neighbor_lists for d in distances if d <= cutoff]
            mean_coordination = sum(coordinations) / len(coordinations)
            coordination_cv = (sum((x - mean_coordination) ** 2 for x in coordinations) / len(coordinations)) ** 0.5 / mean_coordination if mean_coordination > 0 else 1.0
            mean_hop = sum(selected_hops) / len(selected_hops)
            hop_cv = (sum((d - mean_hop) ** 2 for d in selected_hops) / len(selected_hops)) ** 0.5 / mean_hop if mean_hop > 0 else 1.0
            connectivity_score = min(1.0, mean_coordination / 6.0)
            pathway_score = connectivity_score / (1.0 + coordination_cv + hop_cv)
        else:
            pathway_score = 0.0

        if framework_shell_means:
            mean_shell = sum(framework_shell_means) / len(framework_shell_means)
            between_site_cv = (sum((x - mean_shell) ** 2 for x in framework_shell_means) / len(framework_shell_means)) ** 0.5 / mean_shell if mean_shell > 0 else 1.0
            local_distortion = sum(framework_shell_cvs) / len(framework_shell_cvs)
            bottleneck_score = 1.0 / (1.0 + between_site_cv + local_distortion)
        else:
            bottleneck_score = 0.0

        score = 0.30 * symmetry_score + 0.20 * equivalence_score + 0.15 * lattice_score + 0.20 * pathway_score + 0.15 * bottleneck_score
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0

def rule_r08_mobile_ion_network_3d(structure):
    try:
        mobile = {"H": 3.0, "Li": 4.5, "Na": 5.0, "K": 5.5, "Rb": 5.8, "Cs": 6.2, "Ag": 5.0, "Cu": 4.5, "Mg": 4.5, "Ca": 5.0, "Zn": 4.5}
        groups = {}
        for index, site in enumerate(structure):
            for specie, occupancy in site.species.items():
                symbol = getattr(specie, "symbol", str(specie).split("+")[0].split("-")[0])
                if symbol in mobile and float(occupancy) > 0.1:
                    groups.setdefault(symbol, []).append(index)
        if not groups:
            return 0.0

        def vector_rank(vectors):
            matrix = [[float(x) for x in v] for v in vectors if any(abs(float(x)) > 1e-10 for x in v)]
            if not matrix:
                return 0
            rank = 0
            for column in range(3):
                pivot = None
                for row in range(rank, len(matrix)):
                    if abs(matrix[row][column]) > 1e-9:
                        pivot = row
                        break
                if pivot is None:
                    continue
                matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
                divisor = matrix[rank][column]
                matrix[rank] = [x / divisor for x in matrix[rank]]
                for row in range(len(matrix)):
                    if row != rank:
                        factor = matrix[row][column]
                        matrix[row] = [matrix[row][k] - factor * matrix[rank][k] for k in range(3)]
                rank += 1
                if rank == 3:
                    break
            return rank

        best = 0.0
        for symbol, original_indices in groups.items():
            n = len(original_indices)
            if n == 0:
                continue
            frac = [structure[i].frac_coords for i in original_indices]
            candidates = []
            nearest = None
            for i in range(n):
                for j in range(i, n):
                    for tx in (-1, 0, 1):
                        for ty in (-1, 0, 1):
                            for tz in (-1, 0, 1):
                                if i == j and (tx, ty, tz) <= (0, 0, 0):
                                    continue
                                df = (float(frac[j][0] - frac[i][0] + tx), float(frac[j][1] - frac[i][1] + ty), float(frac[j][2] - frac[i][2] + tz))
                                cart = structure.lattice.get_cartesian_coords(df)
                                distance = float(sum(float(x) * float(x) for x in cart) ** 0.5)
                                if distance > 1e-8:
                                    candidates.append((distance, i, j, (tx, ty, tz), df))
                                    if nearest is None or distance < nearest:
                                        nearest = distance
            if nearest is None or nearest > mobile[symbol]:
                continue
            cutoff = min(mobile[symbol], max(3.0, 1.35 * nearest))
            edges = [x for x in candidates if x[0] <= cutoff + 1e-8]
            adjacency = [[] for _ in range(n)]
            for edge_id, edge in enumerate(edges):
                _, i, j, shift, _ = edge
                adjacency[i].append((j, shift, edge_id))
                adjacency[j].append((i, (-shift[0], -shift[1], -shift[2]), edge_id))
            seen = set()
            for start in range(n):
                if start in seen:
                    continue
                stack = [start]
                seen.add(start)
                potential = {start: (0, 0, 0)}
                component = []
                edge_ids = set()
                cycles = []
                while stack:
                    u = stack.pop()
                    component.append(u)
                    for v, shift, edge_id in adjacency[u]:
                        edge_ids.add(edge_id)
                        expected = tuple(potential[u][k] + shift[k] for k in range(3))
                        if v not in potential:
                            potential[v] = expected
                            seen.add(v)
                            stack.append(v)
                        else:
                            cycle = tuple(expected[k] - potential[v][k] for k in range(3))
                            if any(cycle):
                                cycles.append(cycle)
                rank = vector_rank(cycles)
                if rank == 0 or not edge_ids:
                    continue
                directional = [0.0, 0.0, 0.0]
                for edge_id in edge_ids:
                    df = edges[edge_id][4]
                    norm = sum(float(x) * float(x) for x in df)
                    if norm > 1e-12:
                        for k in range(3):
                            directional[k] += float(df[k]) * float(df[k]) / norm
                total_directional = sum(directional)
                isotropy = min(1.0, 3.0 * min(directional) / total_directional) if total_directional > 0 else 0.0
                coordination = 2.0 * len(edge_ids) / max(1, len(component))
                redundancy = max(0.0, min(1.0, (coordination - 2.0) / 4.0))
                coverage = float(len(component)) / n
                dimensionality = float(rank) / 3.0
                score = coverage * dimensionality * dimensionality * (0.5 + 0.5 * isotropy) * (0.5 + 0.5 * redundancy)
                best = max(best, score)
        return float(best)
    except Exception:
        return 0.0

def rule_r09_mobile_site_shallowness(structure):
    try:
        def dominant_species(site):
            try:
                return max(
                    site.species.items(),
                    key=lambda item: float(item[1]),
                )[0]
            except Exception:
                return None

        def symbol_of(site):
            specie = dominant_species(site)
            return _species_symbol(specie) if specie is not None else ""

        def radius_of(site):
            weighted_radius = 0.0
            total_weight = 0.0
            for specie, occupancy in site.species.items():
                symbol = _species_symbol(specie)
                if not symbol:
                    continue
                weight = max(0.0, float(occupancy))
                weighted_radius += weight * _element_radius(symbol)
                total_weight += weight
            return weighted_radius / total_weight if total_weight > 0.0 else 1.0

        charge_map = _composition_charge_map(structure)

        def charge_of(site):
            if not charge_map:
                return None
            charge = 0.0
            represented = 0.0
            for specie, occupancy in site.species.items():
                symbol = _species_symbol(specie)
                if symbol not in charge_map:
                    continue
                weight = max(0.0, float(occupancy))
                charge += weight * charge_map[symbol]
                represented += weight
            return charge if represented > 0.0 else None

        symbols = {symbol_of(site) for site in structure}
        groups = [
            {"Li", "Na", "K", "Rb", "Cs", "Ag", "Cu", "H"},
            {"Mg", "Ca", "Sr", "Ba", "Zn"},
            {"F", "Cl"},
            {"O"},
        ]
        mobile_symbols = set()
        for group in groups:
            mobile_symbols = symbols.intersection(group)
            if mobile_symbols:
                break
        if not mobile_symbols:
            return 0.0

        descriptors = {}
        local_scores = []
        for index, site in enumerate(structure):
            if symbol_of(site) not in mobile_symbols:
                continue

            framework = []
            mobile_radius = radius_of(site)
            mobile_charge = charge_of(site)
            for neighbor in structure.get_neighbors(site, 5.0):
                if symbol_of(neighbor) in mobile_symbols:
                    continue
                distance = float(neighbor.nn_distance)
                if distance <= 0.1:
                    continue
                radius_ratio = distance / max(
                    mobile_radius + radius_of(neighbor),
                    0.5,
                )
                framework_charge = charge_of(neighbor)
                electrostatic_binding = None
                if mobile_charge is not None and framework_charge is not None:
                    electrostatic_binding = (
                        abs(mobile_charge * framework_charge)
                        / (distance * distance)
                    )
                framework.append(
                    (distance, radius_ratio, electrostatic_binding)
                )

            if not framework:
                continue
            framework.sort(key=lambda item: item[0])
            nearest_distance = framework[0][0]
            shell_limit = min(
                nearest_distance * 1.38,
                nearest_distance + 0.75,
            )
            shell = [item for item in framework if item[0] <= shell_limit]
            if not shell:
                continue

            coordination_number = len(shell)
            mean_distance = sum(item[0] for item in shell) / coordination_number
            variance = sum(
                (item[0] - mean_distance) ** 2 for item in shell
            ) / coordination_number
            distance_cv = (
                math.sqrt(variance) / mean_distance
                if mean_distance > 0.0
                else 0.0
            )
            mean_radius_ratio = (
                sum(item[1] for item in shell) / coordination_number
            )
            radial_weakness = max(0.0, mean_radius_ratio - 0.75)
            radial_weakness /= radial_weakness + 0.5

            electrostatic_values = [
                item[2] for item in shell if item[2] is not None
            ]
            if electrostatic_values:
                electrostatic_mean = (
                    sum(electrostatic_values) / len(electrostatic_values)
                )
                electrostatic_weakness = 1.0 / (
                    1.0 + 4.0 * electrostatic_mean
                )
                binding_weakness = (
                    0.58 * radial_weakness
                    + 0.42 * electrostatic_weakness
                )
            else:
                binding_weakness = radial_weakness

            undercoordination = 1.0 / (
                1.0 + coordination_number / 6.0
            )
            distortion = (
                distance_cv / (distance_cv + 0.08)
                if distance_cv > 0.0
                else 0.0
            )
            alternatives = sum(
                1
                for item in framework
                if shell_limit < item[0] <= nearest_distance * 1.75
            )
            flexibility = (
                alternatives / float(coordination_number + alternatives)
                if alternatives
                else 0.0
            )
            total_occupancy = sum(
                float(value) for value in site.species.values()
            )
            vacancy = max(0.0, min(1.0, 1.0 - total_occupancy))

            local_score = (
                0.31 * binding_weakness
                + 0.19 * undercoordination
                + 0.19 * distortion
                + 0.21 * flexibility
                + 0.10 * vacancy
            )
            descriptors[index] = (
                float(coordination_number),
                mean_radius_ratio,
                distance_cv,
            )
            local_scores.append(local_score)

        if not local_scores:
            return 0.0

        transition_scores = []
        for index, descriptor in descriptors.items():
            for neighbor in structure.get_neighbors(structure[index], 6.0):
                if symbol_of(neighbor) not in mobile_symbols:
                    continue
                target = getattr(neighbor, "index", None)
                distance = float(neighbor.nn_distance)
                if (
                    target is None
                    or target not in descriptors
                    or distance <= 0.2
                ):
                    continue
                other = descriptors[target]
                mismatch = abs(descriptor[0] - other[0]) / max(
                    descriptor[0],
                    other[0],
                    1.0,
                )
                mismatch += abs(descriptor[1] - other[1])
                mismatch += abs(descriptor[2] - other[2]) / 0.15
                smoothness = 1.0 / (1.0 + mismatch)
                distance_factor = 1.0 / (
                    1.0 + max(0.0, distance / 3.0 - 1.0)
                )
                transition_scores.append(smoothness * distance_factor)

        local_score = sum(local_scores) / len(local_scores)
        transition_score = (
            sum(transition_scores) / len(transition_scores)
            if transition_scores
            else 0.45
        )
        result = 0.76 * local_score + 0.24 * transition_score
        return float(min(1.0, max(0.0, result)))
    except Exception:
        return 0.0

def rule_mobile_ion_network_uniformity(structure):
    try:
        mobile_cutoffs = {
            "H": 2.4,
            "Li": 3.6,
            "Na": 4.1,
            "K": 4.7,
            "Rb": 5.0,
            "Cs": 5.3,
            "Cu": 3.5,
            "Ag": 3.9,
            "Mg": 3.5,
            "Ca": 4.0,
        }
        mobile = _dominant_mobile_symbol(structure, mobile_cutoffs)
        if mobile is None:
            return 0.0

        mobile_indices = [
            index
            for index, site in enumerate(structure)
            if _site_fraction(site, mobile) > 0.05
        ]
        if not mobile_indices:
            return 0.0

        charge_map = _composition_charge_map(structure)
        mobile_set = set(mobile_indices)

        def framework_weight(site):
            weighted_value = 0.0
            total_weight = 0.0
            for specie, occupancy in site.species.items():
                symbol = _species_symbol(specie)
                if symbol == mobile:
                    continue
                amount = max(0.0, float(occupancy))
                if symbol in charge_map:
                    value = abs(charge_map[symbol])
                else:
                    try:
                        electronegativity = float(Element(symbol).X)
                        value = 0.5 + electronegativity / 4.0
                    except Exception:
                        value = 1.0
                weighted_value += amount * value
                total_weight += amount
            return (
                weighted_value / total_weight
                if total_weight > 0.0
                else 0.0
            )

        def environment(index):
            center = structure[index]
            framework = []
            for neighbor in structure.get_neighbors(center, 4.6):
                distance = float(neighbor.nn_distance)
                if distance <= 0.2 or _site_fraction(neighbor, mobile) > 0.05:
                    continue
                framework.append((neighbor, distance))
            if not framework:
                return (0.0, 0.0, 0.0)

            framework.sort(key=lambda item: item[1])
            nearest_distance = framework[0][1]
            shell_limit = min(4.2, nearest_distance + 0.85)
            shell = [
                item for item in framework if item[1] <= shell_limit
            ]
            if not shell:
                return (0.0, 0.0, 0.0)

            weights = []
            for neighbor, _ in shell:
                nonmobile_occupancy = sum(
                    max(0.0, float(occupancy))
                    for specie, occupancy in neighbor.species.items()
                    if _species_symbol(specie) != mobile
                )
                weights.append(max(1e-6, nonmobile_occupancy))

            coordination = sum(weights)
            mean_distance = sum(
                weight * item[1]
                for weight, item in zip(weights, shell)
            ) / coordination
            bond_valence_proxy = sum(
                weight
                * framework_weight(item[0])
                * math.exp((nearest_distance - item[1]) / 0.37)
                for weight, item in zip(weights, shell)
            )
            return (
                float(coordination),
                float(mean_distance),
                float(bond_valence_proxy),
            )

        environments = {
            index: environment(index) for index in mobile_indices
        }
        adjacency = {index: set() for index in mobile_indices}
        path_counts = {index: 0 for index in mobile_indices}
        hop_distances = []
        seen_edges = set()
        cutoff = mobile_cutoffs[mobile]

        for index in mobile_indices:
            for neighbor in structure.get_neighbors(
                structure[index],
                cutoff,
            ):
                target = getattr(neighbor, "index", None)
                distance = float(neighbor.nn_distance)
                if (
                    target is None
                    or target not in mobile_set
                    or distance <= 0.35
                ):
                    continue

                image = tuple(
                    int(round(float(value)))
                    for value in getattr(neighbor, "image", (0, 0, 0))
                )
                forward = (index, target, image)
                reverse = (
                    target,
                    index,
                    tuple(-value for value in image),
                )
                edge_key = min(forward, reverse)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    hop_distances.append(distance)
                path_counts[index] += 1
                if target != index:
                    adjacency[index].add(target)
                    adjacency[target].add(index)

        if not hop_distances:
            return 0.0

        unvisited = set(mobile_indices)
        largest_component = 0
        while unvisited:
            root = unvisited.pop()
            stack = [root]
            component_size = 0
            while stack:
                current = stack.pop()
                component_size += 1
                for target in adjacency[current]:
                    if target in unvisited:
                        unvisited.remove(target)
                        stack.append(target)
            largest_component = max(largest_component, component_size)

        active_fraction = sum(
            count > 0 for count in path_counts.values()
        ) / float(len(mobile_indices))
        connected_fraction = largest_component / float(len(mobile_indices))
        coverage = 0.68 * connected_fraction + 0.32 * active_fraction

        def uniformity(values, reference_cv):
            if not values:
                return 0.0
            mean_value = sum(values) / len(values)
            if abs(mean_value) <= 1e-12:
                return 1.0 if max(abs(value) for value in values) <= 1e-12 else 0.0
            variance = sum(
                (value - mean_value) ** 2 for value in values
            ) / len(values)
            cv = math.sqrt(variance) / abs(mean_value)
            return 1.0 / (1.0 + (cv / reference_cv) ** 2)

        hop_uniformity = uniformity(hop_distances, 0.18)
        coordination_uniformity = uniformity(
            [value[0] for value in environments.values()],
            0.22,
        )
        distance_uniformity = uniformity(
            [value[1] for value in environments.values()],
            0.12,
        )
        bond_proxy_uniformity = uniformity(
            [value[2] for value in environments.values()],
            0.25,
        )
        environment_uniformity = (
            0.34 * coordination_uniformity
            + 0.31 * distance_uniformity
            + 0.35 * bond_proxy_uniformity
        )
        degree_values = [
            float(path_counts[index]) for index in mobile_indices
        ]
        degree_uniformity = uniformity(degree_values, 0.30)
        mean_degree = sum(degree_values) / len(degree_values)
        degree_support = 1.0 - math.exp(-mean_degree / 2.0)

        score = coverage * degree_support * (
            0.40 * hop_uniformity
            + 0.36 * environment_uniformity
            + 0.24 * degree_uniformity
        )
        return float(min(1.0, max(0.0, score)))
    except Exception:
        return 0.0

def rule_r11_migration_bottleneck_radius_match(structure):
    try:
        mobile_radii = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Mg": 0.72, "Ca": 1.00, "Ag": 1.15, "Cu": 0.77}
        framework_radii = {"F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20, "O": 1.40, "S": 1.84, "Se": 1.98, "Te": 2.21, "N": 1.46, "P": 1.80, "B": 0.85, "Si": 1.11, "Al": 0.68}
        amounts = {}
        site_symbols = []
        for site in structure:
            symbols = []
            for sp, occ in site.species.items():
                sym = getattr(sp, "symbol", getattr(getattr(sp, "element", None), "symbol", ""))
                symbols.append((sym, float(occ)))
                if sym in mobile_radii:
                    amounts[sym] = amounts.get(sym, 0.0) + float(occ)
            site_symbols.append(symbols)
        candidates = [x for x in ("Li", "Na", "K", "Ag", "Cu", "Mg", "Ca", "Rb", "Cs") if amounts.get(x, 0.0) > 0.0]
        if not candidates:
            return 0.0
        mobile = max(candidates, key=lambda x: amounts[x] / (1.0 + list(mobile_radii).index(x)))
        ion_radius = mobile_radii[mobile]
        mobile_sites = [i for i, syms in enumerate(site_symbols) if any(s == mobile and o > 0.0 for s, o in syms)]
        framework_sites = [i for i, syms in enumerate(site_symbols) if not any(s == mobile and o > 0.0 for s, o in syms)]
        if not mobile_sites or not framework_sites:
            return 0.0
        def site_radius(index):
            total = 0.0
            weight = 0.0
            for sym, occ in site_symbols[index]:
                r = framework_radii.get(sym)
                if r is None:
                    try:
                        sp = next(sp for sp in structure[index].species if getattr(sp, "symbol", getattr(getattr(sp, "element", None), "symbol", "")) == sym)
                        r = float(getattr(sp, "ionic_radius", None) or getattr(getattr(sp, "element", sp), "atomic_radius", None) or 1.2)
                    except Exception:
                        r = 1.2
                total += occ * r
                weight += occ
            return total / weight if weight else 1.2
        radii = {i: site_radius(i) for i in framework_sites}
        node_scores = []
        for i in mobile_sites:
            fi = structure[i].frac_coords
            edges = []
            for j in mobile_sites:
                fj = structure[j].frac_coords
                for u in (-1, 0, 1):
                    for v in (-1, 0, 1):
                        for w in (-1, 0, 1):
                            if i == j and u == 0 and v == 0 and w == 0:
                                continue
                            delta = [float(fj[0] + u - fi[0]), float(fj[1] + v - fi[1]), float(fj[2] + w - fi[2])]
                            cart = structure.lattice.get_cartesian_coords(delta)
                            dist = float(sum(float(x) * float(x) for x in cart) ** 0.5)
                            if 1.2 < dist <= 8.0:
                                edges.append((dist, j, u, v, w))
            edges.sort(key=lambda x: x[0])
            scores = []
            for dist, j, u, v, w in edges[:8]:
                fj = structure[j].frac_coords
                limiting = None
                limiting_clearances = None
                for t in (0.30, 0.40, 0.50, 0.60, 0.70):
                    point = [float(fi[k] + t * (fj[k] + (u, v, w)[k] - fi[k])) for k in range(3)]
                    clearances = []
                    for q in framework_sites:
                        d, image = structure.lattice.get_distance_and_image(point, structure[q].frac_coords)
                        clearances.append(float(d) - radii[q])
                    clearances.sort()
                    local = clearances[0]
                    if limiting is None or local < limiting:
                        limiting = local
                        limiting_clearances = clearances[:4]
                if limiting is None:
                    continue
                delta = limiting - ion_radius
                width = (0.22 + 0.30 * ion_radius) if delta < 0.0 else (0.32 + 0.45 * ion_radius)
                match = 2.718281828459045 ** (-min(50.0, (delta / width) ** 2))
                vals = limiting_clearances or [limiting]
                mean = sum(vals) / len(vals)
                distortion = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
                shape = 2.718281828459045 ** (-min(50.0, (distortion / (0.35 + 0.2 * ion_radius)) ** 2))
                long_hop = max(0.0, dist - (4.0 + 1.5 * ion_radius))
                connectivity = 2.718281828459045 ** (-min(50.0, (long_hop / 1.5) ** 2))
                scores.append(match * shape * connectivity)
            if scores:
                scores.sort(reverse=True)
                node_scores.append(scores[1] if len(scores) > 1 else scores[0])
        if not node_scores:
            return 0.0
        node_scores.sort()
        n = max(1, (len(node_scores) + 1) // 2)
        return float(sum(node_scores[:n]) / n)
    except Exception:
        return 0.0

def rule_ion_transport_network_quality(structure):
    try:
        priorities = ["Li", "Na", "Ag", "Cu", "K", "Rb", "Cs", "H", "F", "O", "Cl", "Mg", "Ca"]
        cutoffs = {"H": 2.8, "Li": 3.8, "Na": 4.3, "K": 4.8, "Rb": 5.1, "Cs": 5.4, "Ag": 4.2, "Cu": 3.7, "Mg": 3.8, "Ca": 4.3, "F": 3.6, "O": 3.5, "Cl": 4.2}
        def occupancy(site, symbol):
            total = 0.0
            for specie, amount in site.species.items():
                if getattr(specie, "symbol", str(specie)) == symbol:
                    total += float(amount)
            return total
        mobile = None
        for symbol in priorities:
            if any(occupancy(site, symbol) > 1e-8 for site in structure):
                mobile = symbol
                break
        if mobile is None:
            return 0.0
        indices = [i for i, site in enumerate(structure) if occupancy(site, mobile) > 1e-8]
        if not indices:
            return 0.0
        index_set = set(indices)
        cutoff = cutoffs.get(mobile, 4.0)
        adjacency = {i: set() for i in indices}
        directed = []
        destinations = {i: set() for i in indices}
        hop_distances = []
        seen_hops = set()
        for i in indices:
            for neighbor in structure.get_neighbors(structure[i], cutoff):
                j = int(neighbor.index)
                if j not in index_set:
                    continue
                image_raw = getattr(neighbor, "image", (0, 0, 0))
                image = tuple(int(round(float(x))) for x in image_raw)
                if i == j and image == (0, 0, 0):
                    continue
                distance = float(neighbor.nn_distance)
                if distance <= 1e-8:
                    continue
                adjacency[i].add(j)
                destinations[i].add((j, image))
                directed.append((i, j, image))
                reverse = tuple(-x for x in image)
                key = min((i, j, image), (j, i, reverse))
                if key not in seen_hops:
                    seen_hops.add(key)
                    hop_distances.append(distance)
        if not hop_distances:
            return 0.0
        components = []
        unvisited = set(indices)
        while unvisited:
            root = next(iter(unvisited))
            stack = [root]
            component = set()
            while stack:
                i = stack.pop()
                if i in component:
                    continue
                component.add(i)
                unvisited.discard(i)
                stack.extend(j for j in adjacency[i] if j not in component)
            components.append(component)
        largest = max(components, key=lambda c: sum(occupancy(structure[i], mobile) for i in c))
        total_mobile = sum(occupancy(structure[i], mobile) for i in indices)
        connected_mobile = sum(occupancy(structure[i], mobile) for i in largest)
        coverage = connected_mobile / total_mobile if total_mobile > 0 else 0.0
        potentials = {}
        cycle_vectors = []
        for root in largest:
            if root in potentials:
                continue
            potentials[root] = (0, 0, 0)
            stack = [root]
            while stack:
                i = stack.pop()
                for a, j, image in directed:
                    if a != i or j not in largest:
                        continue
                    proposed = tuple(potentials[i][k] + image[k] for k in range(3))
                    if j not in potentials:
                        potentials[j] = proposed
                        stack.append(j)
                    else:
                        cycle = tuple(proposed[k] - potentials[j][k] for k in range(3))
                        if cycle != (0, 0, 0):
                            cycle_vectors.append(cycle)
        basis = []
        for vector in cycle_vectors:
            work = [float(x) for x in vector]
            for b in basis:
                projection = sum(work[k] * b[k] for k in range(3))
                work = [work[k] - projection * b[k] for k in range(3)]
            norm = sum(x * x for x in work) ** 0.5
            if norm > 1e-8:
                basis.append([x / norm for x in work])
            if len(basis) == 3:
                break
        dimensionality = len(basis) / 3.0
        degrees = [len(destinations[i]) for i in largest]
        degree_score = min(1.0, (sum(degrees) / len(degrees)) / 4.0) if degrees else 0.0
        relevant_distances = []
        clearances = []
        framework = [j for j in range(len(structure)) if j not in index_set]
        for i, j, image in directed:
            if i not in largest or j not in largest:
                continue
            d = float(structure.get_distance(i, j, jimage=image))
            relevant_distances.append(d)
            midpoint = [(float(structure[i].frac_coords[k]) + float(structure[j].frac_coords[k]) + image[k]) / 2.0 for k in range(3)]
            if framework:
                nearest = min(float(structure.lattice.get_distance_and_image(midpoint, structure[k].frac_coords)[0]) for k in framework)
                clearances.append(nearest)
        if not relevant_distances:
            relevant_distances = hop_distances
        mean_distance = sum(relevant_distances) / len(relevant_distances)
        variance = sum((d - mean_distance) ** 2 for d in relevant_distances) / len(relevant_distances)
        regularity = 1.0 / (1.0 + (variance ** 0.5 / max(mean_distance, 1e-8)) ** 2 * 16.0)
        short_hop_score = 1.0 / (1.0 + (mean_distance / (0.72 * cutoff)) ** 4)
        if clearances:
            mean_clearance = sum(clearances) / len(clearances)
            accessibility = max(0.0, min(1.0, (mean_clearance - 0.65) / 1.35))
        else:
            accessibility = 0.5
        environments = []
        for i in largest:
            value = 0.0
            coordination = 0.0
            for neighbor in structure.get_neighbors(structure[i], 4.5):
                if int(neighbor.index) in index_set:
                    continue
                d = max(float(neighbor.nn_distance), 0.5)
                for specie, amount in neighbor.species.items():
                    electronegativity = getattr(specie, "X", None)
                    if electronegativity is not None:
                        value += float(amount) * float(electronegativity) / d
                    coordination += float(amount) / d
            environments.append(value + 0.25 * coordination)
        if environments:
            env_mean = sum(environments) / len(environments)
            env_std = (sum((x - env_mean) ** 2 for x in environments) / len(environments)) ** 0.5
            similarity = 1.0 / (1.0 + 2.0 * env_std / (abs(env_mean) + 1.0))
        else:
            similarity = 0.5
        vacancy = sum(max(0.0, 1.0 - occupancy(structure[i], mobile)) for i in indices) / len(indices)
        vacancy_factor = 0.35 + 0.65 * min(1.0, vacancy ** 0.5)
        score = coverage * dimensionality * degree_score * regularity * short_hop_score * accessibility * similarity * vacancy_factor
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0

def rule_r13_percolating_mobile_ion_free_volume(structure):
    try:
        radius = {"H": 0.25, "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Cu": 0.77, "Ag": 1.15, "Mg": 0.72, "Ca": 1.00, "Zn": 0.74, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20, "O": 1.40, "S": 1.84, "N": 1.46, "P": 1.80, "Si": 1.11, "Al": 1.21, "B": 0.84, "Ti": 1.36, "Zr": 1.48, "La": 1.69, "Ge": 1.20, "Sn": 1.40}
        mobile_order = ["Li", "Na", "K", "H", "Ag", "Cu", "Mg", "Ca", "Rb", "Cs", "Zn"]
        present = set()
        site_symbols = []
        for site in structure:
            symbols = []
            for sp in site.species:
                sym = getattr(getattr(sp, "element", sp), "symbol", str(sp))
                symbols.append(sym)
                present.add(sym)
            site_symbols.append(symbols)
        mobile = next((e for e in mobile_order if e in present), None)
        if mobile is None:
            return 0.0
        probe = float(radius.get(mobile, 0.9))
        framework = []
        for site, symbols in zip(structure, site_symbols):
            nonmobile = [s for s in symbols if s != mobile]
            if nonmobile:
                r = max(float(radius.get(s, 1.35)) for s in nonmobile)
                framework.append((tuple(float(x) for x in site.frac_coords), r))
        if not framework:
            return 0.0
        lengths = structure.lattice.abc
        dims = tuple(max(8, min(16, int(round(float(x) / 1.1)))) for x in lengths)
        nx, ny, nz = dims
        total = nx * ny * nz
        clearance = [0.0] * total
        for ix in range(nx):
            fx = (ix + 0.5) / nx
            for iy in range(ny):
                fy = (iy + 0.5) / ny
                for iz in range(nz):
                    fz = (iz + 0.5) / nz
                    best = 1.0e30
                    for af, ar in framework:
                        dx = fx - af[0]
                        dy = fy - af[1]
                        dz = fz - af[2]
                        dx -= round(dx)
                        dy -= round(dy)
                        dz -= round(dz)
                        c = structure.lattice.get_cartesian_coords((dx, dy, dz))
                        d = (float(c[0]) ** 2 + float(c[1]) ** 2 + float(c[2]) ** 2) ** 0.5 - ar
                        if d < best:
                            best = d
                    clearance[(ix * ny + iy) * nz + iz] = best
        def channel_score(threshold):
            accessible = [v >= threshold for v in clearance]
            visited = set()
            best_score = 0.0
            best_mean = 0.0
            for start in range(total):
                if not accessible[start] or start in visited:
                    continue
                sx = start // (ny * nz)
                rem = start % (ny * nz)
                sy = rem // nz
                sz = rem % nz
                queue = [start]
                visited.add(start)
                lifts = {start: (sx, sy, sz)}
                component = []
                winding = set()
                head = 0
                while head < len(queue):
                    idx = queue[head]
                    head += 1
                    component.append(idx)
                    x = idx // (ny * nz)
                    rem2 = idx % (ny * nz)
                    y = rem2 // nz
                    z = rem2 % nz
                    ux, uy, uz = lifts[idx]
                    for ax, step in ((0, -1), (0, 1), (1, -1), (1, 1), (2, -1), (2, 1)):
                        xx, yy, zz = x, y, z
                        lu = [ux, uy, uz]
                        if ax == 0:
                            xx = (x + step) % nx
                        elif ax == 1:
                            yy = (y + step) % ny
                        else:
                            zz = (z + step) % nz
                        lu[ax] += step
                        nb = (xx * ny + yy) * nz + zz
                        if not accessible[nb]:
                            continue
                        proposed = tuple(lu)
                        if nb not in lifts:
                            lifts[nb] = proposed
                            visited.add(nb)
                            queue.append(nb)
                        else:
                            old = lifts[nb]
                            periods = (nx, ny, nz)
                            for a in range(3):
                                if abs(proposed[a] - old[a]) >= periods[a]:
                                    winding.add(a)
                if winding:
                    s = (len(component) / float(total)) * (len(winding) / 3.0)
                    if s > best_score:
                        best_score = s
                        best_mean = sum(clearance[i] for i in component) / len(component)
            return best_score, best_mean
        scores = []
        means = []
        for factor in (0.8, 1.0, 1.2):
            s, m = channel_score(probe * factor)
            scores.append(s)
            if s > 0.0:
                means.append(m)
        if not scores or max(scores) <= 0.0:
            return 0.0
        persistence_score = sum(scores) / len(scores)
        mean_clearance = sum(means) / len(means) if means else probe
        excess = max(0.0, mean_clearance / max(probe, 1.0e-8) - 2.5)
        cavity_penalty = 1.0 / (1.0 + 0.25 * excess)
        return float(persistence_score * cavity_penalty)
    except Exception:
        return 0.0

def rule_r14_anion_softness_weak_coordination(structure):
    try:
        mobile_radii = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Ag": 1.15, "Cu": 0.77}
        anion_data = {
            "H": (0.67, 1.54, 2.20), "N": (3.00, 1.46, 3.04),
            "O": (2.00, 1.40, 3.44), "F": (1.04, 1.33, 3.98),
            "S": (5.30, 1.84, 2.58), "Cl": (3.66, 1.81, 3.16),
            "Se": (6.00, 1.98, 2.55), "Br": (4.77, 1.96, 2.96),
            "Te": (8.30, 2.21, 2.10), "I": (7.10, 2.20, 2.66)
        }
        def symbol_and_occupancy(site):
            try:
                return site.specie.symbol, 1.0
            except Exception:
                items = list(site.species.items())
                if not items:
                    return "", 0.0
                specie, occupancy = max(items, key=lambda x: x[1])
                return getattr(specie, "symbol", getattr(getattr(specie, "element", None), "symbol", "")), float(occupancy)
        site_info = [symbol_and_occupancy(site) for site in structure]
        mobile_indices = [i for i, (s, o) in enumerate(site_info) if s in mobile_radii and o > 0]
        anion_indices = [i for i, (s, o) in enumerate(site_info) if s in anion_data and o > 0]
        if not mobile_indices or not anion_indices:
            return 0.0
        distances = structure.distance_matrix
        scores = []
        for i in mobile_indices:
            ms, mo = site_info[i]
            neighbors = []
            for j in anion_indices:
                if i == j:
                    continue
                d = float(distances[i][j])
                if d > 0.1:
                    neighbors.append((d, j))
            if not neighbors:
                continue
            dmin = min(x[0] for x in neighbors)
            cutoff = min(dmin + 0.75, dmin * 1.30)
            shell = [(d, j) for d, j in neighbors if d <= cutoff]
            if not shell:
                continue
            cn = sum(site_info[j][1] for d, j in shell)
            weighted_softness = 0.0
            weighted_separation = 0.0
            weight_sum = 0.0
            for d, j in shell:
                symbol, occupancy = site_info[j]
                polarizability, radius, electronegativity = anion_data[symbol]
                weight = occupancy / max(d * d, 1.0e-8)
                weighted_softness += weight * polarizability * radius / electronegativity
                weighted_separation += weight * d / (mobile_radii[ms] + radius)
                weight_sum += weight
            if weight_sum > 0 and cn > 0:
                softness = weighted_softness / weight_sum
                weak_coordination = (weighted_separation / weight_sum) / (cn ** 0.5)
                scores.append(mo * softness * weak_coordination)
        return float(sum(scores) / len(scores)) if scores else 0.0
    except Exception:
        return 0.0

def rule_mobile_ion_pathway_uniformity(structure):
    try:
        def symbol_of(site):
            try:
                return site.specie.symbol
            except Exception:
                items = list(site.species.items())
                return max(items, key=lambda x: x[1])[0].symbol if items else ""

        def mean(values):
            return sum(values) / len(values) if values else 0.0

        def cv(values, floor=0.1):
            if len(values) < 2:
                return 0.0
            m = mean(values)
            s = (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5
            return s / max(abs(m), floor)

        primary = {"Li", "Na", "K", "Rb", "Cs"}
        secondary = {"H", "Cu", "Ag", "Mg", "Ca", "Zn"}
        present = {symbol_of(site) for site in structure}
        mobile = present.intersection(primary)
        if not mobile:
            mobile = present.intersection(secondary)
        if not mobile:
            return 0.0

        environments = []
        pathway_degrees = []
        hop_cutoffs = {"H": 2.8, "Li": 4.2, "Na": 4.8, "K": 5.4, "Rb": 5.7, "Cs": 6.0,
                       "Cu": 4.0, "Ag": 4.5, "Mg": 4.2, "Ca": 4.8, "Zn": 4.0}

        for site in structure:
            msym = symbol_of(site)
            if msym not in mobile:
                continue
            neighbors = list(structure.get_neighbors(site, 6.0))
            framework = [(n, float(n.nn_distance)) for n in neighbors
                         if symbol_of(n) not in mobile and float(n.nn_distance) > 0.1]
            framework.sort(key=lambda x: x[1])
            if not framework:
                continue
            d0 = framework[0][1]
            cutoff = min(3.8, max(d0 * 1.42, d0 + 0.35))
            shell = [(n, d) for n, d in framework if d <= cutoff]
            if not shell:
                continue
            distances = [d for _, d in shell]
            md = mean(distances)
            radial_cv = ((sum((d - md) ** 2 for d in distances) / len(distances)) ** 0.5) / max(md, 0.1)
            cosines = []
            for i in range(len(shell)):
                vi = [float(shell[i][0].coords[k] - site.coords[k]) for k in range(3)]
                ni = sum(x * x for x in vi) ** 0.5
                for j in range(i + 1, len(shell)):
                    vj = [float(shell[j][0].coords[k] - site.coords[k]) for k in range(3)]
                    nj = sum(x * x for x in vj) ** 0.5
                    if ni > 0 and nj > 0:
                        c = sum(vi[k] * vj[k] for k in range(3)) / (ni * nj)
                        cosines.append(max(-1.0, min(1.0, c)))
            angular_mean = mean(cosines)
            angular_second = mean([c * c for c in cosines])
            try:
                mobile_en = float(site.specie.X or 0.0)
            except Exception:
                mobile_en = 0.0
            strengths = []
            for n, d in shell:
                try:
                    nen = float(n.specie.X or 0.0)
                except Exception:
                    nen = 0.0
                strengths.append(max(0.0, nen - mobile_en) / (d * d))
            compression = max(0.0, 0.82 - min(distances) / max(md, 0.1)) / 0.82
            environments.append((float(len(shell)), md, radial_cv, angular_mean,
                                 angular_second, sum(strengths), compression))
            hop = hop_cutoffs.get(msym, 4.5)
            degree = sum(1 for n in neighbors
                         if symbol_of(n) in mobile and 0.2 < float(n.nn_distance) <= hop)
            pathway_degrees.append(float(degree))

        if not environments:
            return 0.0
        columns = [[e[i] for e in environments] for i in range(7)]
        mismatch = (1.4 * cv(columns[0], 1.0) + 1.2 * cv(columns[1], 0.5) +
                    0.8 * cv(columns[2], 0.05) + 0.5 * cv(columns[3], 0.25) +
                    0.5 * cv(columns[4], 0.25) + 1.2 * cv(columns[5], 0.2))
        uniformity = 1.0 / (1.0 + mismatch)
        avg_cn = mean(columns[0])
        undercoordination = max(0.0, 3.0 - avg_cn) / 3.0
        distortion = mean(columns[2])
        compression = mean(columns[6])
        trapping = max(0.0, mean(columns[5]) / max(avg_cn, 1.0) - 0.8)
        flexibility = 1.0 / (1.0 + undercoordination + 1.5 * distortion + 2.0 * compression + 0.5 * trapping)
        avg_degree = mean(pathway_degrees)
        connectivity = avg_degree / (avg_degree + 2.0) if avg_degree > 0 else 0.0
        connectivity *= 1.0 / (1.0 + cv(pathway_degrees, 1.0))
        score = uniformity * flexibility * (0.5 + 0.5 * connectivity)
        return float(max(0.0, min(1.0, score))) if score == score else 0.0
    except Exception:
        return 0.0

def rule_r16_three_dimensional_mobile_network(structure):
    try:
        mobile_order = ("Li", "Na", "K", "Rb", "Cs", "Ag", "H", "Mg", "Ca", "F")
        amounts = {x: 0.0 for x in mobile_order}
        for site in structure:
            for sp, occ in site.species.items():
                sym = getattr(sp, "symbol", str(sp).split("+")[0].split("-")[0])
                if sym in amounts:
                    amounts[sym] += float(occ)
        mobile = next((x for x in mobile_order if amounts[x] > 1e-8), None)
        if mobile is None:
            return 0.0
        hop = {"H": 2.2, "Li": 3.0, "Na": 3.4, "K": 3.8, "Rb": 4.0, "Cs": 4.2, "Ag": 3.3, "Mg": 3.0, "Ca": 3.4, "F": 3.0}.get(mobile, 3.2)
        mobile_radius = {"H": 0.35, "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Ag": 1.15, "Mg": 0.72, "Ca": 1.00, "F": 1.33}.get(mobile, 0.9)
        radii = {"O": 1.40, "S": 1.84, "Se": 1.98, "Te": 2.21, "F": 1.33, "Cl": 1.81, "Br": 1.96, "I": 2.20, "N": 1.46, "P": 1.10, "B": 0.85, "Si": 1.11, "Al": 1.18}
        nodes = []
        mobile_flags = []
        occupancies = []
        framework = []
        for site in structure:
            mo = 0.0
            host_r = 0.0
            host_present = False
            for sp, occ in site.species.items():
                sym = getattr(sp, "symbol", str(sp).split("+")[0].split("-")[0])
                if sym == mobile:
                    mo += float(occ)
                else:
                    host_present = True
                    host_r = max(host_r, radii.get(sym, 1.05))
            if mo > 1e-8:
                nodes.append(tuple(float(x) for x in site.frac_coords))
                mobile_flags.append(True)
                occupancies.append(min(1.0, mo))
            if host_present:
                framework.append((tuple(float(x) for x in site.frac_coords), host_r))
        if not nodes:
            return 0.0
        original_mobile_count = len(nodes)
        a, b, c = structure.lattice.abc
        dims = [max(3, min(8, int(round(x / 1.7)))) for x in (a, b, c)]
        clearance_scale = 0.78
        for i in range(dims[0]):
            for j in range(dims[1]):
                for k in range(dims[2]):
                    f = ((i + 0.5) / dims[0], (j + 0.5) / dims[1], (k + 0.5) / dims[2])
                    if any(structure.lattice.get_distance_and_image(f, q)[0] < 0.55 for q in nodes[:original_mobile_count]):
                        continue
                    accessible = True
                    for q, r in framework:
                        if structure.lattice.get_distance_and_image(f, q)[0] < clearance_scale * (r + mobile_radius):
                            accessible = False
                            break
                    if accessible:
                        nodes.append(f)
                        mobile_flags.append(False)
                        occupancies.append(0.0)
        n = len(nodes)
        adjacency = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d, image = structure.lattice.get_distance_and_image(nodes[i], nodes[j])
                if d > hop or d < 1e-7:
                    continue
                img = tuple(int(round(float(x))) for x in image)
                delta = tuple(nodes[j][q] + img[q] - nodes[i][q] for q in range(3))
                clear = True
                for t in (0.25, 0.5, 0.75):
                    p = tuple(nodes[i][q] + t * delta[q] for q in range(3))
                    for fq, r in framework:
                        if structure.lattice.get_distance_and_image(p, fq)[0] < 0.70 * (r + mobile_radius):
                            clear = False
                            break
                    if not clear:
                        break
                if clear:
                    adjacency[i].append((j, img))
                    adjacency[j].append((i, tuple(-x for x in img)))
        total_mobile_weight = sum(occupancies[:original_mobile_count])
        if total_mobile_weight <= 0:
            return 0.0
        visited = set()
        best = 0.0
        for root in range(n):
            if root in visited:
                continue
            stack = [root]
            shifts = {root: (0, 0, 0)}
            component = []
            cycles = []
            visited.add(root)
            while stack:
                u = stack.pop()
                component.append(u)
                su = shifts[u]
                for v, img in adjacency[u]:
                    proposed = tuple(su[q] + img[q] for q in range(3))
                    if v not in shifts:
                        shifts[v] = proposed
                        visited.add(v)
                        stack.append(v)
                    else:
                        cyc = tuple(proposed[q] - shifts[v][q] for q in range(3))
                        if cyc != (0, 0, 0):
                            cycles.append([float(x) for x in cyc])
            mobile_weight = sum(occupancies[x] for x in component if x < original_mobile_count)
            if mobile_weight <= 0:
                continue
            basis = []
            for vec in cycles:
                w = vec[:]
                for pivot, b in basis:
                    if abs(w[pivot]) > 1e-10:
                        factor = w[pivot] / b[pivot]
                        w = [w[q] - factor * b[q] for q in range(3)]
                pivot = next((q for q in range(3) if abs(w[q]) > 1e-8), None)
                if pivot is not None:
                    scale = w[pivot]
                    w = [x / scale for x in w]
                    basis.append((pivot, w))
                    basis.sort(key=lambda x: x[0])
                if len(basis) == 3:
                    break
            dimensionality = float(len(basis))
            coverage = mobile_weight / total_mobile_weight
            comp_mobile = [occupancies[x] for x in component if x < original_mobile_count]
            partiality = sum(1.0 - x for x in comp_mobile) / len(comp_mobile) if comp_mobile else 0.0
            void_fraction = sum(1 for x in component if not mobile_flags[x]) / max(1.0, float(len(component)))
            accessibility = 0.50 + 0.30 * partiality + 0.20 * min(1.0, 2.0 * void_fraction)
            best = max(best, dimensionality * coverage * accessibility)
        return float(best)
    except Exception:
        return 0.0

def rule_mobile_ion_partial_occupancy_network(structure):
    try:
        volume = float(structure.volume)
        if volume <= 0:
            return 0.0
        mobile = {"H", "Li", "Na", "K", "Rb", "Cs", "Cu", "Ag", "Mg", "Ca", "Zn", "F", "Cl", "Br", "I", "O"}
        by_element = {}
        for i, site in enumerate(structure):
            for specie, occupancy in site.species.items():
                symbol = getattr(specie, "symbol", str(specie))
                if symbol in mobile:
                    p = max(0.0, min(1.0, float(occupancy)))
                    if p > 0:
                        by_element.setdefault(symbol, []).append((i, p))
        best = 0.0
        for entries in by_element.values():
            occupancies = [p for _, p in entries]
            carrier_density = sum(occupancies) / volume
            vacancy_fraction = sum(1.0 - p for p in occupancies) / len(occupancies)
            balance = 4.0 * sum(p * (1.0 - p) for p in occupancies) / len(occupancies)
            mean_p = sum(occupancies) / len(occupancies)
            variance = sum((p - mean_p) ** 2 for p in occupancies) / len(occupancies)
            disorder_factor = 1.0 / (1.0 + 4.0 * variance)
            indices = [i for i, _ in entries]
            nearest = []
            if len(indices) > 1:
                for i in indices:
                    distances = [float(structure.get_distance(i, j)) for j in indices if j != i]
                    if distances:
                        nearest.append(min(distances))
            else:
                nearest.append(min(float(x) for x in structure.lattice.abc))
            connectivity = 0.0
            for d in nearest:
                path_factor = 1.0 / (1.0 + (d / 4.0) ** 4)
                crowding_factor = min(1.0, d / 1.8)
                connectivity += path_factor * crowding_factor
            connectivity = connectivity / len(nearest) if nearest else 0.0
            score = carrier_density * vacancy_fraction * balance * disorder_factor * connectivity
            if score > best:
                best = score
        return float(best)
    except Exception:
        return 0.0

def rule_3d_mobile_ion_network(structure):
    try:
        mobile_order = ("H", "Li", "Na", "K", "Rb", "Cs", "Ag", "Cu")
        site_symbols = []
        for site in structure:
            symbols = set()
            try:
                for sp in site.species:
                    symbols.add(str(getattr(sp, "symbol", sp)))
            except Exception:
                try:
                    symbols.add(str(site.specie.symbol))
                except Exception:
                    pass
            site_symbols.append(symbols)
        present = [s for s in mobile_order if any(s in x for x in site_symbols)]
        if not present:
            return 0.0
        centers, neighbors, offsets, distances = structure.get_neighbor_list(6.5)
        def vector_rank(vectors):
            vectors = [v for v in vectors if v != (0, 0, 0)]
            if not vectors:
                return 0
            a = vectors[0]
            b = None
            for v in vectors[1:]:
                cross = (a[1]*v[2]-a[2]*v[1], a[2]*v[0]-a[0]*v[2], a[0]*v[1]-a[1]*v[0])
                if cross != (0, 0, 0):
                    b = v
                    break
            if b is None:
                return 1
            for v in vectors:
                det = a[0]*(b[1]*v[2]-b[2]*v[1])-a[1]*(b[0]*v[2]-b[2]*v[0])+a[2]*(b[0]*v[1]-b[1]*v[0])
                if det != 0:
                    return 3
            return 2
        best = 0.0
        for symbol in present:
            nodes = [i for i, syms in enumerate(site_symbols) if symbol in syms]
            node_set = set(nodes)
            if not nodes:
                continue
            nearest = {i: 1e9 for i in nodes}
            candidate_edges = []
            for i, j, off, dist in zip(centers, neighbors, offsets, distances):
                i, j, dist = int(i), int(j), float(dist)
                if i in node_set and j in node_set and dist > 0.5:
                    o = tuple(int(round(float(x))) for x in off)
                    candidate_edges.append((i, j, o, dist))
                    if dist < nearest[i]:
                        nearest[i] = dist
            first_hops = sorted(v for v in nearest.values() if v < 1e8)
            if not first_hops:
                continue
            typical = first_hops[len(first_hops)//2]
            size_cap = 5.8 if symbol in ("K", "Rb", "Cs") else 5.2
            cutoff = min(size_cap, max(2.8, 1.35 * typical))
            graph = {i: set() for i in nodes}
            for i, j, o, dist in candidate_edges:
                if dist <= cutoff:
                    graph[i].add((j, o))
                    graph[j].add((i, (-o[0], -o[1], -o[2])))
            seen = set()
            for start in nodes:
                if start in seen:
                    continue
                stack = [start]
                potential = {start: (0, 0, 0)}
                component = []
                cycles = []
                seen.add(start)
                while stack:
                    i = stack.pop()
                    component.append(i)
                    pi = potential[i]
                    for j, o in graph[i]:
                        proposed = (pi[0]+o[0], pi[1]+o[1], pi[2]+o[2])
                        if j not in potential:
                            potential[j] = proposed
                            seen.add(j)
                            stack.append(j)
                        else:
                            pj = potential[j]
                            cycles.append((proposed[0]-pj[0], proposed[1]-pj[1], proposed[2]-pj[2]))
                rank = vector_rank(cycles)
                coverage = float(len(component)) / float(len(nodes))
                route_quality = sum(min(len(graph[i]), 4) / 4.0 for i in component) / float(len(component))
                score = (rank / 3.0) * coverage * route_quality
                if score > best:
                    best = score
        return float(best)
    except Exception:
        return 0.0

def rule_r19_mobile_ion_sublattice_disorder(structure):
    try:
        mobile = {"H", "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Cu", "Ag", "F", "O", "Cl"}

        def symbol_of(specie):
            try:
                return specie.symbol
            except Exception:
                try:
                    return specie.element.symbol
                except Exception:
                    return str(specie)

        entries = []
        for i, site in enumerate(structure):
            amounts = []
            mobile_occupancy = 0.0
            total_occupancy = 0.0
            for specie, occupancy in site.species.items():
                occupancy = float(occupancy)
                total_occupancy += occupancy
                if symbol_of(specie) in mobile:
                    mobile_occupancy += occupancy
                    amounts.append(occupancy)
            if mobile_occupancy <= 1e-8:
                continue
            residual = max(0.0, 1.0 - sum(amounts))
            probabilities = amounts + ([residual] if residual > 1e-8 else [])
            disorder = min(1.0, 2.0 * max(0.0, 1.0 - sum(x * x for x in probabilities)))
            framework_count = 0.0
            framework_distance = 0.0
            framework_en = 0.0
            en_weight = 0.0
            try:
                neighbors = structure.get_neighbors(site, 3.8)
            except Exception:
                neighbors = []
            for neighbor in neighbors:
                distance = float(getattr(neighbor, "nn_distance", 0.0))
                for specie, occupancy in neighbor.species.items():
                    if symbol_of(specie) in mobile:
                        continue
                    occupancy = float(occupancy)
                    framework_count += occupancy
                    framework_distance += occupancy * distance
                    try:
                        en = specie.element.X if hasattr(specie, "element") else specie.X
                        if en is not None:
                            framework_en += occupancy * float(en)
                            en_weight += occupancy
                    except Exception:
                        pass
            mean_distance = framework_distance / framework_count if framework_count > 0 else 0.0
            mean_en = framework_en / en_weight if en_weight > 0 else 0.0
            entries.append((i, mobile_occupancy, disorder, framework_count, mean_distance, mean_en))

        if len(entries) < 2:
            return 0.0

        weighted_score = 0.0
        total_weight = 0.0
        occupancies = []
        for a, entry in enumerate(entries):
            i, occupancy, disorder, count, mean_distance, mean_en = entry
            edge_sum = 0.0
            degree = 0
            for b, other in enumerate(entries):
                if a == b:
                    continue
                j, _, _, other_count, other_distance, other_en = other
                distance = float(structure.get_distance(i, j))
                if distance <= 0.15 or distance > 4.5:
                    continue
                equivalence = 1.0 / (1.0 + 0.30 * abs(count - other_count) + abs(mean_distance - other_distance) + 0.45 * abs(mean_en - other_en))
                proximity = 1.0 / (1.0 + abs(distance - 2.7) / 1.8)
                edge_sum += equivalence * proximity
                degree += 1
            if degree > 0:
                connectivity = (degree / (degree + 1.0)) * (edge_sum / degree)
                weighted_score += occupancy * disorder * connectivity
                total_weight += occupancy
            occupancies.append(occupancy)

        if total_weight <= 0.0:
            return 0.0
        mean_occupancy = sum(occupancies) / len(occupancies)
        occupancy_variance = sum((x - mean_occupancy) ** 2 for x in occupancies) / len(occupancies)
        heterogeneity_penalty = 1.0 / (1.0 + 4.0 * occupancy_variance)
        value = (weighted_score / total_weight) * heterogeneity_penalty
        return float(value) if value == value and abs(value) != float("inf") else 0.0
    except Exception:
        return 0.0

def rule_r20_bond_valence_percolation(structure):
    try:
        mobile_valence = {"Li": 1.0, "Na": 1.0, "K": 1.0, "Rb": 1.0, "Cs": 1.0, "Ag": 1.0, "Cu": 1.0, "Mg": 2.0, "Ca": 2.0}
        oxygen_r0 = {"Li": 1.466, "Na": 1.803, "K": 2.132, "Rb": 2.263, "Cs": 2.417, "Ag": 1.805, "Cu": 1.593, "Mg": 1.693, "Ca": 1.967}
        anion_offset = {"F": -0.10, "O": 0.0, "N": 0.10, "S": 0.38, "Se": 0.50, "Cl": 0.42, "Br": 0.55, "I": 0.72}
        amounts = structure.composition.get_el_amt_dict()
        candidates = [e for e in mobile_valence if amounts.get(e, 0.0) > 0.0]
        if not candidates:
            return 0.0
        mobile = max(candidates, key=lambda e: amounts.get(e, 0.0))

        def species_items(site):
            try:
                return [(sp.symbol, float(occ)) for sp, occ in site.species.items()]
            except Exception:
                try:
                    return [(site.specie.symbol, 1.0)]
                except Exception:
                    return []

        mobile_indices = []
        for i, site in enumerate(structure):
            if any(sym == mobile and occ > 0.0 for sym, occ in species_items(site)):
                mobile_indices.append(i)
        if not mobile_indices:
            return 0.0

        bvs = {}
        for i in mobile_indices:
            total = 0.0
            for neighbor in structure.get_neighbors(structure[i], 4.2):
                try:
                    distance = float(neighbor.nn_distance)
                except Exception:
                    distance = float(structure[i].distance(neighbor))
                if distance <= 0.0:
                    continue
                for symbol, occupancy in species_items(neighbor):
                    if symbol in anion_offset:
                        r0 = oxygen_r0[mobile] + anion_offset[symbol]
                        total += occupancy * (2.718281828459045 ** ((r0 - distance) / 0.37))
            bvs[i] = total

        target = mobile_valence[mobile]
        mismatch = [abs(bvs[i] - target) / target for i in mobile_indices]
        energy = {i: ((bvs[i] - target) / target) ** 2 for i in mobile_indices}
        mobile_set = set(mobile_indices)
        degrees = {i: 0 for i in mobile_indices}
        smoothness = []
        translations = []

        for i in mobile_indices:
            for neighbor in structure.get_neighbors(structure[i], 4.5):
                try:
                    j = int(neighbor.index)
                except Exception:
                    continue
                if j not in mobile_set:
                    continue
                degrees[i] += 1
                smoothness.append(abs(energy[i] - energy[j]))
                try:
                    image = tuple(int(round(x)) for x in neighbor.image)
                except Exception:
                    try:
                        image = tuple(int(round(neighbor.frac_coords[k] - structure[j].frac_coords[k])) for k in range(3))
                    except Exception:
                        image = (0, 0, 0)
                if image != (0, 0, 0):
                    translations.append(image)

        rank = 0
        if translations:
            rank = 1
            cross_vectors = []
            for a in translations:
                for b in translations:
                    c = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
                    if c != (0, 0, 0):
                        rank = 2
                        cross_vectors.append(c)
            for c in cross_vectors:
                if any(abs(c[0] * v[0] + c[1] * v[1] + c[2] * v[2]) > 0 for v in translations):
                    rank = 3
                    break

        connected_fraction = sum(1 for i in mobile_indices if degrees[i] > 0) / float(len(mobile_indices))
        mean_degree = sum(degrees.values()) / float(len(mobile_indices))
        connectivity = connected_fraction * min(1.0, mean_degree / 2.0) * (0.4 + 0.2 * rank)
        mean_mismatch = sum(mismatch) / float(len(mismatch))
        worst_mismatch = max(mismatch)
        mean_smoothness = sum(smoothness) / float(len(smoothness)) if smoothness else worst_mismatch
        penalty = mean_mismatch + 0.5 * worst_mismatch + mean_smoothness
        return float(connectivity * (2.718281828459045 ** (-penalty)))
    except Exception:
        return 0.0

def rule_r21_mobile_ion_vacancy_connectivity(structure):
    try:
        mobile_cutoffs = {"H": 3.2, "Li": 4.0, "Na": 4.3, "K": 4.7, "Rb": 5.0, "Cs": 5.2, "Ag": 4.3, "Cu": 3.8, "Mg": 4.0, "Ca": 4.5, "Zn": 3.8, "O": 3.6, "F": 3.8, "Cl": 4.3}
        volume = float(structure.volume)
        if volume <= 0 or len(structure) == 0:
            return 0.0
        present = set()
        for site in structure:
            for specie, amount in site.species.items():
                symbol = getattr(specie, "symbol", str(specie))
                if symbol in mobile_cutoffs and float(amount) > 0:
                    present.add(symbol)
        best = 0.0
        for symbol in present:
            indices = []
            carriers = 0.0
            vacancies = 0.0
            for i, site in enumerate(structure):
                mobile_occ = 0.0
                total_occ = 0.0
                for specie, amount in site.species.items():
                    amount = float(amount)
                    total_occ += amount
                    if getattr(specie, "symbol", str(specie)) == symbol:
                        mobile_occ += amount
                if mobile_occ > 1e-10:
                    indices.append(i)
                    carriers += min(1.0, mobile_occ)
                    vacancies += max(0.0, 1.0 - min(1.0, total_occ))
            n = len(indices)
            if n == 0 or carriers <= 0 or vacancies <= 0:
                continue
            balance = 4.0 * carriers * vacancies / float(n * n)
            balance = max(0.0, min(1.0, balance))
            position = {idx: j for j, idx in enumerate(indices)}
            adjacency = [set() for _ in indices]
            active = set()
            cutoff = mobile_cutoffs[symbol]
            for j, idx in enumerate(indices):
                try:
                    neighbors = structure.get_neighbors(structure[idx], cutoff)
                except Exception:
                    neighbors = []
                for neighbor in neighbors:
                    neighbor_occ = 0.0
                    for specie, amount in neighbor.species.items():
                        if getattr(specie, "symbol", str(specie)) == symbol:
                            neighbor_occ += float(amount)
                    if neighbor_occ <= 1e-10 or float(neighbor.nn_distance) <= 1e-6:
                        continue
                    other_idx = getattr(neighbor, "index", None)
                    if other_idx in position:
                        k = position[other_idx]
                        adjacency[j].add(k)
                        adjacency[k].add(j)
                        active.add(j)
                        active.add(k)
                    else:
                        active.add(j)
            visited = set()
            largest = 0
            for start in range(n):
                if start in visited:
                    continue
                stack = [start]
                visited.add(start)
                size = 0
                while stack:
                    node = stack.pop()
                    size += 1
                    for nxt in adjacency[node]:
                        if nxt not in visited:
                            visited.add(nxt)
                            stack.append(nxt)
                largest = max(largest, size)
            connectivity = (largest / float(n)) * (len(active) / float(n))
            carrier_density = carriers / volume
            vacancy_density = vacancies / volume
            density_factor = min(1.0, (carrier_density * vacancy_density) ** 0.5 / 0.01)
            score = balance * connectivity * density_factor
            if score > best:
                best = score
        return float(best)
    except Exception:
        return 0.0

def rule_r22_mobile_ion_displacement_pathway(structure):
    try:
        mobile_data = {
            "H": (2.2, 0.38),
            "Li": (3.3, 0.76),
            "Na": (3.8, 1.02),
            "K": (4.5, 1.38),
            "Rb": (4.8, 1.52),
            "Cs": (5.1, 1.67),
            "Cu": (3.2, 0.77),
            "Ag": (3.6, 1.15),
        }
        mobile = _dominant_mobile_symbol(structure, mobile_data)
        if mobile is None:
            return 0.0

        cutoff, mobile_radius = mobile_data[mobile]
        mobile_indices = [
            index
            for index, site in enumerate(structure)
            if _site_fraction(site, mobile) > 0.02
        ]
        if not mobile_indices:
            return 0.0

        mobile_set = set(mobile_indices)
        position = {
            index: offset for offset, index in enumerate(mobile_indices)
        }

        def clearance_at(cartesian_point):
            nearest_surface = float("inf")
            for neighbor in structure.get_sites_in_sphere(
                cartesian_point,
                5.0,
            ):
                framework_radius = _site_mean_radius(
                    neighbor,
                    excluded_symbol=mobile,
                )
                if framework_radius is None:
                    continue
                surface_distance = (
                    float(neighbor.nn_distance) - framework_radius
                )
                nearest_surface = min(
                    nearest_surface,
                    surface_distance,
                )
            if not math.isfinite(nearest_surface):
                return 0.0
            return nearest_surface

        cage_openness = {}
        for index in mobile_indices:
            clearance = clearance_at(structure[index].coords)
            cage_openness[index] = min(
                1.0,
                max(
                    0.0,
                    (clearance - 0.20 * mobile_radius)
                    / max(0.35, 1.45 * mobile_radius),
                ),
            )

        adjacency = [set() for _ in mobile_indices]
        active = set()
        path_records = {index: [] for index in mobile_indices}
        seen_edges = set()

        for index in mobile_indices:
            center = structure[index].coords
            candidate_neighbors = []
            for neighbor in structure.get_neighbors(structure[index], cutoff):
                target = getattr(neighbor, "index", None)
                distance = float(neighbor.nn_distance)
                if (
                    target is None
                    or target not in mobile_set
                    or distance <= 0.35
                ):
                    continue
                candidate_neighbors.append(neighbor)

            candidate_neighbors.sort(
                key=lambda neighbor: float(neighbor.nn_distance)
            )
            for neighbor in candidate_neighbors[:12]:
                target = int(neighbor.index)
                distance = float(neighbor.nn_distance)

                image = tuple(
                    int(round(float(value)))
                    for value in getattr(neighbor, "image", (0, 0, 0))
                )
                forward = (index, target, image)
                reverse = (
                    target,
                    index,
                    tuple(-value for value in image),
                )
                edge_key = min(forward, reverse)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                vector = np.asarray(neighbor.coords - center, dtype=float)
                norm = float(np.linalg.norm(vector))
                if norm <= 1e-10:
                    continue
                direction = vector / norm

                bottleneck_clearance = min(
                    clearance_at(center + fraction * vector)
                    for fraction in (0.25, 0.50, 0.75)
                )
                bottleneck_score = min(
                    1.0,
                    max(
                        0.0,
                        (bottleneck_clearance - 0.10 * mobile_radius)
                        / max(0.35, 1.25 * mobile_radius),
                    ),
                )
                target_distance = 0.72 * cutoff
                distance_width = max(0.40, 0.25 * cutoff)
                hop_score = math.exp(
                    -((distance - target_distance) / distance_width) ** 2
                )
                cage_score = 0.5 * (
                    cage_openness[index] + cage_openness[target]
                )
                path_quality = hop_score * (
                    0.58 * bottleneck_score + 0.42 * cage_score
                )

                path_records[index].append(
                    (
                        direction,
                        path_quality,
                        hop_score,
                        bottleneck_score,
                    )
                )
                path_records[target].append(
                    (
                        -direction,
                        path_quality,
                        hop_score,
                        bottleneck_score,
                    )
                )
                active.add(index)
                active.add(target)
                if target != index:
                    source_position = position[index]
                    target_position = position[target]
                    adjacency[source_position].add(target_position)
                    adjacency[target_position].add(source_position)

        if not seen_edges:
            return 0.0

        direction_scores = []
        node_path_scores = []
        hop_scores = []
        bottleneck_scores = []
        for index in mobile_indices:
            records = path_records[index]
            if not records:
                continue
            weights = np.asarray(
                [max(1e-8, record[1]) for record in records],
                dtype=float,
            )
            tensor = np.zeros((3, 3), dtype=float)
            for weight, record in zip(weights, records):
                direction = record[0]
                tensor += weight * np.outer(direction, direction)
            tensor /= float(weights.sum())
            eigenvalues = np.clip(
                np.linalg.eigvalsh(tensor),
                0.0,
                None,
            )
            denominator = float(np.sum(eigenvalues * eigenvalues))
            participation = (
                float(np.sum(eigenvalues) ** 2 / denominator)
                if denominator > 1e-12
                else 1.0
            )
            direction_score = 0.35 + 0.65 * min(
                1.0,
                max(0.0, (participation - 1.0) / 2.0),
            )
            direction_scores.append(direction_score)
            best_paths = sorted(
                (record[1] for record in records),
                reverse=True,
            )[:4]
            node_path_scores.append(sum(best_paths) / len(best_paths))
            hop_scores.extend(record[2] for record in records)
            bottleneck_scores.extend(record[3] for record in records)

        visited = set()
        largest_component = 0
        for start in range(len(mobile_indices)):
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            component_size = 0
            while stack:
                current = stack.pop()
                component_size += 1
                for target in adjacency[current]:
                    if target not in visited:
                        visited.add(target)
                        stack.append(target)
            largest_component = max(largest_component, component_size)

        active_fraction = len(active) / float(len(mobile_indices))
        connected_fraction = largest_component / float(len(mobile_indices))
        coverage = 0.55 * active_fraction + 0.45 * connected_fraction
        hop_component = sum(hop_scores) / len(hop_scores)
        bottleneck_component = (
            sum(bottleneck_scores) / len(bottleneck_scores)
        )
        cage_component = sum(cage_openness.values()) / len(cage_openness)
        direction_component = (
            sum(direction_scores) / len(direction_scores)
            if direction_scores
            else 0.0
        )
        path_component = (
            sum(node_path_scores) / len(node_path_scores)
            if node_path_scores
            else 0.0
        )

        score = (
            0.17 * hop_component
            + 0.24 * bottleneck_component
            + 0.15 * cage_component
            + 0.16 * direction_component
            + 0.16 * path_component
            + 0.12 * coverage
        )
        return float(min(1.0, max(0.0, score)))
    except Exception:
        return 0.0

def rule_r23_open_flexible_percolation(structure):
    try:
        if structure is None or len(structure) == 0 or float(structure.volume) <= 0:
            return 0.0

        mobile_symbols = {"Li", "Na", "K", "Rb", "Cs", "Ag"}

        def dominant_specie(site):
            try:
                return max(site.species.items(), key=lambda x: x[1])[0]
            except Exception:
                try:
                    return site.specie
                except Exception:
                    return None

        def radius(sp):
            if sp is None:
                return 1.0
            for name in ("ionic_radius", "atomic_radius", "atomic_radius_calculated"):
                try:
                    value = getattr(sp, name)
                    if value is not None:
                        value = float(value)
                        if value > 0:
                            return min(3.0, max(0.3, value))
                except Exception:
                    pass
            return 1.0

        mobile = []
        host = []
        sphere_volume = 0.0
        for site in structure:
            sp = dominant_specie(site)
            if sp is None:
                continue
            r = radius(sp)
            try:
                occ = float(site.species[sp])
            except Exception:
                occ = 1.0
            sphere_volume += occ * 4.1887902047863905 * r ** 3
            symbol = getattr(sp, "symbol", str(sp))
            if symbol in mobile_symbols:
                mobile.append((site, sp, r))
            else:
                host.append((site, sp, r))

        if not mobile or not host:
            return 0.0

        packing = sphere_volume / float(structure.volume)
        openness = max(0.0, min(1.0, 1.0 - packing))

        host_radii = {}
        for site, sp, r in host:
            host_radii[getattr(sp, "symbol", str(sp))] = r

        coherent_sites = 0
        framework_bonds = []
        for site, sp, r in host:
            bonds = 0
            try:
                neighbors = structure.get_neighbors(site, 4.5)
            except Exception:
                neighbors = []
            for neighbor in neighbors:
                nsp = dominant_specie(neighbor)
                nsym = getattr(nsp, "symbol", str(nsp)) if nsp is not None else ""
                if nsym in mobile_symbols:
                    continue
                nr = host_radii.get(nsym, radius(nsp))
                d = float(neighbor.nn_distance)
                if d <= 1.35 * (r + nr):
                    bonds += 1
                    framework_bonds.append(d)
            if bonds >= 2:
                coherent_sites += 1
        coherence = coherent_sites / float(len(host))

        if framework_bonds:
            mean_bond = sum(framework_bonds) / len(framework_bonds)
            variance = sum((d - mean_bond) ** 2 for d in framework_bonds) / len(framework_bonds)
            cv = variance ** 0.5 / mean_bond if mean_bond > 0 else 0.0
            flexibility = 1.0 / (1.0 + ((cv - 0.08) / 0.08) ** 2)
        else:
            flexibility = 0.0

        active_axes = set()
        bottlenecks = []
        for site, sp, mobile_radius in mobile:
            try:
                neighbors = structure.get_neighbors(site, 6.0)
            except Exception:
                neighbors = []
            for neighbor in neighbors:
                nsp = dominant_specie(neighbor)
                if getattr(nsp, "symbol", str(nsp)) not in mobile_symbols:
                    continue
                delta = [float(neighbor.frac_coords[i] - site.frac_coords[i]) for i in range(3)]
                for i in range(3):
                    if abs(delta[i]) > 0.25:
                        active_axes.add(i)
                midpoint = tuple(float(site.frac_coords[i]) + 0.5 * delta[i] for i in range(3))
                clearance = None
                for hsite, hsp, hr in host:
                    try:
                        d = float(structure.lattice.get_distance_and_image(midpoint, hsite.frac_coords)[0]) - hr
                        if clearance is None or d < clearance:
                            clearance = d
                    except Exception:
                        pass
                if clearance is not None:
                    score = (clearance - mobile_radius + 0.5) / 1.5
                    bottlenecks.append(max(0.0, min(1.0, score)))

        dimensionality = len(active_axes) / 3.0
        bottleneck_score = sum(bottlenecks) / len(bottlenecks) if bottlenecks else 0.0
        percolation = dimensionality * bottleneck_score
        return float(openness * coherence * flexibility * percolation)
    except Exception:
        return 0.0

def rule_r24_mobile_ion_pathway_connectivity(structure):
    try:
        mobile_groups = (("Li", "Na", "K", "Rb", "Cs", "Ag"), ("H", "Cu", "Mg", "Ca"), ("F", "O"))
        radius = {"H": 0.25, "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67, "Ag": 1.15, "Cu": 0.77, "Mg": 0.72, "Ca": 1.00, "F": 1.33, "O": 1.40, "S": 1.84, "Cl": 1.81, "Br": 1.96, "I": 2.20, "P": 0.38, "Si": 0.40, "Al": 0.54, "B": 0.27, "N": 1.46}
        amounts = structure.composition.get_el_amt_dict()
        targets = []
        for group in mobile_groups:
            targets = [x for x in group if amounts.get(x, 0.0) > 0.0]
            if targets:
                break
        if not targets:
            return 0.0

        def vector_rank(vectors):
            rows = [[float(x) for x in v] for v in vectors if any(abs(float(x)) > 1e-8 for x in v)]
            rank = 0
            col = 0
            while rank < len(rows) and col < 3:
                pivot = next((i for i in range(rank, len(rows)) if abs(rows[i][col]) > 1e-8), None)
                if pivot is None:
                    col += 1
                    continue
                rows[rank], rows[pivot] = rows[pivot], rows[rank]
                p = rows[rank][col]
                rows[rank] = [x / p for x in rows[rank]]
                for i in range(len(rows)):
                    if i != rank and abs(rows[i][col]) > 1e-8:
                        f = rows[i][col]
                        rows[i] = [rows[i][j] - f * rows[rank][j] for j in range(3)]
                rank += 1
                col += 1
            return rank

        def score_target(target):
            mobile = []
            framework = []
            for site in structure:
                occ = 0.0
                blocker_radius = 0.0
                for sp, amount in site.species.items():
                    symbol = getattr(sp, "symbol", str(sp).split("+")[0].split("-")[0])
                    if symbol == target:
                        occ += float(amount)
                    else:
                        blocker_radius = max(blocker_radius, radius.get(symbol, 1.20) * min(1.0, max(0.25, float(amount))))
                if occ > 0.0:
                    mobile.append((site.frac_coords, occ))
                if blocker_radius > 0.0:
                    framework.append((site.frac_coords, blocker_radius))
            if not mobile:
                return 0.0
            rm = radius.get(target, 0.9)
            cutoff = max(3.2, min(6.0, 2.8 + 2.0 * rm))
            adjacency = [[] for _ in mobile]
            edges = []
            shifts = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
            for i, (fi, oi) in enumerate(mobile):
                for j in range(i, len(mobile)):
                    fj, oj = mobile[j]
                    for t in shifts:
                        if i == j and t == (0, 0, 0):
                            continue
                        delta = [float(fj[k]) + t[k] - float(fi[k]) for k in range(3)]
                        cart = structure.lattice.get_cartesian_coords(delta)
                        d = sum(float(x) * float(x) for x in cart) ** 0.5
                        if d < 0.8 or d > cutoff:
                            continue
                        clearance = 99.0
                        for s in (0.2, 0.4, 0.5, 0.6, 0.8):
                            q = [float(fi[k]) + s * delta[k] for k in range(3)]
                            for ff, rf in framework:
                                df = float(structure.lattice.get_distance_and_image(q, ff)[0]) - rf
                                if df < clearance:
                                    clearance = df
                        if not framework:
                            clearance = 2.0 * rm
                        if clearance < 0.45 * rm:
                            continue
                        eid = len(edges)
                        edges.append((i, j, t, clearance, d))
                        adjacency[i].append((j, t, eid))
                        adjacency[j].append((i, (-t[0], -t[1], -t[2]), eid))
            if not edges:
                return 0.0
            total_weight = sum(x[1] for x in mobile)
            visited = set()
            best = 0.0
            for start in range(len(mobile)):
                if start in visited:
                    continue
                queue = [start]
                visited.add(start)
                offsets = {start: (0, 0, 0)}
                nodes = []
                edge_ids = set()
                cycles = []
                while queue:
                    u = queue.pop(0)
                    nodes.append(u)
                    for v, t, eid in adjacency[u]:
                        edge_ids.add(eid)
                        candidate = tuple(offsets[u][k] + t[k] for k in range(3))
                        if v not in offsets:
                            offsets[v] = candidate
                            visited.add(v)
                            queue.append(v)
                        else:
                            diff = tuple(candidate[k] - offsets[v][k] for k in range(3))
                            if any(diff):
                                cycles.append(diff)
                rank = vector_rank(cycles)
                if rank == 0 or not edge_ids:
                    continue
                component_weight = sum(mobile[i][1] for i in nodes) / total_weight
                clearances = [edges[e][3] for e in edge_ids]
                lengths = [edges[e][4] for e in edge_ids]
                mean_c = sum(clearances) / len(clearances)
                mean_l = sum(lengths) / len(lengths)
                var_c = sum((x - mean_c) ** 2 for x in clearances) / len(clearances)
                var_l = sum((x - mean_l) ** 2 for x in lengths) / len(lengths)
                uniformity = 1.0 / (1.0 + (var_c ** 0.5) / max(mean_c, 0.1) + 0.5 * (var_l ** 0.5) / max(mean_l, 0.1))
                width = max(0.0, min(1.0, (mean_c - 0.45 * rm) / max(1.1 * rm, 0.2)))
                dead = sum(1 for i in nodes if len(adjacency[i]) <= 1) / len(nodes)
                anisotropy = rank / 3.0
                score = component_weight * anisotropy * width * uniformity * (1.0 - 0.6 * dead)
                best = max(best, score)
            return best

        return float(max(0.0, min(1.0, max(score_target(t) for t in targets))))
    except Exception:
        return 0.0

def rule_r25_crystallographic_disorder(structure):
    try:
        mobile = {"H", "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Cu", "Ag", "Zn", "O", "F", "Cl", "Br", "I"}
        site_impurities = []
        mobile_impurities = []
        mobile_data = []
        element_capacity = {}
        element_amount = {}
        for i, site in enumerate(structure):
            entries = []
            for specie, occupancy in site.species.items():
                value = float(occupancy)
                if value <= 0.0 or value != value:
                    continue
                symbol = getattr(specie, "symbol", None)
                if symbol is None:
                    symbol = getattr(getattr(specie, "element", None), "symbol", str(specie))
                entries.append((str(symbol), value))
            total = sum(x[1] for x in entries)
            if total <= 0.0:
                continue
            scale = max(1.0, total)
            probabilities = [x[1] / scale for x in entries]
            if total < 1.0:
                probabilities.append(1.0 - total)
            impurity = max(0.0, min(1.0, 1.0 - sum(p * p for p in probabilities)))
            site_impurities.append(impurity)
            present_mobile = set()
            mobile_occupancy = 0.0
            for symbol, occupancy in entries:
                if symbol in mobile:
                    present_mobile.add(symbol)
                    mobile_occupancy += occupancy
                    element_capacity[symbol] = element_capacity.get(symbol, 0.0) + 1.0
                    element_amount[symbol] = element_amount.get(symbol, 0.0) + occupancy
            if present_mobile:
                mobile_impurities.append(impurity)
                if impurity > 1e-8 or mobile_occupancy < 0.999:
                    mobile_data.append((i, present_mobile, impurity, mobile_occupancy))
        if not site_impurities:
            return 0.0
        global_disorder = sum(site_impurities) / len(site_impurities)
        mobile_disorder = sum(mobile_impurities) / len(mobile_impurities) if mobile_impurities else 0.0
        excess_values = []
        for symbol, capacity in element_capacity.items():
            if capacity > 0.0:
                excess_values.append(max(0.0, min(1.0, 1.0 - element_amount.get(symbol, 0.0) / capacity)))
        site_excess = sum(excess_values) / len(excess_values) if excess_values else 0.0
        degrees = [0] * len(mobile_data)
        split_strength = 0.0
        for a in range(len(mobile_data)):
            for b in range(a + 1, len(mobile_data)):
                if not mobile_data[a][1].intersection(mobile_data[b][1]):
                    continue
                distance = float(structure[mobile_data[a][0]].distance(structure[mobile_data[b][0]]))
                if 0.5 < distance <= 4.2:
                    degrees[a] += 1
                    degrees[b] += 1
                    if distance < 2.2:
                        split_strength += (2.2 - distance) / 1.7
        node_count = len(mobile_data)
        split_score = min(1.0, split_strength / node_count) if node_count else 0.0
        connectivity = sum(1 for degree in degrees if degree >= 2) / node_count if node_count else 0.0
        score = 0.30 * global_disorder + 0.35 * mobile_disorder + 0.20 * site_excess + 0.10 * split_score + 0.05 * connectivity
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0
