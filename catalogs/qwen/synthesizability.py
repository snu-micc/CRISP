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


def rule_02(structure):

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
    return np.mean([len(row)<=2 or len(row)>10 for row in shell(s)])


def rule_03(structure):

    def cov(sym):
        fixed = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.5, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69}
        if sym not in fixed: raise ValueError("No tabulated covalent radius: "+sym)
        return fixed[sym]

    s=structure
    if not len(s): return float("nan")
    v=1.-sum(n*(4*math.pi/3)*cov(e.symbol)**3 for e,n in s.composition.element_composition.items())/s.volume
    return max(.10-v,v-.35,0.)


def rule_04(structure):

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
    return max(0.,1.-min(contacts(s,"all")))


def rule_05(structure):

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
    pen=[]
    for i,row in enumerate(shell(s)):
        unit=[s.lattice.get_cartesian_coords(s[j].frac_coords+np.asarray(t)-s[i].frac_coords)/d for j,t,d in row]
        vals_a=[np.degrees(np.arccos(np.clip(np.dot(a,b),-1.,1.))) for a,b in itertools.combinations(unit,2)]
        pen.append(sum(min(abs(v-k) for k in (60.,90.,109.47122063449069,120.,180.)) for v in vals_a)/max(len(vals_a),1))
    return np.mean(pen)


def rule_06(structure):

    s=structure
    if not len(s): return float("nan")
    system=SpacegroupAnalyzer(s,symprec=0.01).get_crystal_system()
    return {"triclinic":6,"monoclinic":4,"orthorhombic":3,"tetragonal":2,"trigonal":2,"hexagonal":2,"cubic":1}[system]


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

    s=structure
    if not len(s): return float("nan")
    return np.mean([len(row) in (4,6) for row in shell(s)])


def rule_08(structure):

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

    s=structure
    if not len(s): return float("nan")
    rad=[cov(x) for x in symbols(s)]
    return np.mean([abs(d-rad[i]-rad[j]) for i,row in enumerate(shell(s)) for j,t,d in row])


def rule_09(structure):

    def symbols(s):

        return [max(site.species.items(),key=lambda x:(x[1],x[0].symbol))[0].symbol for site in s]

    def vals(sym,mono):
        if mono: return (0,)
        e=Element(sym)
        z=e.common_oxidation_states or e.oxidation_states
        if not z: raise ValueError("No tabulated nominal charge: "+sym)
        return tuple(int(q) for q in z)

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
    b=bvs(s)
    groups=defaultdict(list)
    for sym,v in zip(symbols(s),b): groups[sym].append(v)
    return max(max(row)-min(row) for row in groups.values())


def rule_10(structure):

    s=structure
    if not len(s): return float("nan")
    return len(s.composition.element_composition.elements)


def rule_11(structure):

    s=structure
    if not len(s): return float("nan")
    ds=SpacegroupAnalyzer(s,symprec=0.01).get_symmetry_dataset()
    return len(set(tuple(map(int,r.flatten())) for r in ds.rotations))


def rule_12(structure):

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
    return max(np.std([d for j,t,d in row])/np.mean([d for j,t,d in row]) for row in shell(s))


def rule_13(structure):

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

    s=structure
    if not len(s): return float("nan")
    sy=symbols(s); tally=0
    for i,row in enumerate(shell(s)):
        if not all(sy[j]=="O" for j,t,d in row): continue
        if (len(row)==4 and sy[i] in ("Si","P","B","Al")) or (len(row)==6 and sy[i] in ("Al","Ti","Mo","Re","Zr")): tally+=1
    return tally/len(s)


def rule_14(structure):

    def asym(s):
        ds=SpacegroupAnalyzer(s,symprec=0.01).get_symmetry_dataset()
        if ds is None: raise ValueError("Symmetry not resolved")
        return Counter(map(int,ds.equivalent_atoms))

    s=structure
    if not len(s): return float("nan")
    mult=asym(s)
    return sum(n*n for n in mult.values())/len(s)


def rule_15(structure):

    s=structure
    if not len(s): return float("nan")
    return np.mean([len(x.species)>1 or abs(x.species.num_atoms-1.)>1e-8 for x in s])


def rule_16(structure):
    def _signed_site_bvs(st):

        params = {'Ag': [1.12, 1.42], 'Al': [1.13, 1.47], 'As': [1.21, 2.23], 'B': [0.79, 1.6], 'Ba': [1.88, 0.97], 'Be': [0.81, 1.47], 'Bi': [1.54, 1.67], 'Br': [1.13, 2.58], 'C': [0.78, 2.0], 'Ca': [1.5, 1.04], 'Cd': [1.28, 1.46], 'Ce': [1.68, 1.08], 'Cl': [0.99, 2.74], 'Co': [1.09, 1.7], 'Cr': [1.16, 1.56], 'Cs': [2.05, 0.86], 'Cu': [0.87, 1.75], 'Dy': [1.54, 1.1], 'Er': [1.51, 1.11], 'Eu': [1.62, 1.01], 'F': [0.58, 3.98], 'Fe': [1.16, 1.64], 'Ga': [1.14, 1.82], 'Gd': [1.58, 1.11], 'Ge': [1.21, 1.51], 'H': [0.38, 0.89], 'Hf': [1.42, 1.23], 'Hg': [1.32, 1.44], 'Ho': [1.53, 1.1], 'I': [1.33, 2.38], 'In': [1.34, 1.49], 'Ir': [1.37, 1.55], 'K': [1.73, 0.91], 'La': [1.71, 1.08], 'Li': [1.0, 0.97], 'Lu': [1.47, 1.14], 'Mg': [1.21, 1.23], 'Mn': [1.17, 1.6], 'Mo': [1.37, 1.3], 'N': [0.72, 2.61], 'Na': [1.36, 1.01], 'Nb': [1.4, 1.23], 'Nd': [1.64, 1.07], 'Ni': [1.04, 1.75], 'O': [0.63, 3.15], 'P': [1.09, 1.96], 'Pb': [1.53, 1.55], 'Pd': [1.11, 1.35], 'Pr': [1.66, 1.07], 'Rb': [1.84, 0.89], 'Re': [1.37, 1.46], 'Rh': [1.18, 1.54], 'Ru': [1.21, 1.42], 'S': [1.03, 2.35], 'Sb': [1.41, 1.72], 'Sc': [1.34, 1.2], 'Se': [1.18, 2.51], 'Si': [1.12, 1.58], 'Sm': [1.61, 1.07], 'Sn': [1.37, 1.72], 'Sr': [1.66, 0.99], 'Ta': [1.39, 1.33], 'Tb': [1.56, 1.1], 'Te': [1.4, 2.72], 'Th': [1.7, 1.11], 'Ti': [1.27, 1.32], 'Tl': [1.62, 1.44], 'Tm': [1.5, 1.11], 'U': [1.59, 1.22], 'V': [1.21, 1.45], 'W': [1.38, 1.4], 'Y': [1.52, 1.11], 'Yb': [1.49, 1.06], 'Zn': [1.07, 1.66], 'Zr': [1.43, 1.22]}
        en = ('As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'Sb', 'Se', 'Si', 'Te')
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
    _missing_attr = []
    struct = structure.copy()

    if len(struct) == 0:
        return float('nan')

    comp = struct.composition
    reduced = comp.reduced_composition
    n_elements = len(reduced)

    try:
        sga = SpacegroupAnalyzer(struct, symprec=1e-5)
        sg_number = sga.get_space_group_number()
        ptg_ops = sga.get_point_group_operations()
        n_point_ops = len(ptg_ops)
        n_independent = float(len(struct)) / n_point_ops if n_point_ops > 0 else float(len(struct))
    except Exception:
        return float('nan')

    try:
        cnn = CrystalNN()
        cn_list = []
        neighbor_dists = []
        for i in range(len(struct)):
            nbrs = cnn.get_nn_info(struct, i)
            cn_list.append(len(nbrs))
            neighbor_dists.append([float(np.linalg.norm(n['site'].coords - struct[i].coords)) for n in nbrs])
    except Exception:
        return float('nan')

    oxi_states = []
    for site in struct:
        sp = site.specie
        if (getattr(sp, 'oxi_state', _missing_attr) is not _missing_attr) and sp.oxi_state is not None:
            oxi_states.append(float(sp.oxi_state))
        else:
            oxi_states.append(None)

    all_oxi = all(s is not None for s in oxi_states)
    charge_balance = sum(oxi_states) if all_oxi else None

    bvs_list = []
    bvs_computed = False
    try:
        bvs_list = [float(x) for x in _signed_site_bvs(struct)]
        bvs_computed = True
    except Exception:
        pass

    if not bvs_computed:
        for i in range(len(struct)):
            elem_i = struct[i].specie
            r_i = float(elem_i.atomic_radius)
            bvs_i = 0.0
            for j in range(len(struct)):
                if i == j:
                    continue
                d_ij = struct.get_distance(i, j)
                if d_ij < 0.4 or d_ij > 4.5:
                    continue
                elem_j = struct[j].specie
                r_j = float(elem_j.atomic_radius)
                R0 = r_i + r_j
                b_param = 0.37
                s_ij = math.exp(-(d_ij - R0) / b_param)
                bvs_i += s_ij
            bvs_list.append(bvs_i)
        bvs_computed = True

    if not bvs_computed or len(bvs_list) != len(struct):
        return float('nan')

    total_atom_vol = 0.0
    for site in struct:
        r = float(site.specie.atomic_radius)
        total_atom_vol += (4.0 / 3.0) * math.pi * r ** 3
    cell_vol = struct.volume
    if cell_vol <= 0:
        return float('nan')
    packing_fraction = total_atom_vol / cell_vol
    void_fraction = 1.0 - packing_fraction

    min_dist = float('inf')
    for i in range(len(struct)):
        for j in range(i + 1, len(struct)):
            d = struct.get_distance(i, j)
            if d < min_dist:
                min_dist = d
    if min_dist < 0.4:
        return float('nan')

    partial_occ_count = 0
    for site in struct:
        if sum(site.species.values()) < 0.90:
            partial_occ_count += 1

    eah = struct.properties.get('energy_above_hull', None)
    bg = struct.properties.get('band_gap', None)

    score = 1.0

    if charge_balance is not None:
        if abs(charge_balance) > 0.01:
            return 0.0
    else:
        score *= 0.5

    if all_oxi:
        non_int = sum(1 for s in oxi_states if abs(s - round(s)) > 0.01)
        if non_int > 0:
            score *= max(0.0, 1.0 - 0.25 * non_int / len(oxi_states))

    if all_oxi:
        devs = []
        for bvs, oxi in zip(bvs_list, oxi_states):
            expected = abs(oxi)
            if expected > 0.01:
                devs.append(abs(bvs - expected) / expected)
        if devs:
            mean_dev = sum(devs) / len(devs)
            score *= math.exp(-2.5 * mean_dev)

    cn_bad = sum(1 for cn in cn_list if cn < 1 or cn > 14)
    if cn_bad > 0:
        score *= max(0.0, 1.0 - 0.15 * cn_bad / len(cn_list))

    if packing_fraction < 0.10 or packing_fraction > 0.92:
        score *= 0.15
    elif packing_fraction < 0.20 or packing_fraction > 0.82:
        score *= 0.50
    elif packing_fraction < 0.28 or packing_fraction > 0.75:
        score *= 0.75

    sym_factor = min(n_point_ops / 48.0, 1.0)
    score *= (0.35 + 0.65 * sym_factor)

    if n_independent > 100.0:
        score *= 0.15
    elif n_independent > 50.0:
        score *= 0.35
    elif n_independent > 25.0:
        score *= 0.60
    elif n_independent > 12.0:
        score *= 0.80

    if n_elements > 10:
        score *= 0.15
    elif n_elements > 7:
        score *= 0.35
    elif n_elements > 5:
        score *= 0.55
    elif n_elements > 3:
        score *= 0.75

    if eah is not None:
        if eah > 1.5:
            score *= 0.05
        elif eah > 1.0:
            score *= 0.15
        elif eah > 0.5:
            score *= 0.35
        elif eah > 0.1:
            score *= 0.60
        elif eah > 0.0:
            score *= 0.85

    if bg is not None:
        if bg < -0.05 or bg > 15.0:
            score *= 0.25
        elif bg < 0.0:
            score *= 0.50

    if partial_occ_count > 0:
        frac_partial = partial_occ_count / len(struct)
        score *= max(0.0, 1.0 - 1.5 * frac_partial)

    overlap_violations = 0
    for i in range(len(struct)):
        for j in range(i + 1, len(struct)):
            d = struct.get_distance(i, j)
            r_sum = float(struct[i].specie.atomic_radius) + float(struct[j].specie.atomic_radius)
            if d < 0.70 * r_sum:
                overlap_violations += 1
    if overlap_violations > 0:
        score *= max(0.0, 1.0 - 0.1 * overlap_violations)

    return max(0.0, min(1.0, score))


def rule_17(structure):

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
    scale=min(1.,min(contacts(s,"all")))
    filled=sum(n*4*math.pi*(scale*cov(e.symbol))**3/3 for e,n in s.composition.element_composition.items())/s.volume
    return 1.-filled


def rule_18(structure):

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
    q=np.asarray(shellratios(s))
    return np.mean(np.abs(q-1.)<=.15)


def rule_19(structure):

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
    return np.mean([max(.8-q,q-1.3,0.) for q in shellratios(s)])


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
    b=bvs(s);z=nearest_states(s,b)
    return abs(np.average(z,weights=[x.species.num_atoms for x in s]))


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

    def symbols(s):

        return [max(site.species.items(),key=lambda x:(x[1],x[0].symbol))[0].symbol for site in s]

    s=structure
    if not len(s): return float("nan")
    g=shell(s); sy=symbols(s)
    cnt=Counter((sy[i],tuple(sorted(sy[j] for j,t,d in row))) for i,row in enumerate(g))
    return sum((v/len(s))**2 for v in cnt.values())


def rule_22(structure):

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
    return min(abs(z) for z in spectrum(s))/s.composition.num_atoms


def rule_23(structure):

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

    s=structure
    if not len(s): return float("nan")
    rr=[cov(x) for x in symbols(s)]
    return np.mean([abs(np.mean([d-rr[j] for j,t,d in row])-rr[i])/rr[i] for i,row in enumerate(shell(s))])


def rule_24(structure):

    def asym(s):
        ds=SpacegroupAnalyzer(s,symprec=0.01).get_symmetry_dataset()
        if ds is None: raise ValueError("Symmetry not resolved")
        return Counter(map(int,ds.equivalent_atoms))

    s=structure
    if not len(s): return float("nan")
    return len(asym(s))


def rule_25(structure):

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

    s=structure
    if not len(s): return float("nan")
    g=shell(s); sy=symbols(s); counts=Counter()
    for i,row in enumerate(g):
        nn=min(d for j,t,d in row)
        sig=[sy[j] for j,t,d in row if d<=1.08*nn+1e-7]
        if not sig or len(set(sig))!=1: continue
        if sig[0]!=sy[i] and len(sig) in (4,6,8): counts[(len(sig),True)]+=1
        elif sig[0]==sy[i] and len(sig) in (4,8,12): counts[(len(sig),False)]+=1
    return max([0]+list(counts.values()))/len(s)


def rule_26(structure):

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
    return np.sqrt(np.mean(np.square(np.asarray(shellratios(s))-1.)))


def rule_27(structure):

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

    def electr(sym):
        x=float(Element(sym).X)
        if not np.isfinite(x): raise ValueError("No Pauling electronegativity: "+sym)
        return x

    s=structure
    if not len(s): return float("nan")
    X=[electr(x) for x in symbols(s)]
    return np.mean([abs(X[i]-X[j]) for i,row in enumerate(shell(s)) for j,t,d in row])


def rule_28(structure):

    def parts(s):
        comp=s.composition.element_composition
        return [(el,float(n)/comp.num_atoms) for el,n in comp.items()]

    s=structure
    if not len(s): return float("nan")
    return sum(p for e,p in parts(s) if e.Z in (43,61) or e.Z>=83)


def rule_29(structure):

    s=structure
    if not len(s): return float("nan")
    comp,factor=s.composition.element_composition.get_reduced_composition_and_factor()
    return factor


def rule_30(structure):

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

    def dimension(s):

        g=shell(s)
        adj=[[] for row in g]
        for i,row in enumerate(g):
            for j,t,d in row:
                tr=np.array(t,dtype=int)
                adj[i].append((j,tr)); adj[j].append((i,-tr))
        pos={}; weighted=0
        for start in range(len(s)):
            if start in pos: continue
            pos[start]=np.zeros(3,dtype=int); stack=[start]; size=0; cycles=[]
            while stack:
                i=stack.pop();size+=1
                for j,t in adj[i]:
                    p=pos[i]+t
                    if j not in pos: pos[j]=p;stack.append(j)
                    else:
                        v=p-pos[j]
                        if np.any(v): cycles.append(v)
            rank=int(np.linalg.matrix_rank(np.asarray(cycles))) if cycles else 0
            weighted+=size*rank
        return weighted/len(s)

    s=structure
    if not len(s): return float("nan")
    return dimension(s)


def rule_31(structure):

    def asym(s):
        ds=SpacegroupAnalyzer(s,symprec=0.01).get_symmetry_dataset()
        if ds is None: raise ValueError("Symmetry not resolved")
        return Counter(map(int,ds.equivalent_atoms))

    s=structure
    if not len(s): return float("nan")
    mult=asym(s);n=len(s)
    return -sum(v*math.log2(v/n) for v in mult.values())


def rule_32(structure):

    n_sites = len(structure)
    if n_sites == 0:
        return float('nan')

    total_occu = 0.0
    for site in structure:
        occu = sum(site.species.values())
        if occu is None:
            return float('nan')
        total_occu += float(occu)

    mean_occu = total_occu / n_sites
    return 1.0 - mean_occu


def rule_33(structure):

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
    ways=spectrum(s)
    return ways.get(0.,0)/sum(ways.values())


def rule_34(structure):

    def cov(sym):
        fixed = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.5, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69}
        if sym not in fixed: raise ValueError("No tabulated covalent radius: "+sym)
        return fixed[sym]

    s=structure
    if not len(s): return float("nan")
    return sum(n*(4*math.pi/3)*cov(e.symbol)**3 for e,n in s.composition.element_composition.items())/s.volume


def rule_35(structure):

    s=structure
    if not len(s): return float("nan")
    return s.density


def rule_36(structure):

    try:
        eah = structure.properties['energy_above_hull']
    except (KeyError, TypeError, AttributeError):
        return float('nan')

    if eah is None:
        return float('nan')

    try:
        val = float(eah)
    except (TypeError, ValueError):
        return float('nan')

    if math.isnan(val) or math.isinf(val):
        return float('nan')

    return val


def rule_37(structure):

    def asym(s):
        ds=SpacegroupAnalyzer(s,symprec=0.01).get_symmetry_dataset()
        if ds is None: raise ValueError("Symmetry not resolved")
        return Counter(map(int,ds.equivalent_atoms))

    s=structure
    if not len(s): return float("nan")
    return len(asym(s))/len(s)


def rule_38(structure):

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
    scores=[]
    for i,row in enumerate(shell(s)):
        n=len(row)
        if n not in (4,6):
            scores.append(0.);continue
        u=[s.lattice.get_cartesian_coords(s[j].frac_coords+np.asarray(t)-s[i].frac_coords)/d for j,t,d in row]
        a=np.asarray(u)
        c=np.sort(np.clip((a@a.T)[np.triu_indices(n,1)],-1.,1.))
        ref=np.array([-1/3.]*6 if n==4 else [-1.]*3+[0.]*12)
        scores.append(1.-np.sqrt(np.mean((c-ref)**2))/2.)
    return np.mean(scores)


def rule_39(structure):

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
    return min(contacts(s,"all"))


def rule_40(structure):

    lattice = structure.lattice
    a = lattice.a
    b = lattice.b
    c = lattice.c
    alpha = math.radians(lattice.alpha)
    beta = math.radians(lattice.beta)
    gamma = math.radians(lattice.gamma)

    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_a = math.sin(alpha)
    sin_b = math.sin(beta)
    sin_g = math.sin(gamma)

    det = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    if det <= 0.0:
        return float('nan')

    V = a * b * c * math.sqrt(det)
    if V <= 0.0:
        return float('nan')

    comp = structure.composition
    if len(comp) == 0:
        return float('nan')

    reduced_comp = comp.reduced_composition
    n_atoms_formula = sum(reduced_comp.values())
    n_atoms_cell = len(structure)
    if n_atoms_formula == 0 or n_atoms_cell == 0:
        return float('nan')

    Z = n_atoms_cell / n_atoms_formula

    formula_weight = reduced_comp.weight
    if formula_weight is None or formula_weight <= 0.0:
        return float('nan')

    N_A = 6.02214076e23
    V_cm3 = V * 1.0e-24
    density = (Z * formula_weight) / (V_cm3 * N_A)

    V_per_Z = V / Z

    total_sphere_volume = 0.0
    for site in structure:
        elem = site.specie
        r = elem.atomic_radius
        if r is None or r <= 0.0:
            return float('nan')
        total_sphere_volume += (4.0 / 3.0) * math.pi * r ** 3

    apf = total_sphere_volume / V

    ratios = [a / b, b / c, a / c]
    axial_penalty = 1.0
    for ratio in ratios:
        if ratio > 3.0 or ratio < 1.0 / 3.0:
            axial_penalty = 0.5
            break

    result = apf * axial_penalty

    if not math.isfinite(result):
        return float('nan')

    return float(result)


def rule_41(structure):
    struct = structure.copy()

    n_atoms = len(struct)
    if n_atoms == 0:
        return float('nan')

    sga = SpacegroupAnalyzer(struct)
    sym_ops = sga.get_symmetry_operations()

    frac_coords = [np.array(atom.frac_coords, dtype=float) for atom in struct]
    species_strs = [str(atom.specie) for atom in struct]

    tol = 1e-4

    def positions_equivalent(frac1, frac2):
        diff = frac1 - frac2
        return np.allclose(diff, np.round(diff), atol=tol)

    visited = [False] * n_atoms
    n_orbits = 0

    for i in range(n_atoms):
        if visited[i]:
            continue
        n_orbits += 1

        queue = [i]
        visited[i] = True
        qi = 0
        while qi < len(queue):
            j = queue[qi]
            qi += 1
            for op in sym_ops:
                new_frac = np.asarray(op.operate(frac_coords[j]), dtype=float)
                for k in range(n_atoms):
                    if visited[k]:
                        continue
                    if species_strs[k] != species_strs[j]:
                        continue
                    if positions_equivalent(new_frac, frac_coords[k]):
                        visited[k] = True
                        queue.append(k)

    Z_prime = n_orbits

    comp = struct.composition
    reduced_comp = comp.reduced_composition
    atoms_per_formula = sum(reduced_comp.values())
    total_atoms = sum(comp.values())

    if atoms_per_formula == 0:
        return float('nan')

    Z = total_atoms / atoms_per_formula

    return float(Z_prime * Z)


def rule_42(structure):
    try:
        struct = structure.copy()
        n_atoms = len(struct)
        if n_atoms == 0:
            return float('nan')

        sga = SpacegroupAnalyzer(struct, symprec=0.1)
        ops = sga.get_symmetry_operations()
        lattice = struct.lattice
        n_ops = len(ops)
        if n_ops == 0:
            return float('nan')

        frac_coords = np.array([s.frac_coords for s in struct])
        cart_coords = np.array([lattice.get_cartesian_coords(s.frac_coords) for s in struct])
        species_list = [s.specie for s in struct]

        max_dev = 0.0
        for i in range(n_atoms):
            for op in ops:
                new_frac = (op.rotation_matrix @ frac_coords[i] + op.translation_vector) % 1.0
                cart_new = lattice.get_cartesian_coords(new_frac)
                min_d = float('inf')
                for j in range(n_atoms):
                    if species_list[j] != species_list[i]:
                        continue
                    d = np.linalg.norm(cart_new - cart_coords[j])
                    if d < min_d:
                        min_d = d
                if min_d < float('inf'):
                    if min_d > max_dev:
                        max_dev = min_d
                else:

                    max_dev = max(max_dev, 10.0)

        wyckoff_mismatch = 0
        try:
            site_syms = sga.get_symmetry_dataset().site_symmetry_symbols
            for i in range(n_atoms):
                sym_str = site_syms[i]
                if sym_str != '1':
                    frac = frac_coords[i]
                    is_on_special = False
                    for c in frac:
                        for target in (0.0, 0.25, 0.5, 0.75):
                            if abs(c - target) < 0.02:
                                is_on_special = True
                                break
                        if is_on_special:
                            break
                    if not is_on_special:
                        wyckoff_mismatch += 1
        except Exception:
            wyckoff_mismatch = 0

        overlap_count = 0
        for i in range(n_atoms):
            r_i = species_list[i].atomic_radius
            for op in ops:
                new_frac = (op.rotation_matrix @ frac_coords[i] + op.translation_vector) % 1.0
                cart_new = lattice.get_cartesian_coords(new_frac)
                for j in range(n_atoms):
                    if j == i:
                        continue
                    r_j = species_list[j].atomic_radius
                    d = np.linalg.norm(cart_new - cart_coords[j])
                    if 0.0 < d < 0.35 * (r_i + r_j):
                        overlap_count += 1

        n_elements = len(set(species_list))
        complexity = float(n_elements) + float(n_atoms) / max(n_elements, 1)

        if complexity <= 3.0:

            order_mismatch = max(0.0, (4.0 - float(n_ops)) / 4.0)
        elif complexity <= 6.0:
            order_mismatch = max(0.0, (float(n_ops) - 8.0) / 8.0)
        else:
            order_mismatch = max(0.0, (float(n_ops) - 4.0) / 4.0)

        residual = max_dev + 0.1 * float(wyckoff_mismatch) + 0.1 * float(overlap_count) + 0.5 * order_mismatch

        return float(residual)

    except Exception:
        return float('nan')


def rule_43(structure):

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
    weight=0.
    for site,row in zip(s,shell(s)):
        for el,occ in site.species.items():
            if el.symbol=="H" or (el.symbol in ("Li","Na","K","Rb","Cs") and len(row)<=4): weight+=occ
    return weight/s.composition.num_atoms


def rule_44(structure):

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

    def electr(sym):
        x=float(Element(sym).X)
        if not np.isfinite(x): raise ValueError("No Pauling electronegativity: "+sym)
        return x

    s=structure
    if not len(s): return float("nan")
    X=[electr(x) for x in symbols(s)]
    return np.std([abs(X[i]-X[j]) for i,row in enumerate(shell(s)) for j,t,d in row])


def rule_45(structure):
    try:
        comp = structure.composition
        n_elements = len(comp)
        if n_elements == 0:
            return float('nan')

        elem_score = min(1.0, max(0.0, (n_elements - 2) / 6.0))

        amounts = [float(amt) for amt in comp.values()]
        positive = [a for a in amounts if a > 0]
        if not positive:
            return float('nan')
        min_amt = min(positive)
        normalized = [a / min_amt for a in positive]

        def _stoich_complexity(x, tol=0.06):

            if abs(x - round(x)) < tol:
                return 0.0

            for d in (2, 3, 4):
                for n in range(1, 20):
                    if abs(x - n / d) < tol:
                        return 0.5
            return 1.0

        stoich_score = max(_stoich_complexity(v) for v in normalized)

        n_sites = len(structure)
        if n_sites == 0:
            return float('nan')
        n_defective = 0
        for site in structure:
            species = site.species
            total_occ = sum(occ for _, occ in species.items())
            if total_occ < 1.0 - 1e-4 or len(species) > 1:
                n_defective += 1
        occ_score = n_defective / n_sites

        std_ox = {
            'H': 1, 'Li': 1, 'Na': 1, 'K': 1, 'Rb': 1, 'Cs': 1,
            'Be': 2, 'Mg': 2, 'Ca': 2, 'Sr': 2, 'Ba': 2,
            'B': 3, 'Al': 3, 'Ga': 3, 'In': 3, 'Tl': 3,
            'Sc': 3, 'Y': 3, 'La': 3, 'Ce': 4, 'Pr': 3, 'Nd': 3,
            'Sm': 3, 'Eu': 3, 'Gd': 3, 'Tb': 3, 'Dy': 3, 'Ho': 3,
            'Er': 3, 'Tm': 3, 'Yb': 3, 'Lu': 3,
            'Ti': 4, 'Zr': 4, 'Hf': 4,
            'V': 5, 'Nb': 5, 'Ta': 5,
            'Cr': 3, 'Mo': 6, 'W': 6,
            'Mn': 4, 'Fe': 3, 'Co': 3, 'Ni': 2,
            'Cu': 2, 'Zn': 2, 'Cd': 2, 'Hg': 2,
            'Sn': 4, 'Pb': 4,
            'U': 6, 'Th': 4, 'Pa': 5, 'Np': 4, 'Pu': 4,
            'O': -2, 'F': -1, 'Cl': -1, 'Br': -1, 'I': -1,
            'S': -2, 'Se': -2, 'Te': -2,
            'N': -3, 'P': -3, 'As': -3,
            'C': -4, 'Si': 4, 'Ge': 4,
        }

        total_charge = 0.0
        total_abs_charge = 0.0
        all_known = True
        for el, amt in comp.items():
            sym = el.symbol
            if sym in std_ox:
                c = std_ox[sym] * float(amt)
                total_charge += c
                total_abs_charge += abs(c)
            else:
                all_known = False
                break

        if all_known and total_abs_charge > 0:
            charge_score = 0.0 if abs(total_charge) / total_abs_charge < 0.05 else 1.0
        else:
            charge_score = 0.5

        subunit_match = False
        if 'O' in comp:
            n_O = float(comp['O'])
            non_O = {el: float(amt) for el, amt in comp.items() if el.symbol != 'O'}
            n_non_O = len(non_O)

            if n_O > 0 and n_non_O > 0:
                ratios = [amt / n_O for amt in non_O.values()]

                if n_non_O == 1:
                    r = ratios[0]
                    targets_1 = [1.0, 0.5, 1.0 / 3.0, 2.0 / 3.0, 0.25, 2.0, 3.0]
                    if any(abs(r - t) < 0.03 for t in targets_1):
                        subunit_match = True

                elif n_non_O == 2:
                    r1, r2 = ratios
                    target_pairs = [
                        (1.0 / 3.0, 1.0 / 3.0),
                        (1.0 / 4.0, 0.5),
                        (0.5, 1.0 / 4.0),
                        (2.0 / 7.0, 2.0 / 7.0),
                        (1.0 / 4.0, 1.0 / 4.0),
                        (1.0 / 3.0, 2.0 / 3.0),
                        (2.0 / 3.0, 1.0 / 3.0),
                        (1.0 / 4.0, 3.0 / 4.0),
                        (3.0 / 4.0, 1.0 / 4.0),
                        (0.5, 0.5),
                    ]
                    for (a, b) in target_pairs:
                        if (abs(r1 - a) < 0.03 and abs(r2 - b) < 0.03) or \
                           (abs(r1 - b) < 0.03 and abs(r2 - a) < 0.03):
                            subunit_match = True
                            break

                elif n_non_O == 3:
                    r_vals = sorted(ratios)
                    targets_3 = [
                        (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
                        (2.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0),
                        (1.0 / 4.0, 1.0 / 4.0, 2.0 / 4.0),
                    ]
                    for t in targets_3:
                        if all(abs(r_vals[i] - t[i]) < 0.03 for i in range(3)):
                            subunit_match = True
                            break
        else:

            if n_elements == 2:
                vals = [float(a) for a in comp.values()]
                if vals[1] > 0:
                    ratio = vals[0] / vals[1]
                    simple_ratios = [1, 2, 3, 4, 0.5, 1.0 / 3.0, 0.25, 1.5, 2.5, 3.5, 0.75, 4.0]
                    if any(abs(ratio - s) < 0.03 for s in simple_ratios):
                        subunit_match = True
            elif n_elements == 3:
                vals = sorted(float(a) for a in comp.values())
                if vals[2] > 0:
                    r = [v / vals[2] for v in vals]
                    ternary_targets = [
                        (1.0 / 3.0, 1.0 / 3.0, 1.0),
                        (1.0 / 4.0, 0.5, 1.0),
                        (0.5, 0.5, 1.0),
                    ]
                    for t in ternary_targets:
                        if all(abs(r[i] - t[i]) < 0.03 for i in range(3)):
                            subunit_match = True
                            break

        subunit_score = 0.0 if subunit_match else 1.0

        w = [0.25, 0.20, 0.20, 0.20, 0.15]
        score = (w[0] * elem_score +
                 w[1] * stoich_score +
                 w[2] * occ_score +
                 w[3] * charge_score +
                 w[4] * subunit_score)

        return float(score)

    except Exception:
        return float('nan')


def rule_46(structure):
    _cov_table = {'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.73, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.7, 'Ti': 1.6, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.5, 'Fe': 1.42, 'Co': 1.38, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.2, 'As': 1.19, 'Se': 1.2, 'Br': 1.2, 'Kr': 1.16, 'Rb': 2.2, 'Sr': 1.95, 'Y': 1.9, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54, 'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44, 'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'Xe': 1.4, 'Cs': 2.44, 'Ba': 2.15, 'La': 2.07, 'Ce': 2.04, 'Pr': 2.03, 'Nd': 2.01, 'Pm': 1.99, 'Sm': 1.98, 'Eu': 1.98, 'Gd': 1.96, 'Tb': 1.94, 'Dy': 1.92, 'Ho': 1.92, 'Er': 1.89, 'Tm': 1.9, 'Yb': 1.87, 'Lu': 1.87, 'Hf': 1.75, 'Ta': 1.7, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41, 'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Tl': 1.45, 'Pb': 1.46, 'Bi': 1.48, 'Po': 1.4, 'At': 1.5, 'Rn': 1.5, 'Fr': 2.6, 'Ra': 2.21, 'Ac': 2.15, 'Th': 2.06, 'Pa': 2.0, 'U': 1.96, 'Np': 1.9, 'Pu': 1.87, 'Am': 1.8, 'Cm': 1.69}

    struct = structure.copy()

    lattice = struct.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    volume = lattice.volume
    n_atoms = len(struct)

    if n_atoms == 0 or volume <= 0 or min(a, b, c) <= 0:
        return float('nan')

    try:
        sga = SpacegroupAnalyzer(struct)
        crystal_system = sga.get_crystal_system()
    except Exception:
        return float('nan')

    edges = [a, b, c]
    edge_ratio = max(edges) / min(edges)
    edge_anisotropy = edge_ratio - 1.0

    expected_vol = 0.0
    for site in struct:
        el = site.specie
        try:
            r = _cov_table.get((el).symbol)
        except Exception:
            return float('nan')
        if r is None or r <= 0:
            return float('nan')
        expected_vol += (4.0 / 3.0) * math.pi * r ** 3

    if expected_vol <= 0:
        return float('nan')

    vol_ratio = volume / expected_vol
    if vol_ratio <= 0:
        return float('nan')
    vol_score = abs(math.log(vol_ratio))

    angle_penalty = 0.0
    n_angle_constraints = 0

    if crystal_system in ('cubic', 'tetragonal', 'orthorhombic'):
        angle_penalty += abs(alpha - 90.0)
        angle_penalty += abs(beta - 90.0)
        angle_penalty += abs(gamma - 90.0)
        n_angle_constraints = 3
    elif crystal_system == 'hexagonal':
        angle_penalty += abs(alpha - 90.0)
        angle_penalty += abs(beta - 90.0)
        angle_penalty += abs(gamma - 120.0)
        n_angle_constraints = 3
    elif crystal_system == 'trigonal':

        if abs(gamma - 120.0) < abs(gamma - 90.0):
            angle_penalty += abs(alpha - 90.0)
            angle_penalty += abs(beta - 90.0)
            angle_penalty += abs(gamma - 120.0)
            n_angle_constraints = 3
        else:
            angle_penalty += abs(alpha - beta)
            angle_penalty += abs(beta - gamma)
            angle_penalty += abs(alpha - gamma)
            n_angle_constraints = 3
    elif crystal_system == 'monoclinic':

        angle_penalty += abs(alpha - 90.0)
        angle_penalty += abs(gamma - 90.0)
        n_angle_constraints = 2
    elif crystal_system == 'triclinic':
        n_angle_constraints = 0
    else:
        n_angle_constraints = 0

    angle_score = (angle_penalty / n_angle_constraints) if n_angle_constraints > 0 else 0.0

    edge_penalty = 0.0
    n_edge_constraints = 0

    if crystal_system == 'cubic':
        edge_penalty += abs(a - b) / a
        edge_penalty += abs(b - c) / b
        edge_penalty += abs(a - c) / a
        n_edge_constraints = 3
    elif crystal_system == 'tetragonal':
        edge_penalty += abs(a - b) / a
        n_edge_constraints = 1
    elif crystal_system == 'hexagonal':
        edge_penalty += abs(a - b) / a
        n_edge_constraints = 1
    elif crystal_system == 'trigonal':
        if abs(gamma - 120.0) < abs(gamma - 90.0):
            edge_penalty += abs(a - b) / a
            n_edge_constraints = 1
        else:
            edge_penalty += abs(a - b) / a
            edge_penalty += abs(b - c) / b
            edge_penalty += abs(a - c) / a
            n_edge_constraints = 3

    edge_consistency_score = (edge_penalty / n_edge_constraints) if n_edge_constraints > 0 else 0.0

    score = (
        edge_anisotropy
        + vol_score
        + angle_score / 10.0
        + edge_consistency_score * 5.0
    )

    return float(score)


def rule_47(structure):

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
    return min(contacts(s,"same"))


def rule_48(structure):

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
    return sum(z[i]*z[j]/d for i,row in enumerate(shell(s)) for j,t,d in row)/(2*len(s))


def rule_49(structure):

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

    def shares(s):

        incoming=[[] for site in s]
        for i,row in enumerate(shell(s)):
            for j,img,d in row: incoming[j].append((i,tuple(-v for v in img)))
        pair=Counter()
        for ligand in incoming:
            for (i,a),(j,b) in itertools.combinations(ligand,2):
                t=tuple(y-x for x,y in zip(a,b))
                f=(i,j,t); rev=(j,i,tuple(-x for x in t))
                pair[min(f,rev)]+=1
        return sum(n>=2 for n in pair.values())/len(pair) if pair else 0.

    s=structure
    if not len(s): return float("nan")
    return shares(s)


def rule_50(structure):

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
    b=np.asarray(bvs(s)); z=nearest_states(s,b)
    return np.sqrt(np.mean(np.where(b<0.,(b-z)**2,0.)))
