#!/usr/bin/env python3
"""
analyze_packages.py
-------------------
Static analysis of a collection of Python packages:
- Dependency graph (imports between modules)
- Inventory of public symbols (classes, functions)
- Redundancy detection (similar names, similar patterns)
- Suggested merge report
"""

import ast
import os
import sys
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

# ─────────────────────────────────────────────────────────────────
# 1. PARSING
# ─────────────────────────────────────────────────────────────────

def parse_module(filepath: Path) -> dict:
    """
    Extracts from a .py file:
    - imports (internal and external)
    - defined classes
    - defined functions
    - top-level constants
    - module docstring
    """
    source = filepath.read_text(encoding='utf-8')
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {'error': str(e), 'path': filepath}

    info = {
        'path'      : filepath,
        'name'      : filepath.stem,
        'docstring' : ast.get_docstring(tree) or '',
        'imports'   : [],   # (module, [names], is_from)
        'classes'   : [],   # (name, methods, bases, lineno)
        'functions' : [],   # (name, args, lineno)
        'constants' : [],   # name
        'lines'     : len(source.splitlines()),
    }

    for node in ast.walk(tree):

        # ── imports ──────────────────────────────────────────────
        if isinstance(node, ast.Import):
            for alias in node.names:
                info['imports'].append({
                    'module' : alias.name,
                    'names'  : [],
                    'is_from': False,
                    'lineno' : node.lineno,
                })

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names  = [a.name for a in node.names]
            info['imports'].append({
                'module' : module,
                'names'  : names,
                'is_from': True,
                'lineno' : node.lineno,
            })

        # ── classes ──────────────────────────────────────────────
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in ast.walk(node)
                if isinstance(n, ast.FunctionDef) or
                   isinstance(n, ast.AsyncFunctionDef)
            ]
            bases = [
                ast.unparse(b) for b in node.bases
            ]
            info['classes'].append({
                'name'   : node.name,
                'methods': methods,
                'bases'  : bases,
                'lineno' : node.lineno,
                'doc'    : ast.get_docstring(node) or '',
            })

        # ── top-level functions ──────────────────────────────────
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_nested(node, tree):
                args = [a.arg for a in node.args.args]
                info['functions'].append({
                    'name'  : node.name,
                    'args'  : args,
                    'lineno': node.lineno,
                    'doc'   : ast.get_docstring(node) or '',
                })

        # ── top-level constants ─────────────────────────────────
        elif isinstance(node, ast.Assign):
            if not _is_nested(node, tree):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        info['constants'].append(t.id)

    return info

def _is_nested(node, tree) -> bool:
    """True if the node is inside a class or function."""
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.ClassDef,
                               ast.FunctionDef,
                               ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                if child is node and child is not parent:
                    return True
    return False

# ─────────────────────────────────────────────────────────────────
# 2. DEPENDENCY GRAPH
# ─────────────────────────────────────────────────────────────────

def build_dependency_graph(modules: list[dict]) -> dict:
    """
    Builds the dependency graph between local modules.
    Returns { module_name: set(dependencies) }
    """
    local_names = {m['name'] for m in modules}
    graph = defaultdict(set)

    for mod in modules:
        for imp in mod['imports']:
            # Direct import: 'from wkb import ...' or 'import wkb'
            imported = imp['module'].split('.')[0]
            if imported in local_names and imported != mod['name']:
                graph[mod['name']].add(imported)
            # Case 'from . import caustics'
            for name in imp['names']:
                if name in local_names and name != mod['name']:
                    graph[mod['name']].add(name)

    return dict(graph)

def find_cycles(graph: dict) -> list:
    """Detects cycles in the dependency graph (DFS)."""
    visited, in_stack, cycles = set(), set(), []

    def dfs(node, path):
        visited.add(node)
        in_stack.add(node)
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
            elif neighbor in in_stack:
                cycle_start = path.index(neighbor)
                cycles.append(path[cycle_start:] + [neighbor])
        in_stack.discard(node)

    for node in graph:
        if node not in visited:
            dfs(node, [node])

    return cycles

# ─────────────────────────────────────────────────────────────────
# 3. REDUNDANCY DETECTION
# ─────────────────────────────────────────────────────────────────

def find_redundancies(modules: list[dict]) -> dict:
    """
    Detects:
    - Identical class/function names in multiple modules
    - Similar names (ratio > 0.8)
    - Same external imports in multiple modules
    """
    # Index: name -> [(module, type)]
    symbol_index = defaultdict(list)

    for mod in modules:
        for cls in mod['classes']:
            symbol_index[cls['name']].append((mod['name'], 'class'))
        for fn in mod['functions']:
            symbol_index[fn['name']].append((mod['name'], 'function'))

    # Exact duplicates
    exact_duplicates = {
        name: locs for name, locs in symbol_index.items()
        if len(locs) > 1
    }

    # Similar names
    all_names = list(symbol_index.keys())
    similar_pairs = []
    for i, n1 in enumerate(all_names):
        for n2 in all_names[i+1:]:
            ratio = SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
            if ratio > 0.8 and n1 != n2:
                similar_pairs.append((n1, n2, ratio,
                                      symbol_index[n1],
                                      symbol_index[n2]))

    # Shared external imports
    ext_imports = defaultdict(list)
    for mod in modules:
        for imp in mod['imports']:
            root = imp['module'].split('.')[0]
            if root not in {m['name'] for m in modules}:
                ext_imports[root].append(mod['name'])

    shared_imports = {
        lib: mods for lib, mods in ext_imports.items()
        if len(mods) > 1
    }

    return {
        'exact_duplicates': exact_duplicates,
        'similar_pairs'   : similar_pairs,
        'shared_imports'  : shared_imports,
    }

# ─────────────────────────────────────────────────────────────────
# 4. MERGE SUGGESTIONS
# ─────────────────────────────────────────────────────────────────

def suggest_merges(modules, graph, redundancies) -> list[dict]:
    """
    Merge heuristics:
    1. Strongly coupled modules (A→B and B→A)
    2. Modules with many duplicated symbols
    3. Small module (< 100 lines) imported by only one other
    """
    suggestions = []

    # 1. Mutual coupling
    for mod_a, deps_a in graph.items():
        for mod_b in deps_a:
            deps_b = graph.get(mod_b, set())
            if mod_a in deps_b:
                suggestions.append({
                    'type'   : 'mutual_coupling',
                    'modules': sorted([mod_a, mod_b]),
                    'reason' : f'{mod_a} ↔ {mod_b} : mutual imports',
                })

    # 2. Duplicated symbols
    dup = redundancies['exact_duplicates']
    if dup:
        # Group by affected module pairs
        pair_counts = defaultdict(list)
        for name, locs in dup.items():
            mods = tuple(sorted(set(m for m, _ in locs)))
            pair_counts[mods].append(name)
        for mods, names in pair_counts.items():
            if len(names) >= 2:
                suggestions.append({
                    'type'   : 'duplicate_symbols',
                    'modules': list(mods),
                    'reason' : f'Duplicated symbols: {", ".join(names[:5])}',
                })

    # 3. Small satellite module
    mod_sizes  = {m['name']: m['lines'] for m in modules}
    # How many modules import each module?
    imported_by = defaultdict(list)
    for mod_a, deps in graph.items():
        for dep in deps:
            imported_by[dep].append(mod_a)

    for mod in modules:
        name = mod['name']
        if (mod_sizes[name] < 150
                and len(imported_by.get(name, [])) == 1):
            parent = imported_by[name][0]
            suggestions.append({
                'type'   : 'satellite_module',
                'modules': [parent, name],
                'reason' : (f'{name} ({mod_sizes[name]} lines) '
                            f'imported only by {parent}'),
            })

    # Deduplicate
    seen, unique = set(), []
    for s in suggestions:
        key = (s['type'], tuple(s['modules']))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique

# ─────────────────────────────────────────────────────────────────
# 5. REPORT
# ─────────────────────────────────────────────────────────────────

def print_report(modules, graph, redundancies, suggestions):

    sep  = "=" * 65
    sep2 = "-" * 65

    print(f"\n{sep}")
    print("  MODULE INVENTORY")
    print(sep)
    for mod in sorted(modules, key=lambda m: -m['lines']):
        n_cls = len(mod['classes'])
        n_fn  = len(mod['functions'])
        print(f"\n  {mod['name']:30s}  {mod['lines']:4d} lines  "
              f"{n_cls} classes  {n_fn} functions")
        for cls in mod['classes']:
            print(f"    class {cls['name']:30s} "
                  f"({len(cls['methods'])} methods)")
        for fn in mod['functions']:
            if not fn['name'].startswith('_'):
                print(f"    def   {fn['name']}")

    print(f"\n{sep}")
    print("  DEPENDENCY GRAPH")
    print(sep)
    if not graph:
        print("  No local dependencies detected.")
    for mod, deps in sorted(graph.items()):
        print(f"  {mod:30s} → {', '.join(sorted(deps))}")

    cycles = find_cycles(graph)
    if cycles:
        print(f"\n  ⚠ CYCLES DETECTED:")
        for c in cycles:
            print(f"    {' → '.join(c)}")

    print(f"\n{sep}")
    print("  REDUNDANCIES")
    print(sep)

    dups = redundancies['exact_duplicates']
    if dups:
        print(f"\n  Duplicated symbols ({len(dups)}):")
        for name, locs in sorted(dups.items()):
            locations = ', '.join(f"{m}({t})" for m, t in locs)
            print(f"    {name:35s} ← {locations}")

    sim = redundancies['similar_pairs']
    if sim:
        print(f"\n  Similar names ({len(sim)}):")
        for n1, n2, ratio, l1, l2 in sorted(sim, key=lambda x: -x[2])[:15]:
            m1 = ', '.join(m for m, _ in l1)
            m2 = ', '.join(m for m, _ in l2)
            print(f"    {n1} ≈ {n2}  ({ratio:.0%})  [{m1}] ↔ [{m2}]")

    shared = redundancies['shared_imports']
    if shared:
        print(f"\n  Shared external imports:")
        for lib, mods in sorted(shared.items()):
            print(f"    {lib:20s} ← {', '.join(sorted(mods))}")

    print(f"\n{sep}")
    print("  MERGE SUGGESTIONS")
    print(sep)
    if not suggestions:
        print("  No merge suggestions.")
    for i, s in enumerate(suggestions, 1):
        mods = ' + '.join(s['modules'])
        print(f"\n  [{i}] {s['type'].upper()}")
        print(f"       Modules: {mods}")
        print(f"       Reason:  {s['reason']}")

    print(f"\n{sep}\n")

def export_dot(graph: dict, output: Path):
    """Exports the graph in Graphviz .dot format"""
    lines = ["digraph packages {",
             '  rankdir=LR;',
             '  node [shape=box, fontname="monospace"];']
    for mod, deps in graph.items():
        for dep in deps:
            lines.append(f'  "{mod}" -> "{dep}";')
    lines.append("}")
    output.write_text('\n'.join(lines))
    print(f"  Graph exported → {output}")
    print(f"  Visualize: dot -Tpng {output} -o graph.png")

# ─────────────────────────────────────────────────────────────────
# 6. ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def analyze(directory: str = '.', dot_output: str = 'packages.dot'):
    directory = Path(directory)
    py_files  = sorted(directory.glob('*.py'))

    if not py_files:
        print(f"No .py files found in {directory}")
        return

    print(f"Analyzing {len(py_files)} files in {directory}...")
    modules = [parse_module(f) for f in py_files
               if 'error' not in parse_module(f)]

    graph        = build_dependency_graph(modules)
    redundancies = find_redundancies(modules)
    suggestions  = suggest_merges(modules, graph, redundancies)

    print_report(modules, graph, redundancies, suggestions)
    export_dot(graph, Path(dot_output))

if __name__ == '__main__':
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    analyze(directory)
