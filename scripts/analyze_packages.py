#!/usr/bin/env python3
"""
analyze_packages.py (enhanced with parallel parsing)
-----------------------------------------------------
Static analysis of a collection of Python packages:
- Dependency graph (imports between modules)
- Inventory of public symbols (classes, functions, signatures)
- Redundancy detection (similar names, duplicate symbols)
- Naming inconsistency analysis (generic param names, style violations)
- Documentation volume estimation (lines of docstrings)
- Suggested merge report

Parallel parsing speeds up analysis of many files.
"""

import ast
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional, Tuple, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────
# 1. PARSING (enhanced signatures + doc volume)
# ─────────────────────────────────────────────────────────────────

def parse_module(filepath: Path) -> dict:
    """
    Extracts from a .py file:
    - imports (internal and external)
    - defined classes (with methods, signatures)
    - defined functions (with signatures)
    - top-level constants
    - module docstring
    - total lines of docstrings (documentation volume)
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
        'functions' : [],   # (name, signature_dict, signature_str, lineno)
        'constants' : [],   # name
        'lines'     : len(source.splitlines()),
        'doc_lines' : 0,    # total lines of docstrings in this module
    }

    # Add module docstring lines
    if info['docstring']:
        info['doc_lines'] += len(info['docstring'].splitlines())

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
            class_doc = ast.get_docstring(node) or ''
            if class_doc:
                info['doc_lines'] += len(class_doc.splitlines())

            methods = []
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig_dict, sig_str = _extract_signature(item)
                    method_doc = ast.get_docstring(item) or ''
                    if method_doc:
                        info['doc_lines'] += len(method_doc.splitlines())
                    methods.append({
                        'name'       : item.name,
                        'signature'  : sig_dict,
                        'signature_str': sig_str,
                        'lineno'     : item.lineno,
                        'doc'        : method_doc,
                        'decorators' : [ast.unparse(d) for d in item.decorator_list],
                    })
            bases = [ast.unparse(b) for b in node.bases]
            info['classes'].append({
                'name'   : node.name,
                'methods': methods,
                'bases'  : bases,
                'lineno' : node.lineno,
                'doc'    : class_doc,
            })

        # ── top-level functions ──────────────────────────────────
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not _is_nested(node, tree):
                sig_dict, sig_str = _extract_signature(node)
                func_doc = ast.get_docstring(node) or ''
                if func_doc:
                    info['doc_lines'] += len(func_doc.splitlines())
                info['functions'].append({
                    'name'       : node.name,
                    'signature'  : sig_dict,
                    'signature_str': sig_str,
                    'lineno'     : node.lineno,
                    'doc'        : func_doc,
                    'decorators' : [ast.unparse(d) for d in node.decorator_list],
                })

        # ── top-level constants ─────────────────────────────────
        elif isinstance(node, ast.Assign):
            if not _is_nested(node, tree):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id.isupper():
                        info['constants'].append(t.id)

    return info

def _extract_signature(node):
    """
    Return (dict, str) describing the function signature.
    dict contains: args (list of names), defaults (dict name->value_str),
                   vararg, kwarg, kwonlyargs, annotations (if any).
    str is a readable signature like "func(a, b=1, *args, **kwargs) -> int".
    """
    args = node.args
    # Collect all argument names and their default values
    arg_names = []
    defaults = {}
    # positional args
    pos_args = [a.arg for a in args.args]
    arg_names.extend(pos_args)
    # defaults for positional args
    num_no_default = len(pos_args) - len(args.defaults)
    for i, d in enumerate(args.defaults):
        param_name = pos_args[num_no_default + i]
        defaults[param_name] = ast.unparse(d)
    # *vararg
    vararg = args.vararg.arg if args.vararg else None
    # keyword-only args
    kwonly_args = [a.arg for a in args.kwonlyargs]
    arg_names.extend(kwonly_args)
    # defaults for keyword-only args
    for i, d in enumerate(args.kw_defaults):
        if d is not None:
            param_name = kwonly_args[i]
            defaults[param_name] = ast.unparse(d)
    # **kwarg
    kwarg = args.kwarg.arg if args.kwarg else None

    # Annotations (if any)
    annotations = {}
    for a in args.args:
        if a.annotation:
            annotations[a.arg] = ast.unparse(a.annotation)
    for a in args.kwonlyargs:
        if a.annotation:
            annotations[a.arg] = ast.unparse(a.annotation)
    if args.vararg and args.vararg.annotation:
        annotations[args.vararg.arg] = ast.unparse(args.vararg.annotation)
    if args.kwarg and args.kwarg.annotation:
        annotations[args.kwarg.arg] = ast.unparse(args.kwarg.annotation)
    return_annotation = ast.unparse(node.returns) if node.returns else None

    # Build readable signature string
    parts = []
    for a in pos_args:
        if a in defaults:
            parts.append(f"{a}={defaults[a]}")
        else:
            parts.append(a)
    if vararg:
        parts.append(f"*{vararg}")
    for a in kwonly_args:
        if a in defaults:
            parts.append(f"{a}={defaults[a]}")
        else:
            parts.append(a)
    if kwarg:
        parts.append(f"**{kwarg}")

    sig_str = f"{node.name}({', '.join(parts)})"
    if return_annotation:
        sig_str += f" -> {return_annotation}"

    return {
        'args'       : arg_names,
        'defaults'   : defaults,
        'vararg'     : vararg,
        'kwarg'      : kwarg,
        'annotations': annotations,
        'return_ann' : return_annotation,
    }, sig_str

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
# 4. NAMING INCONSISTENCY ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_naming_inconsistencies(modules: list[dict]) -> dict:
    """
    Detect:
    - Generic parameter names (arg, param, value, x, y, etc.)
    - Non-snake_case parameter names
    - Methods with same name but different signatures (possible polymorphism issue)
    """
    generic_names = {'arg', 'args', 'param', 'params', 'val', 'value',
                     'x', 'y', 'z', 'data', 'item', 'obj'}

    issues = {
        'generic_params': defaultdict(list),   # param -> [(func_path, signature)]
        'non_snake_case': defaultdict(list),   # param -> [(func_path, signature)]
        'mismatched_methods': []                # list of (method_name, class1_sig, class2_sig)
    }

    # Collect all methods for signature comparison
    methods_by_name = defaultdict(list)   # method_name -> [(class_name, signature_dict, module_name)]

    for mod in modules:
        mod_name = mod['name']
        for cls in mod['classes']:
            for m in cls['methods']:
                method_name = m['name']
                # Check each parameter of this method
                for param in m['signature']['args']:
                    if param in generic_names:
                        issues['generic_params'][param].append(
                            f"{mod_name}.{cls['name']}.{method_name} (params: {', '.join(m['signature']['args'])})"
                        )
                    if not param.isidentifier() or (param.lower() != param and '_' in param):
                        # Not snake_case: contains uppercase or not all lowercase (except single-letter)
                        if not (len(param) == 1 or param.islower() and '_' not in param):
                            issues['non_snake_case'][param].append(
                                f"{mod_name}.{cls['name']}.{method_name}"
                            )
                # Store for cross-class comparison
                methods_by_name[method_name].append({
                    'class': cls['name'],
                    'module': mod_name,
                    'signature': m['signature'],
                    'signature_str': m['signature_str']
                })

        # Also check top-level functions
        for fn in mod['functions']:
            fn_name = fn['name']
            for param in fn['signature']['args']:
                if param in generic_names:
                    issues['generic_params'][param].append(
                        f"{mod_name}.{fn_name} (params: {', '.join(fn['signature']['args'])})"
                    )
                if not param.isidentifier() or (param.lower() != param and '_' in param):
                    if not (len(param) == 1 or param.islower() and '_' not in param):
                        issues['non_snake_case'][param].append(
                            f"{mod_name}.{fn_name}"
                        )

    # Compare methods with same name but different signatures
    for method_name, occurrences in methods_by_name.items():
        if len(occurrences) < 2:
            continue
        # Compare signatures: simple check on number of parameters (excluding 'self')
        sigs = []
        for occ in occurrences:
            # Exclude 'self' if present (typical first param in methods)
            params = [p for p in occ['signature']['args'] if p != 'self']
            sigs.append((occ['class'], occ['module'], params))
        # If any differ in parameter count, flag
        base_params = sigs[0][2]
        for cls, mod, params in sigs[1:]:
            if params != base_params:
                issues['mismatched_methods'].append({
                    'method': method_name,
                    'variants': [
                        f"{mod}.{cls} ({', '.join(params)})"
                        for cls, mod, params in sigs
                    ]
                })
                break   # Only report once per method

    return issues

# ─────────────────────────────────────────────────────────────────
# 5. MERGE SUGGESTIONS
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
# 6. REPORT (enhanced with doc volume)
# ─────────────────────────────────────────────────────────────────

def print_report(modules, graph, redundancies, naming_issues, suggestions):

    sep  = "=" * 75
    sep2 = "-" * 75

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n{sep}")
    print("  MODULE SUMMARY (lines, documentation, counts)")
    print(sep)
    header = f"  {'Module':30s} {'Lines':>6s} {'Doc':>6s} {'D/C':>6s}  {'Classes':>7s}  {'Functions':>9s}"
    print(header)
    print(sep2)
    for mod in sorted(modules, key=lambda m: -m['lines']):
        n_cls = len(mod['classes'])
        n_fn  = len(mod['functions'])
        doc_lines = mod['doc_lines']
        ratio = doc_lines / mod['lines'] if mod['lines'] > 0 else 0
        print(f"  {mod['name']:30s} {mod['lines']:6d} {doc_lines:6d} {ratio:5.2f}   {n_cls:7d}  {n_fn:9d}")
    print(sep)

    # ── Collect detailed inventory (classes & functions) ─────────
    detailed_lines = []
    for mod in sorted(modules, key=lambda m: m['name']):
        detailed_lines.append(f"\n  {mod['name']} ({mod['lines']} lines)")
        for cls in mod['classes']:
            detailed_lines.append(f"    class {cls['name']:30s} ({len(cls['methods'])} methods)")
            for method in cls['methods']:
                if not method['name'].startswith('_'):
                    detailed_lines.append(f"        {method['signature_str']}")
        for fn in mod['functions']:
            if not fn['name'].startswith('_'):
                detailed_lines.append(f"    def   {fn['signature_str']}")

    # ── Dependency graph ─────────────────────────────────────────
    print("\n  DEPENDENCY GRAPH")
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

    # ── Redundancies ─────────────────────────────────────────────
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

    # ── Naming inconsistencies ───────────────────────────────────
    print(f"\n{sep}")
    print("  NAMING INCONSISTENCIES")
    print(sep)

    if naming_issues['generic_params']:
        print("\n  Generic parameter names (maybe rename):")
        for param, locations in sorted(naming_issues['generic_params'].items()):
            print(f"    {param}:")
            for loc in locations[:5]:
                print(f"      {loc}")
            if len(locations) > 5:
                print(f"      ... and {len(locations)-5} more")

    if naming_issues['non_snake_case']:
        print("\n  Non-snake_case parameter names:")
        for param, locations in sorted(naming_issues['non_snake_case'].items()):
            print(f"    {param}:")
            for loc in locations[:5]:
                print(f"      {loc}")
            if len(locations) > 5:
                print(f"      ... and {len(locations)-5} more")

    if naming_issues['mismatched_methods']:
        print("\n  Methods with same name but different signatures:")
        for issue in naming_issues['mismatched_methods']:
            print(f"    {issue['method']}:")
            for v in issue['variants']:
                print(f"      {v}")

    # ── Merge suggestions ────────────────────────────────────────
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

    # ── Detailed inventory (printed at the end) ──────────────────
    print(f"\n{sep}")
    print("  DETAILED INVENTORY (classes and functions)")
    print(sep)
    for line in detailed_lines:
        print(line)

    print(f"\n{sep}\n")
    
def print_report_old(modules, graph, redundancies, naming_issues, suggestions):

    sep  = "=" * 75
    sep2 = "-" * 75

    print(f"\n{sep}")
    print("  MODULE INVENTORY (with signatures and doc volume)")
    print(sep)
    # Print header with columns: Module, Lines, Doc, Doc/Code, Classes, Functions
    header = f"  {'Module':30s} {'Lines':>6s} {'Doc':>6s} {'D/C':>6s}  {'Classes':>7s}  {'Functions':>9s}"
    print(header)
    print(sep2)
    for mod in sorted(modules, key=lambda m: -m['lines']):
        n_cls = len(mod['classes'])
        n_fn  = len(mod['functions'])
        doc_lines = mod['doc_lines']
        ratio = doc_lines / mod['lines'] if mod['lines'] > 0 else 0
        print(f"  {mod['name']:30s} {mod['lines']:6d} {doc_lines:6d} {ratio:5.2f}   {n_cls:7d}  {n_fn:9d}")

        # Optionally show class/function details
        for cls in mod['classes']:
            print(f"    class {cls['name']:30s} "
                  f"({len(cls['methods'])} methods)")
            for method in cls['methods']:
                if not method['name'].startswith('_'):
                    print(f"        {method['signature_str']}")
        for fn in mod['functions']:
            if not fn['name'].startswith('_'):
                print(f"    def   {fn['signature_str']}")

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
    print("  NAMING INCONSISTENCIES")
    print(sep)

    if naming_issues['generic_params']:
        print("\n  Generic parameter names (maybe rename):")
        for param, locations in sorted(naming_issues['generic_params'].items()):
            print(f"    {param}:")
            for loc in locations[:5]:
                print(f"      {loc}")
            if len(locations) > 5:
                print(f"      ... and {len(locations)-5} more")

    if naming_issues['non_snake_case']:
        print("\n  Non-snake_case parameter names:")
        for param, locations in sorted(naming_issues['non_snake_case'].items()):
            print(f"    {param}:")
            for loc in locations[:5]:
                print(f"      {loc}")
            if len(locations) > 5:
                print(f"      ... and {len(locations)-5} more")

    if naming_issues['mismatched_methods']:
        print("\n  Methods with same name but different signatures:")
        for issue in naming_issues['mismatched_methods']:
            print(f"    {issue['method']}:")
            for v in issue['variants']:
                print(f"      {v}")

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
# 7. ENTRY POINT (with parallel parsing)
# ─────────────────────────────────────────────────────────────────

def analyze(directory: str = '.', dot_output: str = 'packages.dot', jobs: int = None):
    directory = Path(directory)
    py_files  = sorted(directory.glob('*.py'))

    if not py_files:
        print(f"No .py files found in {directory}")
        return

    print(f"Analyzing {len(py_files)} files in {directory}...")
    if jobs is None:
        jobs = os.cpu_count() or 1
    print(f"Using {jobs} parallel worker(s) for parsing.")

    modules = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_file = {executor.submit(parse_module, f): f for f in py_files}
        for future in as_completed(future_to_file):
            f = future_to_file[future]
            try:
                parsed = future.result()
            except Exception as e:
                print(f"Error parsing {f}: {e}", file=sys.stderr)
                continue
            if 'error' in parsed:
                print(f"Syntax error in {f}: {parsed['error']}", file=sys.stderr)
            else:
                modules.append(parsed)

    if not modules:
        print("No modules could be parsed successfully.")
        return

    graph          = build_dependency_graph(modules)
    redundancies   = find_redundancies(modules)
    naming_issues  = analyze_naming_inconsistencies(modules)
    suggestions    = suggest_merges(modules, graph, redundancies)

    print_report(modules, graph, redundancies, naming_issues, suggestions)
    export_dot(graph, Path(dot_output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze Python packages in a directory.")
    parser.add_argument('directory', nargs='?', default='.',
                        help='Directory containing .py files (default: current directory)')
    parser.add_argument('-o', '--dot-output', default='packages.dot',
                        help='Output file for Graphviz .dot (default: packages.dot)')
    parser.add_argument('-j', '--jobs', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()
    analyze(args.directory, args.dot_output, args.jobs)