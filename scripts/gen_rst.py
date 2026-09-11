import os
import ast

# Modules à exclure de la documentation
EXCLUDE = {"imports", "mupsipy", "cotangent_bundle", "hpc_examples", "metric_catalogue"}

# Modules that have a custom, manually written .rst file (do not auto-generate them)
CUSTOM_RST = {"psiop"}

modules = []
for f in os.listdir("src"):
    path = os.path.join("src", f)
    
    # 1. Check for standard .py files
    if f.endswith(".py") and not f.startswith("__"):
        mod_name = f[:-3]
        if mod_name not in EXCLUDE and mod_name not in CUSTOM_RST:
            modules.append(mod_name)
            
    # 2. Check for packages (directories containing __init__.py)
    elif os.path.isdir(path) and os.path.exists(os.path.join(path, "__init__.py")):
        mod_name = f
        if mod_name not in EXCLUDE and mod_name not in CUSTOM_RST:
            modules.append(mod_name)

os.makedirs("docs/sphinx/source", exist_ok=True)

# Générer les fichiers .rst pour chaque module standard
for mod in sorted(modules):
    content = f"""{mod}
{"=" * len(mod)}

.. automodule:: {mod}
   :members:
   :undoc-members:
   :show-inheritance:
"""
    with open(f"docs/sphinx/source/{mod}.rst", "w") as f:
        f.write(content)

# Extraire la docstring de __init__.py
def get_init_docstring():
    init_path = os.path.join("src", "__init__.py")
    if os.path.exists(init_path):
        with open(init_path, "r") as f:
            tree = ast.parse(f.read(), filename=init_path)
        docstring = ast.get_docstring(tree)
        return docstring or "No description available."
    return "No description available."

# Générer l'index avec la description
description = get_init_docstring()

# Include both auto-generated modules AND custom RST modules in the sidebar
all_toc_modules = sorted(modules + [m for m in CUSTOM_RST if m not in EXCLUDE])
toc_entries = "\n".join(f"   {mod}" for mod in all_toc_modules)

index = f"""psipy — Documentation
=====================

{description}

.. toctree::
   :maxdepth: 2
   :caption: Modules

{toc_entries}
"""
with open("docs/sphinx/source/index.rst", "w") as f:
    f.write(index)

print(f"✅ {len(modules)} fichiers .rst générés + {len(CUSTOM_RST)} custom RST préservés.")
print(f"✅ index.rst mis à jour avec {len(all_toc_modules)} modules dans la barre latérale.")