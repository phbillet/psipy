import os
import ast

# Modules à exclure de la documentation
EXCLUDE = {"imports", "mupsipy", "cotangent_bundle", "hpc_examples"}

modules = [
    f[:-3] for f in os.listdir("src")
    if f.endswith(".py") and not f.startswith("__") and f[:-3] not in EXCLUDE
]

os.makedirs("docs/sphinx/source", exist_ok=True)

# Générer les fichiers .rst pour chaque module
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
toc_entries = "\n".join(f"   {mod}" for mod in sorted(modules))
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

print(f"✅ {len(modules)} fichiers .rst générés + index.rst avec description.")
