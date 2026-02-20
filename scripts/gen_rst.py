import os

modules = [
    f[:-3] for f in os.listdir("src")
    if f.endswith(".py") and not f.startswith("__")
]

import os

# Modules à exclure de la documentation
EXCLUDE = {"imports"}

modules = [
    f[:-3] for f in os.listdir("src")
    if f.endswith(".py") and not f.startswith("__") and f[:-3] not in EXCLUDE
]
os.makedirs("docs/sphinx/source", exist_ok=True)

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

# Générer l'index
toc_entries = "\n".join(f"   {mod}" for mod in sorted(modules))
index = f"""psipy — Documentation
=====================

.. toctree::
   :maxdepth: 2
   :caption: Modules

{toc_entries}
"""
with open("docs/sphinx/source/index.rst", "w") as f:
    f.write(index)

print(f"✅ {len(modules)} fichiers .rst générés")
