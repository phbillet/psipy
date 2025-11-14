#!/usr/bin/env python3
import os

# Répertoire des tests
tests_dir = "tests"

# Liste des fichiers .py dans tests/
test_files = [
    f for f in os.listdir(tests_dir)
    if f.endswith(".py") and os.path.isfile(os.path.join(tests_dir, f))
]

# Nettoyer chaque fichier
for test_file in test_files:
    file_path = os.path.join(tests_dir, test_file)
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Filtrer les lignes Jupyter
    cleaned_lines = [
        line for line in lines
        if not line.strip().startswith(("%", "get_ipython()", "# In["))
    ]

    # Réécrire le fichier
    with open(file_path, "w") as f:
        f.writelines(cleaned_lines)

    print(f"Fichier {test_file} nettoyé.")
