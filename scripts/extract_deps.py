with open("requirements.txt", "r") as f:
    lines = f.readlines()

dependencies = []
for line in lines:
    line = line.strip()
    # Ignorer les lignes vides, les commentaires et les chemins locaux
    if not line or line.startswith("#") or "@ file://" in line or line.startswith("-e"):
        continue
    # Garder uniquement les lignes avec un nom de package et une version
    if "==" in line or ">=" in line or "<=" in line or ">" in line or "<" in line or "~=" in line:
        dependencies.append(line)

# Afficher le résultat pour pyproject.toml
print("[project]")
print("dependencies = [")
for dep in sorted(dependencies):
    print(f'    "{dep}",')
print("]")
