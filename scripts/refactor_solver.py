import re
import shutil
import os

# 1. Configuration
INPUT_FILE = './src/solver.py'
OUTPUT_FILE = './src/solver_refactored.py'

# Liste des méthodes qui DOIVENT rester publiques (Whitelist)
# __init__ est exclus implicitement par la logique (commence déjà par _),
# mais on le garde ici pour la clarté.
PUBLIC_METHODS = {
    '__init__',
    'setup',
    'solve_stationary_psiOp',
    'test',
    'solve',
    'plot_energy',
    'show_stationary_solution',
    'animate'
}

def refactor_class(file_path, output_path):
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 2. Trouver toutes les méthodes définies dans le fichier
    # Regex : cherche "def nom_methode("
    method_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    all_methods = re.findall(method_pattern, content)
    
    # Filtrer pour ne garder que celles à privatiser
    methods_to_rename = []
    for method in all_methods:
        # Si la méthode n'est pas dans la liste publique 
        # ET ne commence pas déjà par un underscore
        if method not in PUBLIC_METHODS and not method.startswith('_'):
            methods_to_rename.append(method)

    # Trier par longueur décroissante pour éviter les conflits de sous-chaînes
    # (ex: éviter que renommer "setup" ne casse "setup_1D")
    methods_to_rename.sort(key=len, reverse=True)

    print(f"Analyse de {file_path}...")
    print(f"{len(methods_to_rename)} méthodes identifiées pour passage en privé.\n")

    new_content = content
    count_defs = 0
    count_calls = 0

    for method in methods_to_rename:
        new_name = f"_{method}"
        
        # A. Remplacer la définition : def ma_methode(
        # On utilise \s* pour gérer les espaces éventuels avant la parenthèse
        def_regex = rf'def\s+{method}\s*\('
        if re.search(def_regex, new_content):
            new_content = re.sub(def_regex, f'def {new_name}(', new_content)
            count_defs += 1

        # B. Remplacer les appels internes : self.ma_methode
        # \b assure qu'on ne remplace pas "ma_methode_2" par erreur
        call_regex = rf'self\.{method}\b'
        # On compte les occurrences avant de remplacer pour le rapport
        calls_found = len(re.findall(call_regex, new_content))
        if calls_found > 0:
            new_content = re.sub(call_regex, f'self.{new_name}', new_content)
            count_calls += calls_found
            
        print(f"Renommé : {method} -> {new_name} ({calls_found} appels internes mis à jour)")

    # 3. Sauvegarde
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print("-" * 40)
    print("Terminé !")
    print(f"Total définitions modifiées : {count_defs}")
    print(f"Total appels internes mis à jour : {count_calls}")
    print(f"Nouveau fichier créé : {output_path}")

if __name__ == "__main__":
    refactor_class(INPUT_FILE, OUTPUT_FILE)