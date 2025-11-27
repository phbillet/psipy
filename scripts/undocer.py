import ast

def remove_docstrings(source):
    """
    Supprime les docstrings d'un code source Python.
    """
    tree = ast.parse(source)

    # Supprime les docstrings des modules, fonctions, classes, etc.
    for node in ast.walk(tree):
        if not hasattr(node, 'body') or not isinstance(node.body, list):
            continue

        # Vérifie si le premier élément est une docstring
        if (isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]

    # Reconstruit le code source sans les docstrings
    code_without_docstrings = ast.unparse(tree)

    return code_without_docstrings

def process_file(input_file, output_file):
    """
    Traite un fichier en supprimant les docstrings et écrit le résultat dans un nouveau fichier.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        source = file.read()

    source_without_docstrings = remove_docstrings(source)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(source_without_docstrings)


# Exemple d'utilisation
input_file = '../src/psiop.py'
output_file = '../src/psiop_ud.py'
process_file(input_file, output_file)
