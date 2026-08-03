"""
Wrapper Streamlit pour le ChatBot RH
Exécute depuis la racine du projet pour éviter les problèmes d'import
"""

import sys
from pathlib import Path

# Ajouter la racine au path
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Importer et exécuter l'app avec l'encodage UTF-8
with open(root / "app" / "app.py", "r", encoding="utf-8") as f:
    exec(f.read())
