"""
Adiciona ao bundle de produção os metadados exigidos pelo contrato (api/bundle_spec.py):
version, feature_order e feature_count. NÃO retreina nem altera nenhum modelo:
só grava metadados e depois roda a mesma validação que a API roda no startup.

Uso (dentro de backend/):
    python scripts/carimbar_bundle.py                       # models/stacking_miopia_v2.joblib
    python scripts/carimbar_bundle.py --versao v2 --caminho models/stacking_miopia_v2.joblib
"""
import argparse
import json
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND_DIR)

import joblib  # noqa: E402

from api.bundle_spec import ESTILO_FEATURE_NAMES, FEATURE_COUNT, FEATURE_ORDER  # noqa: E402
from api.inferencia import carregar_bundle_stacking  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caminho", default=os.path.join(BACKEND_DIR, "models", "stacking_miopia_v2.joblib"))
    parser.add_argument("--versao", default=None, help="Padrão: model_version do model_manifest.json")
    args = parser.parse_args()

    manifest_path = os.path.join(os.path.dirname(args.caminho), "model_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    bundle = joblib.load(args.caminho)
    n_xgb = getattr(bundle["xgb_denso"], "n_features_in_", None)
    if n_xgb != FEATURE_COUNT:
        sys.exit(f"Recusado: xgb_denso tem {n_xgb} features, contrato exige {FEATURE_COUNT}. Retreine o modelo.")

    bundle["version"] = args.versao or manifest.get("model_version") or "v1"
    bundle["feature_order"] = list(FEATURE_ORDER)
    bundle["feature_count"] = FEATURE_COUNT
    bundle["feature_names_estilo"] = list(ESTILO_FEATURE_NAMES)

    backup = args.caminho + ".bak"
    if not os.path.exists(backup):
        os.replace(args.caminho, backup)
        print(f"Backup do original: {backup}")
    joblib.dump(bundle, args.caminho, compress=3)

    # Mesma validação do startup da API (contrato + manifesto + inferência de fumaça)
    carregar_bundle_stacking(args.caminho)
    print(f"OK: {args.caminho} carimbado com version={bundle['version']!r} e validado.")


if __name__ == "__main__":
    main()
