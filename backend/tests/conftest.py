"""
Configuração global dos testes.

Os testes NÃO usam o artefato de produção em backend/models: um bundle sintético,
gerado pelo próprio train.py e validado pelo mesmo contrato da API, é criado numa
pasta temporária e apontado via MODEL_PATH antes de qualquer import da API.
"""
import os
import sys
import tempfile

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

_TMP_DIR = tempfile.mkdtemp(prefix="miopia_bundle_teste_")
BUNDLE_TESTE_PATH = os.path.join(_TMP_DIR, "stacking_teste.joblib")


def _gerar_bundle_teste() -> None:
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit
    from train.train import gerar_dados_sinteticos_para_teste, treinar_stacking

    np.random.seed(42)
    df = gerar_dados_sinteticos_para_teste(n_samples=200)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_treino, idx_val = next(gss.split(df, groups=df["id_noticia"]))
    treinar_stacking(
        df.iloc[idx_treino],
        df_val=df.iloc[idx_val],
        output_path=BUNDLE_TESTE_PATH,
        version="sintetico-teste",
    )


_gerar_bundle_teste()
os.environ["MODEL_PATH"] = BUNDLE_TESTE_PATH

from api.config import get_settings  # noqa: E402

get_settings.cache_clear()
