import os
import pytest
import pandas as pd
import hashlib

import sys
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

from train.train import carregar_dados_reais, gerar_dados_sinteticos_para_teste

@pytest.fixture
def dummy_csvs(tmp_path):
    def _create(df11_data, dfmaster_data):
        caminho_11 = tmp_path / "dataset_11.csv"
        caminho_master = tmp_path / "fake_br_master.csv"
        pd.DataFrame(df11_data).to_csv(caminho_11, index=False)
        pd.DataFrame(dfmaster_data).to_csv(caminho_master, index=False)
        return str(caminho_11), str(caminho_master)
    return _create

def test_ingestao_coluna_ausente(dummy_csvs):
    # Faltando texto_truncado no dataset_11
    d11, dmaster = dummy_csvs(
        {"id_noticia": [1], "target": [1]}, 
        {"id_noticia": [1], "target": [1], "texto_bert": ["Apenas um texto para teste longo suficiente com mais de trinta palavras repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo repetindo."]}
    )
    with pytest.raises(ValueError, match="faltam colunas"):
        carregar_dados_reais(d11, dmaster)

def test_ingestao_target_invalido(dummy_csvs):
    d11, dmaster = dummy_csvs(
        {"id_noticia": [1], "target": [2], "texto_truncado": ["A"]}, 
        {"id_noticia": [1], "target": [2], "texto_bert": ["A"]}
    )
    with pytest.raises(ValueError, match="Targets inválidos"):
        carregar_dados_reais(d11, dmaster)

def test_ingestao_texto_ausente_dataset_vazio(dummy_csvs):
    # Texto vazio após strip -> será filtrado, dataset ficará vazio
    d11, dmaster = dummy_csvs(
        {"id_noticia": [1], "target": [1], "texto_truncado": ["   "]}, 
        {"id_noticia": [1], "target": [1], "texto_bert": [float("nan")]}
    )
    with pytest.raises(ValueError, match="Dataset ficou vazio"):
        carregar_dados_reais(d11, dmaster)

def test_ingestao_id_multiplicacao_indevida(dummy_csvs):
    # id_noticia/target duplicados nos dois lados causa join N:M -> falha no validate="1:1"
    d11, dmaster = dummy_csvs(
        {"id_noticia": [1, 1], "target": [1, 1], "texto_truncado": ["A", "B"]}, 
        {"id_noticia": [1, 1], "target": [1, 1], "texto_bert": ["A", "B"]}
    )
    with pytest.raises(ValueError, match="Junção falhou"):
        carregar_dados_reais(d11, dmaster)

def test_ingestao_grupos_conflitantes(dummy_csvs):
    # Dois ids diferentes, mas o mesmo texto limpo (mesmo hash), e targets diferentes!
    texto_longo = "Este é um texto gigante apenas para garantir que as trinta palavras mínimas sejam atingidas de maneira muito consistente e sem problemas durante o processamento da feature extraction que exige isso sempre."
    d11, dmaster = dummy_csvs(
        {
            "id_noticia": [1, 2], 
            "target": [0, 1], 
            "texto_truncado": [texto_longo, texto_longo]
        }, 
        {
            "id_noticia": [1, 2], 
            "target": [0, 1], 
            "texto_bert": [texto_longo, texto_longo]
        }
    )
    with pytest.raises(ValueError, match="targets conflitantes"):
        carregar_dados_reais(d11, dmaster)

def test_ingestao_insuficiente_grupos(dummy_csvs):
    # Textos diferentes (para ter hashes diferentes) mas apenas 2 grupos, menos que 10
    textos = ["Texto longo o bastante numero " + str(i) + " " + "palavra " * 30 for i in range(2)]
    d11, dmaster = dummy_csvs(
        {
            "id_noticia": [1, 2], 
            "target": [1, 0], 
            "texto_truncado": textos
        }, 
        {
            "id_noticia": [1, 2], 
            "target": [1, 0], 
            "texto_bert": textos
        }
    )
    with pytest.raises(ValueError, match="Quantidade insuficiente de grupos"):
        carregar_dados_reais(d11, dmaster)

def test_gerar_dados_sinteticos():
    df = gerar_dados_sinteticos_para_teste(12)
    assert "grupo_identidade" in df.columns
    # Mesmos grupos devem ter o MESMO target
    targets_por_grupo = df.groupby("grupo_identidade")["target"].nunique()
    assert (targets_por_grupo == 1).all(), "Grupos textuais sintéticos têm targets conflitantes"
