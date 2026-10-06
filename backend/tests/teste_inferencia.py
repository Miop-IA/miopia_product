import sys
import os

# Força a raiz do backend no path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("-> [1/4] Iniciando execução do teste...", flush=True)

try:
    print("-> [2/4] Importando módulos do pipeline...", flush=True)
    from api.feature_extraction import extrair_pacote_analise
    from api.inferencia import predizer_risco_stacking
    print("   Módulos importados com sucesso.", flush=True)

    texto_teste = (
        "O ministro da Fazenda confirmou nesta tarde que a equipe econômica concluiu a elaboração "
        "das novas medidas fiscais que serão apresentadas ao Congresso Nacional na próxima semana, "
        "com foco no equilíbrio das contas públicas."
    )

    print("-> [3/4] Extraindo features e processando texto...", flush=True)
    feats, txts = extrair_pacote_analise(texto_teste)
    print(f"   Features extraídas: {len(feats)} métricas.", flush=True)
    print(f"   MATTR (Janela 25): {feats['trunc_diversity']}", flush=True)

    print("-> [4/4] Executando predição no Stacking...", flush=True)
    prob, faixa, orientacao, f1 = predizer_risco_stacking(feats, txts)

    print("\n" + "=" * 45, flush=True)
    print("=== SUCESSO: TESTE DE INFERÊNCIA CONCLUÍDO ===", flush=True)
    print(f"Probabilidade Calculada: {prob}", flush=True)
    print(f"Faixa de Risco:          {faixa}", flush=True)
    print(f"F1 de Referência:        {f1}", flush=True)
    print(f"Orientação:              {orientacao}", flush=True)
    print("=" * 45 + "\n", flush=True)

except Exception as e:
    print(f"\n[ERRO CAPTURADO]: {type(e).__name__} -> {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc()