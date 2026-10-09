import time
import requests
import concurrent.futures
import statistics

API_URL = "http://127.0.0.1:8000/analisar"
DUMMY_TEXT = (
    "O Ministério da Fazenda divulgou hoje uma nota oficial com as diretrizes econômicas "
    "e parâmetros fiscais que serão apresentados aos governadores na reunião marcada "
    "para o início do próximo mês em Brasília, com o objetivo de equilibrar as contas públicas."
)

def run_request(session, idx):
    payload = {
        "texto": f"{DUMMY_TEXT} (Variante {idx})",
        "url": "http://teste.com"
    }
    start = time.time()
    try:
        resp = session.post(API_URL, json=payload, timeout=5)
        status = resp.status_code
    except Exception as e:
        status = str(e)
    end = time.time()
    return status, end - start

def run_load_test(concurrency=10, total_requests=100):
    print(f"Iniciando Teste de Carga: {total_requests} requisições com {concurrency} workers concorrentes...")
    
    session = requests.Session()
    
    start_total = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(run_request, session, i) for i in range(total_requests)]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    end_total = time.time()
    
    latencies = [r[1] for r in results if isinstance(r[0], int) and r[0] == 200]
    errors = [r for r in results if not isinstance(r[0], int) or r[0] != 200]
    
    if latencies:
        print("\n--- Resultados do Teste de Carga ---")
        print(f"Requisições de Sucesso: {len(latencies)}/{total_requests}")
        print(f"Tempo Total: {end_total - start_total:.2f}s")
        print(f"Requests Por Segundo (RPS): {len(latencies) / (end_total - start_total):.2f}")
        print(f"Latência Média: {statistics.mean(latencies)*1000:.2f} ms")
        print(f"P95 Latência: {statistics.quantiles(latencies, n=20)[18]*1000:.2f} ms")
        print(f"Latência Max: {max(latencies)*1000:.2f} ms")
    else:
        print("\nFalha total do teste. Nenhuma requisição retornou 200 OK.")
        
    if errors:
        print(f"\nErros detectados: {len(errors)}")
        for e in errors[:5]:
            print(f"  - Status/Erro: {e[0]}")

if __name__ == "__main__":
    run_load_test()
