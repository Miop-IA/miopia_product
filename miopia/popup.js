const API_URL = "http://localhost:8000/analisar";

document.getElementById("btnRead").addEventListener("click", async () => {
  const btn = document.getElementById("btnRead");
  const statusDiv = document.getElementById("status");
  const outputOriginal = document.getElementById("outputOriginal");
  const outputTruncated = document.getElementById("outputTruncated");
  const outputFeatures = document.getElementById("outputFeatures");

  btn.disabled = true;
  statusDiv.style.color = "#1a73e8";
  statusDiv.innerText = "⏳ Extraindo texto da página ativa...";
  outputOriginal.innerText = "Lendo...";
  outputTruncated.innerText = "Aguardando backend...";
  outputFeatures.innerText = "Aguardando backend...";

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

    if (!tab || !tab.id) {
      throw new Error("Nenhuma aba ativa encontrada.");
    }

    if (!tab.url || (!tab.url.startsWith("http://") && !tab.url.startsWith("https://"))) {
      throw new Error("Abra uma página web HTTP/HTTPS válida para realizar a leitura.");
    }

    // Envia mensagem ao content script (com tentativa de injeção direta caso necessário)
    let response;
    try {
      response = await sendMessagePromise(tab.id, { action: "READ_NEWS" });
    } catch (msgErr) {
      if (msgErr.message.includes("Receiving end does not exist") || msgErr.message.includes("Could not establish connection")) {
        await chrome.scripting.executeScript({
          target: { tabId: tab.id },
          files: ["scripts/readability.js", "scripts/content.js"]
        });
        await new Promise((resolve) => setTimeout(resolve, 100));
        response = await sendMessagePromise(tab.id, { action: "READ_NEWS" });
      } else {
        throw msgErr;
      }
    }

    const rawText = response ? response.text : "";
    if (!rawText || rawText.trim().length === 0) {
      throw new Error("Nenhum texto pôde ser lido nesta página pelo seletor de conteúdo.");
    }

    const rawWordCount = rawText.trim().split(/\s+/).length;
    outputOriginal.innerText = `[Tamanho: ${rawText.length} caracteres | ${rawWordCount} palavras]\n\n${rawText}`;

    statusDiv.innerText = "🔬 Enviando texto ao backend para truncamento e extração de features...";

    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        url: tab.url || "",
        texto: rawText,
        text: rawText
      })
    });

    if (!res.ok) {
      const errorText = await res.text();
      throw new Error(`Erro na API (${res.status}): ${errorText}`);
    }

    const data = await res.json();

    // 1. Exibe o texto truncado processado pelo backend
    const textoTruncado = data.texto_truncado || rawText;
    const totalPalavras = data.total_palavras_truncado || (textoTruncado ? textoTruncado.trim().split(/\s+/).length : 0);
    outputTruncated.innerText = `[Tokens/Palavras no modelo: ${totalPalavras}]\n\n${textoTruncado}`;

    // 2. Exibe o diagnóstico e as features extraídas
    const featuresDict = data.features || data.metricas || {};
    const resultadoExibicao = {
      ...(data.faixa ? {
        faixa: data.faixa,
        prob_suspeita: data.prob_suspeita,
        orientacao: data.orientacao
      } : {}),
      features: featuresDict
    };
    outputFeatures.innerText = JSON.stringify(resultadoExibicao, null, 2);

    statusDiv.style.color = "#137333";
    statusDiv.innerText = "✅ Texto extraído, truncado e analisado com sucesso!";

  } catch (err) {
    statusDiv.style.color = "#d93025";
    statusDiv.innerText = `❌ Erro: ${err.message}`;
    if (outputTruncated.innerText.includes("Aguardando")) {
      outputTruncated.innerText = "Falha antes de enviar ao backend.";
    }
    if (outputFeatures.innerText.includes("Aguardando")) {
      outputFeatures.innerText = "Nenhuma feature gerada.";
    }
  } finally {
    btn.disabled = false;
  }
});

function sendMessagePromise(tabId, message) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      if (chrome.runtime.lastError) {
        return reject(new Error(chrome.runtime.lastError.message));
      }
      resolve(response);
    });
  });
}