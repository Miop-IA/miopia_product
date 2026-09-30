document.getElementById("btnRead").addEventListener("click", async () => {
  const outputDiv = document.getElementById("output");
  outputDiv.innerText = "Lendo texto da página...";

  // 1. Obtém a guia ativa e focada no momento
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab) {
    outputDiv.innerText = "Nenhuma guia ativa encontrada.";
    return;
  }

  // 2. Envia mensagem ao content script da guia ativa
  chrome.tabs.sendMessage(tab.id, { action: "READ_NEWS" }, (response) => {
    if (chrome.runtime.lastError) {
      outputDiv.innerText = "Erro: certifique-se de estar em uma página web (HTTP/HTTPS) válida e recarregue-a.";
      return;
    }

    if (response && response.text) {
      outputDiv.innerText = response.text;
    } else {
      outputDiv.innerText = "Nenhum texto pôde ser lido nesta página.";
    }
  });
});