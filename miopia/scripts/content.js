// Listener para receber requisições enviadas pelo popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "READ_NEWS") {
    // Busca os seletores mais comuns de artigos de notícias
    const articleElement =
      document.querySelector("article") || 
      document.querySelector("main") || 
      document.body;

    // Obtém o texto exato contido no nó selecionado
    const extractedText = articleElement ? articleElement.innerText.trim() : "";

    // Envia o texto de volta para o popup
    sendResponse({ text: extractedText });
  }
  return true; // Mantém o canal de resposta assíncrona aberto
});