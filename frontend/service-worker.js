/**
 * Service Worker do Miop.IA v2
 * Gerencia injeção de content script e mensagens da extensão.
 */
chrome.action.onClicked.addListener((tab) => {
  // Fallback: se o popup não abrir por algum motivo
  console.log('[Miop.IA] Extensão clicada');
});

// Listener para mensagens do popup ou content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'INJECT_CONTENT_SCRIPT') {
    chrome.scripting.executeScript({
      target: { tabId: sender.tab.id },
      files: ['src/content.js']
    }).then(() => {
      sendResponse({ success: true });
    }).catch((err) => {
      sendResponse({ success: false, error: err.message });
    });
    return true; // Keep channel open for async response
  }
});