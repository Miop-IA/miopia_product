console.log("🔄 Popup JS carregado no navegador.");

document.addEventListener("DOMContentLoaded", () => {
    console.log("DOMContentLoaded disparado.");
    
    const btnAnalisar = document.getElementById("analyze-btn");
    const divStatus = document.getElementById("status-message");
    const divNoticia = document.getElementById("news-content-preview");

    if (!btnAnalisar) {
        console.error("❌ ERRO: Elemento com ID #analyze-btn não foi encontrado no popup.html!");
        return;
    }

    btnAnalisar.addEventListener("click", async () => {
        console.log("👉 Botão de análise clicado.");
        btnAnalisar.disabled = true;

        if (divStatus) {
            divStatus.textContent = "A ler artigo e validar contexto...";
            divStatus.style.color = "#333";
        }
        if (divNoticia) divNoticia.textContent = "";

        try {
            // 1. Obtém a aba ativa
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            console.log("📌 Aba ativa encontrada:", tab?.url);

            if (!tab || !tab.id || !tab.url || tab.url.startsWith("chrome://") || tab.url.startsWith("chrome-extension://")) {
                throw new Error("Esta página não pode ser analisada (abra um site HTTP/HTTPS comum).");
            }

            // 2. Executa a injeção do script na aba
            console.log("🚀 Executando scripting.executeScript na aba ID:", tab.id);
            const injectionResults = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                func: extractDataInPage
            });

            console.log("📦 Resposta bruta do executeScript:", injectionResults);

            // Acessa o elemento [0] do array retornado
            const resultObj = (injectionResults && injectionResults.length > 0) ? injectionResults[0] : null;
            const result = resultObj ? resultObj.result : null;

            if (!result || !result.success) {
                throw new Error(result?.error || "Erro ao extrair dados da página.");
            }

            const { hash, text } = result;
            console.log("🔑 Hash da notícia:", hash);
            console.log("📝 Texto extraído (caracteres):", text.length);

            const storageKey = `ctx_${hash}`;
            const TTL_MS = 24 * 60 * 60 * 1000; // 24 horas

            // 3. Checagem de limite no chrome.storage.local
            const storageData = await chrome.storage.local.get([storageKey]);
            const data = storageData[storageKey];
            const now = Date.now();

            if (data && (now - data.timestamp < TTL_MS)) {
                throw new Error("Limite atingido: Já analisou esta notícia nas últimas 24 horas.");
            }

            // 4. Armazena no storage
            await chrome.storage.local.set({ [storageKey]: { timestamp:  now } });

            // 5. Sucesso e exibição do resultado
            if (divStatus) {
                divStatus.textContent = "✅ Contexto validado com sucesso!";
                divStatus.style.color = "green";
            }

            if (divNoticia) {
                divNoticia.textContent = text.substring(0, 300) + "...";
            }

        } catch (err) {
            console.error("🛑 Erro na execução:", err);
            if (divStatus) {
                divStatus.textContent = "🛑 " + err.message;
                divStatus.style.color = "red";
            }
        } finally {
            btnAnalisar.disabled = false;
        }
    });
});

/**
 * Função injetada na página aberta
 */
function extractDataInPage() {
    try {
        let url = window.location.href;
        const canonical = document.querySelector('link[rel="canonical"]');
        if (canonical && canonical.href) {
            url = canonical.href;
        } else {
            const ogUrl = document.querySelector('meta[property="og:url"]');
            if (ogUrl && ogUrl.content) {
                url = ogUrl.content;
            } else {
                // Uso dos índices [0] para limpar a URL corretamente
                url = url.split('?')[0].split('#')[0];
            }
        }

        // Tenta capturar parágrafos em seletores comuns de notícias
        const articleElements = document.querySelectorAll('article p, main p, .noticia p, .post-content p, .m-content p');
        let paragraphs = articleElements.length > 0 ? articleElements : document.querySelectorAll('p');

        let text = Array.from(paragraphs)
            .map(p => p.innerText.trim())
            .filter(p => p.length > 20)
            .join('\n\n');

        // Fallback para obter o container principal caso os <p> falhem
        if (!text || text.length < 50) {
            const mainContainer = document.querySelector('article') || document.querySelector('main') || document.body;
            text = mainContainer ? mainContainer.innerText.trim() : "";
        }

        if (!text) {
            return { success: false, error: "Não foi possível extrair o texto desta notícia." };
        }

        // Gera o Hash da URL
        let hashNum = 0;
        for (let i = 0; i < url.length; i++) {
            const char = url.charCodeAt(i);
            hashNum = ((hashNum << 5) - hashNum) + char;
            hashNum = hashNum & hashNum;
        }
        const hash = Math.abs(hashNum).toString(16);

        return { success: true, hash: hash, text: text };
    } catch (e) {
        return { success: false, error: e.message };
    }
}