/**
 * Content Script do MiopIA
 * Extrai estritamente o corpo textual da notícia, preservando citações,
 * aspas (“ ”), autoria de falas (— Autor) e filtrando ruídos (anúncios, players, disclaimers).
 */

// Evita injeção múltipla de listeners
if (!window.__MIOPIA_CONTENT_SCRIPT_INITIALIZED__) {
  window.__MIOPIA_CONTENT_SCRIPT_INITIALIZED__ = true;

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "READ_NEWS") {
      try {
        const result = extractNewsContent();
        sendResponse({ success: true, ...result });
      } catch (err) {
        console.error("[MiopIA] Erro na extração de texto:", err);
        sendResponse({ success: false, error: err.message, text: "" });
      }
    }
    return true;
  });
}

function extractNewsContent() {
  const title = getNewsTitle();

  // 1. Seletores específicos do corpo de texto da notícia
  const bodySelectors = [
    '[itemprop="articleBody"]',
    '.c-news__body',           // UOL / Folha
    '.news-body',              // Estadão
    '.content-text',           // G1 / Globo
    '.mc-article-body',        // G1 Globo
    '.materia-conteudo',       // R7
    '.conteudo-materia',       // Metrópoles / R7
    '.m-article__content',     // Metrópoles
    '.article__content',
    '.article__body',
    '.article-body',
    '.article-text',
    '.entry-content',
    '.story-body',
    '.post-content',           // CNN Brasil / Poder360 / WordPress
    '.n--noticia__content',    // Estadão
    'article .text',           // UOL
    '.text',
    'article',
    'main',
    '#main-content'
  ];

  let targetElement = null;
  for (const selector of bodySelectors) {
    const elements = document.querySelectorAll(selector);
    for (const el of elements) {
      const pCount = el.querySelectorAll('p').length;
      const textLen = el.innerText ? el.innerText.trim().length : 0;
      if (pCount >= 2 && textLen > 150) {
        targetElement = el;
        break;
      }
    }
    if (targetElement) break;
  }

  if (!targetElement) {
    targetElement = document.querySelector('article') || document.querySelector('main') || document.body;
  }

  if (!targetElement) {
    return { title, text: "" };
  }

  // 2. Clona o elemento para higienização em memória
  const clone = targetElement.cloneNode(true);

  // Lista de seletores de ruído para remoção imediata
  const noiseSelectors = [
    // Tags estruturais e mídia
    'header', 'footer', 'nav', 'aside', 'noscript', 'script', 'style', 'svg', 'iframe', 'form', 'button',
    'audio', 'video', 'time', 'figure', 'figcaption',
    // Players de áudio/vídeo e controles (ex: "0:00 / 0:00x")
    '[class*="player"]', '[class*="audio"]', '[class*="video"]', '[id*="player"]',
    // Cabeçalho institucional, chapéu, editoria, data
    '.c-news__header', '.article-header', '[class*="breadcrumb"]', '[class*="chapeu"]', '[class*="kicker"]',
    '.author-info', '.byline', '[class*="byline"]', '[class*="data-publicacao"]',
    // Anúncios, publicidade e banners
    '[class*="ad-"]', '[class*="ads-"]', '[class*="publicidade"]', '[class*="anuncio"]', '[class*="banner"]',
    '[id*="ad-"]', '[id*="ads-"]', '[id*="publicidade"]', '[id*="anuncio"]', '[id*="banner"]',
    '.ad', '.ads',
    // Legendas e créditos
    '[class*="caption"]', '[class*="legenda"]', '[class*="credito"]',
    // Disclaimers, caixas de opinião institucional, reportar erro
    '[class*="disclaimer"]', '[class*="aviso"]', '[class*="comunicar"]', '[class*="erro"]',
    '[class*="opiniao-box"]', '[class*="report-error"]', '[class*="reportar-erro"]', '[class*="report-button"]', '[class*="reportar"]',
    // Comentários
    '[class*="comment"]', '[class*="comentario"]', '[id*="comment"]', '[id*="comentario"]',
    // Redes sociais e newsletters
    '[class*="share"]', '[class*="social"]', '[class*="compartilh"]', '[class*="newsletter"]',
    // Links recomendados e tags
    '[class*="relacionad"]', '[class*="recommended"]', '[class*="tags"]', '[class*="tag-"]',
    '[class*="veja-mais"]', '[class*="leia-mais"]',
    // Caixas de resumo automático de IA
    '[class*="resumo-ia"]', '[class*="ai-summary"]', '[class*="ia-summary"]'
  ];

  noiseSelectors.forEach((sel) => {
    clone.querySelectorAll(sel).forEach((el) => el.remove());
  });

  // 3. RECUPERAÇÃO DE CITAÇÕES E ASPAS VISUAIS
  // Em muitos portais (como o UOL), as aspas e o travessão do autor são gerados via CSS (::before/::after).
  // Como clones DOM não herdam pseudo-elementos CSS, restauramos as aspas e o travessão explicitamente.
  formatQuotesAndCitations(clone);

  // Converte tags <br> em quebras de linha reais para não colar palavras
  clone.querySelectorAll('br').forEach((br) => {
    br.replaceWith(document.createTextNode('\n'));
  });

  // 4. Padrões de frases e ruídos a serem descartados
  const boilerplateRegex = [
    /^continua\s+após\s+a\s+publicidade/i,
    /^publicidade/i,
    /^comunicar\s+erro/i,
    /^deixe\s+seu\s+comentário/i,
    /^o\s+autor\s+da\s+mensagem.*responsável/i,
    /^este\s+texto\s+não\s+reflete.*opinião/i,
    /^texto\s+em\s+que\s+o\s+autor\s+apresenta.*opiniões/i,
    /^leia\s+as\s+regras\s+de\s+uso/i,
    /^veja\s+horários\s+das\s+lives/i,
    /^imagem:\s*/i,
    /^foto:\s*/i,
    /^\d{1,2}:\d{2}\s*\/\s*\d{1,2}:\d{2}/i, // 0:00 / 0:00
    /^veja\s+também/i,
    /^leia\s+também/i,
    /^leia\s+mais/i,
    /^todos\s+os\s+direitos\s+reservados/i,
    /^compartilhe:\s*/i,
    /^adicione\s+como\s+fonte\s+preferencial/i,
    /^produzido\s+pela\s+ri7a/i,
    /^gerando\s+resumo/i,
    /^siga\s+(o|a)?\s*(nosso|nossa)?\s*canal/i,
    /^receba\s+(no|as)\s+whatsapp/i,
    /^conteúdo\s+criado\s+em\s+parceria/i,
    /^clique\s+aqui\s+para\s+seguir/i,
    /^fonte:\s*/i
  ];

  // 5. Extração baseada em parágrafos reais (<p>)
  const pElements = Array.from(clone.querySelectorAll('p'));
  const validParagraphs = [];

  for (const p of pElements) {
    const raw = (p.innerText || p.textContent || '').trim();

    if (raw.length < 25) continue;

    const isBoilerplate = boilerplateRegex.some((rx) => rx.test(raw));
    if (isBoilerplate) continue;

    validParagraphs.push(raw);
  }

  let text = '';
  if (validParagraphs.length >= 2) {
    text = validParagraphs.join('\n\n');
  } else {
    const rawLines = (clone.innerText || clone.textContent || '')
      .split('\n')
      .map((l) => l.trim())
      .filter((l) => l.length >= 25)
      .filter((l) => !boilerplateRegex.some((rx) => rx.test(l)));
    text = rawLines.join('\n\n');
  }

  return {
    title,
    text
  };
}

/**
 * Restaura elementos de citação que os portais estilizam via CSS:
 * - Aspas vermelhas em volta de <cite> (“ ... ”)
 * - Travessão / hífen antes do autor (— Autor)
 */
function formatQuotesAndCitations(container) {
  // 1. Aspas em torno de <cite> (depoimentos em destaque)
  container.querySelectorAll('cite').forEach((cite) => {
    let t = (cite.innerText || cite.textContent || '').trim();
    if (!t) return;
    if (!/^[\u201c\u201d"«'']/.test(t)) {
      t = `“${t}”`;
    }
    cite.textContent = t;
  });

  // 2. Travessão antes de autores associados à citação (ex: <cite>...</cite><br><strong>PVC</strong>)
  container.querySelectorAll('cite ~ strong, p.bullet > strong, cite + br + strong, blockquote cite, blockquote footer').forEach((authorEl) => {
    let a = (authorEl.innerText || authorEl.textContent || '').trim();
    if (!a || a.length > 50) return; // Nomes de autores são curtos
    if (!/^[—–\-]/.test(a)) {
      authorEl.textContent = `— ${a}`;
    }
  });

  // 3. Aspas em tags <q> (inline quotes)
  container.querySelectorAll('q').forEach((q) => {
    let t = (q.innerText || q.textContent || '').trim();
    if (!t) return;
    if (!/^[\u201c\u201d"«'']/.test(t)) {
      q.textContent = `“${t}”`;
    }
  });

  // 4. Tags <blockquote> que não contenham <cite> interno
  container.querySelectorAll('blockquote').forEach((bq) => {
    if (!bq.querySelector('cite')) {
      let t = (bq.innerText || bq.textContent || '').trim();
      if (t && !/^[\u201c\u201d"«'']/.test(t)) {
        bq.textContent = `“${t}”`;
      }
    }
  });
}

function getNewsTitle() {
  const h1 = document.querySelector('h1');
  if (h1 && h1.innerText.trim()) {
    return h1.innerText.trim();
  }
  return document.title || "";
}