/**
 * Content Script do MiopIA
 * Arquitetura Híbrida de Extração:
 * 1. Seletores Específicos Prioritários (altíssima velocidade e aderência aos portais brasileiros)
 * 2. Fallback Inteligente Mozilla Readability (densidade textual e pontuação semântica para qualquer portal)
 * 3. Filtro Dinâmico de Link Density (remove caixas de 'leia mais', carrosséis e banners sem depender de classes CSS)
 * 4. Preservação Especializada MiopIA: restaura aspas geradas por CSS (“ ”), travessões de citação (— Autor)
 * 5. Sanitização de Ruídos e Boilerplates (anúncios, disclaimers, players, chamadas de canal)
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
        sendResponse({ success: false, error: err.message, text: "", words: 0, paragraphs: 0 });
      }
    }
    return true;
  });
}

// 1. Padrões de frases e ruídos a serem descartados
const BOILERPLATE_REGEX = [
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

// Seletores estruturais estáticos de ruído
const NOISE_SELECTORS = [
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
  // Disclaimers, caixas de opinião institucional, reportar erro (específicos para não colidir com reportagem)
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

/**
 * Função principal de extração com arquitetura híbrida
 */
function extractNewsContent() {
  const pageTitle = getNewsTitle();
  let extractedTitle = pageTitle;
  let targetContainer = null;
  let extractionMethod = "specific_selector";

  // 1. Camada 1: Tenta seletores específicos mapeados dos grandes portais
  const specific = trySpecificSelectors(document);
  if (specific) {
    targetContainer = specific.container;
    extractionMethod = `specific_selector (${specific.selector})`;
  }

  // 2. Camada 2: Se nenhum seletor casou, executa o motor Mozilla Readability
  if (!targetContainer) {
    const readable = tryReadability(document);
    if (readable && readable.container) {
      targetContainer = readable.container;
      extractionMethod = "mozilla_readability";
      if (!extractedTitle && readable.title) {
        extractedTitle = readable.title;
      }
    }
  }

  // 3. Camada 3: Fallback semântico (article, main, body)
  if (!targetContainer) {
    const fallbackEl = document.querySelector('article') || document.querySelector('main') || document.body;
    if (fallbackEl) {
      targetContainer = fallbackEl.cloneNode(true);
      extractionMethod = "semantic_fallback";
    }
  }

  if (!targetContainer) {
    return { title: extractedTitle, text: "", words: 0, paragraphs: 0, method: "none" };
  }

  // 4. Limpeza de ruído e eliminação por Densidade de Links (Link Density)
  cleanNoise(targetContainer);
  filterHighLinkDensityNodes(targetContainer);

  // 5. Preservação de citações e aspas visuais (MiopIA)
  formatQuotesAndCitations(targetContainer);

  // Normaliza quebras de linha em tags <br>
  targetContainer.querySelectorAll('br').forEach((br) => {
    br.replaceWith(document.createTextNode('\n'));
  });

  // 6. Extração dos parágrafos
  let result = extractCleanParagraphs(targetContainer);

  // Se a extração por seletor específico resultou em menos de 30 palavras, tenta Readability como fallback de resgate
  if (result.words < 30 && extractionMethod.startsWith("specific_selector")) {
    const readableRescue = tryReadability(document);
    if (readableRescue && readableRescue.container) {
      cleanNoise(readableRescue.container);
      filterHighLinkDensityNodes(readableRescue.container);
      formatQuotesAndCitations(readableRescue.container);
      readableRescue.container.querySelectorAll('br').forEach((br) => {
        br.replaceWith(document.createTextNode('\n'));
      });
      const rescueResult = extractCleanParagraphs(readableRescue.container);
      if (rescueResult.words > result.words) {
        return {
          title: extractedTitle || readableRescue.title,
          text: rescueResult.text,
          words: rescueResult.words,
          paragraphs: rescueResult.paragraphs,
          method: "readability_rescue"
        };
      }
    }
  }

  return {
    title: extractedTitle,
    text: result.text,
    words: result.words,
    paragraphs: result.paragraphs,
    method: extractionMethod
  };
}

/**
 * Camada 1: Testa seletores específicos conhecidos
 */
function trySpecificSelectors(doc) {
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
    '.text'
  ];

  for (const selector of bodySelectors) {
    const elements = doc.querySelectorAll(selector);
    for (const el of elements) {
      const pCount = el.querySelectorAll('p').length;
      const textLen = el.innerText ? el.innerText.trim().length : (el.textContent ? el.textContent.trim().length : 0);
      if (pCount >= 2 && textLen > 150) {
        return {
          selector: selector,
          container: el.cloneNode(true)
        };
      }
    }
  }
  return null;
}

/**
 * Camada 2: Fallback com motor Mozilla Readability
 */
function tryReadability(doc) {
  try {
    const ReadabilityClass = window.Readability || (typeof Readability !== 'undefined' ? Readability : null);
    if (!ReadabilityClass) {
      return null;
    }

    const documentClone = doc.cloneNode(true);
    const reader = new ReadabilityClass(documentClone, {
      charThreshold: 100,
      keepClasses: false
    });
    const article = reader.parse();

    if (article && article.content) {
      const container = document.createElement('div');
      container.innerHTML = article.content;
      return {
        title: article.title || "",
        container: container,
        byline: article.byline || ""
      };
    }
  } catch (err) {
    console.warn("[MiopIA] Readability fallback error:", err);
  }
  return null;
}

/**
 * Filtro de Densidade de Links (Link Density):
 * Remove automaticamente caixas de 'leia mais', carrosséis de artigos relacionados,
 * caixas de patrocinados e menus, sem precisar conhecer suas classes CSS.
 */
function filterHighLinkDensityNodes(root) {
  const candidates = root.querySelectorAll('div, ul, ol, section, aside, nav, p');
  candidates.forEach((el) => {
    if (el === root || !el.parentNode) return;

    // Não remove elementos que contenham citações explícitas
    if (el.querySelector('blockquote, cite, q')) return;

    const text = (el.innerText || el.textContent || '').trim();
    if (text.length < 25) return;

    let linkTextLength = 0;
    el.querySelectorAll('a').forEach((a) => {
      linkTextLength += (a.innerText || a.textContent || '').trim().length;
    });

    const density = linkTextLength / text.length;
    // Se mais de 45% do texto é formado por links, é navegação/propaganda
    if (density > 0.45) {
      el.remove();
    }
  });
}

/**
 * Remove ruídos estruturais conhecidos
 */
function cleanNoise(container) {
  NOISE_SELECTORS.forEach((sel) => {
    try {
      container.querySelectorAll(sel).forEach((el) => el.remove());
    } catch (_) {}
  });
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

/**
 * Higieniza e extrai os parágrafos com descarte de boilerplates
 */
function extractCleanParagraphs(clone) {
  const pElements = Array.from(clone.querySelectorAll('p'));
  const validParagraphs = [];

  for (const p of pElements) {
    const raw = (p.innerText || p.textContent || '').trim();

    if (raw.length < 25) continue;
    if (BOILERPLATE_REGEX.some((rx) => rx.test(raw))) continue;

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
      .filter((l) => !BOILERPLATE_REGEX.some((rx) => rx.test(l)));
    text = rawLines.join('\n\n');
  }

  const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
  return {
    text,
    words,
    paragraphs: validParagraphs.length || (text ? text.split('\n\n').length : 0)
  };
}

function getNewsTitle() {
  const h1 = document.querySelector('h1');
  if (h1 && h1.innerText && h1.innerText.trim()) {
    return h1.innerText.trim();
  }
  return document.title || "";
}