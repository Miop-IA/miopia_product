// ============================================================
// Miop.IA — Chrome Extension Popup
// ============================================================

(function () {
  "use strict";

  /* ---- DOM refs ---- */
  var modal           = document.getElementById("modal");
  var badgeSite       = document.getElementById("badgeSite");
  var loadingState    = document.getElementById("loadingState");
  var errorState      = document.getElementById("errorState");
  var errorMessage    = document.getElementById("errorMessage");
  var conteudoAnalise = document.getElementById("conteudoAnalise");
  var tituloNoticia   = document.getElementById("tituloNoticia");
  var corpoNoticia    = document.getElementById("corpoNoticia");
  var destaquesGrid   = document.getElementById("destaquesGrid");
  var botoesAvaliacao = document.getElementById("botoesAvaliacao");
  var botaoEnviar     = document.getElementById("botaoEnviar");
  var telaPosEnvio    = document.getElementById("telaPosEnvio");
  var botoesFinalEl   = document.getElementById("botoesAvaliacaoFinal");

  var valorSelecionado = null;
  var noticiaId    = null;
  var CLIENTE_ID   = null; // UUID anônimo por instalação (chrome.storage.local)
  var dominioAtual = "";

  /* ---- helpers ---- */
  function mostra(el)  { if (el) el.hidden = false; }
  function esconde(el) { if (el) el.hidden = true; }

  function defineSite(dominio) {
    dominioAtual = dominio || "";
    if (badgeSite) badgeSite.textContent = dominioAtual;
  }

  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /* ---- extrai primeiras 3 frases do texto (até ~450 chars) ---- */
  function extrairSnippet(texto) {
    if (!texto) return "";
    var frases = texto.match(/[^.!?]+[.!?]+/g) || [texto];
    var snippet = frases.slice(0, 3).join(" ").trim();
    if (snippet.length > 450) {
      snippet = snippet.slice(0, 450).replace(/\s+\S+$/, "…");
    }
    return snippet;
  }

  /* ---- mapa de fallback: chave métrica → label + desc ---- */
  var MAPA = {
    trunc_pausality:            { label: "Pausas excessivas",        desc: "Uso intenso de vírgulas" },
    trunc_emotiveness:          { label: "Tom emotivo",              desc: "Linguagem carregada de emoção" },
    trunc_diversity:            { label: "Vocabulário repetitivo",   desc: "Pouca variedade de palavras" },
    trunc_upper_case_density:   { label: "MAIÚSCULAS em excesso",    desc: "Uso exagerado de CAIXA ALTA" },
    trunc_verb_density:         { label: "Muitos verbos",            desc: "Alta densidade verbal" },
    trunc_noun_density:         { label: "Muitos substantivos",      desc: "Alta densidade de substantivos" },
    trunc_adj_density:          { label: "Muitos adjetivos",         desc: "Linguagem carregada de opinião" },
    trunc_adv_density:          { label: "Muitos advérbios",         desc: "Excesso de advérbios" },
    trunc_pron_density:         { label: "Personalização",           desc: "Muitos pronomes pessoais" },
    link_density:               { label: "Links suspeitos",          desc: "Alta densidade de links" },
    rc_spelling_errors:         { label: "Erros ortográficos",       desc: "Problemas de grafia" },
    rc_modal_verbs_density:     { label: "Tom de certeza",           desc: "Verbos modais de convicção" },
    rc_subj_imp_verbs_density:  { label: "Tom imperativo",           desc: "Verbos no imperativo" },
    rc_pron_1_2_sing_density:   { label: "1.ª pessoa do singular",  desc: "Relato pessoal (eu/você)" },
    rc_pron_1_plur_density:     { label: "1.ª pessoa do plural",    desc: "Relato coletivo (nós)" }
  };

  /* ---- referência do treino (média, desvio) para dizer o que é "acima do normal" ----
     Valores de scaler_estilo do modelo v1 (Fake.br). Cada métrica tem escala própria:
     um limiar fixo único (0.3) dispara sempre para umas e nunca para outras. */
  var REFERENCIA = {
    trunc_pausality:           [2.706, 0.909],
    trunc_emotiveness:         [0.167, 0.076],
    trunc_diversity:           [0.591, 0.086],
    trunc_upper_case_density:  [0.016, 0.017],
    trunc_verb_density:        [0.132, 0.031],
    trunc_noun_density:        [0.306, 0.052],
    trunc_adj_density:         [0.041, 0.021],
    trunc_adv_density:         [0.030, 0.018],
    trunc_pron_density:        [0.028, 0.019],
    link_density:              [0.001, 0.003],
    rc_spelling_errors:        [0.003, 0.008],
    rc_modal_verbs_density:    [0.020, 0.011],
    rc_subj_imp_verbs_density: [0.006, 0.006],
    rc_pron_1_2_sing_density:  [0.002, 0.005],
    rc_pron_1_plur_density:    [0.001, 0.002]
  };
  var Z_SINAL = 1.5;
  // Métricas em que o sinal é o valor BAIXO (diversidade baixa = vocabulário repetitivo)
  var SINAL_INVERTIDO = { trunc_diversity: true };

  // Quantos desvios a métrica está na direção "suspeita"; <= 0 significa normal
  function intensidadeSinal(chave, val) {
    var ref = REFERENCIA[chave];
    if (!ref || typeof val !== "number" || !isFinite(val)) return 0;
    var z = (val - ref[0]) / ref[1];
    return SINAL_INVERTIDO[chave] ? -z : z;
  }
  function sinalAtivo(chave, val) { return intensidadeSinal(chave, val) > Z_SINAL; }

  /* ---- padrões de destaque por métrica (suspeito) ---- */
  var PADROES_HIGHLIGHT = {
    trunc_upper_case_density: {
      regex: function() { return /[A-ZÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÃÕÇÑ]{3,}/g; },
      desc: "Caixa alta pode ser recurso de sensacionalismo"
    },
    trunc_emotiveness: {
      regex: function() { return /\b(urgente|alerta|bomba|chocante|absurdo|imperdível|inacreditável|alarmante|catastrófico|horror|escândalo|denúncia|traição|golpe|caos|ameaça|chocante|revoltante)\b/gi; },
      desc: "Linguagem emotiva e alarmante"
    },
    rc_modal_verbs_density: {
      regex: function() { return /\b(pode|deve|precisa|quer|poderia|deveria|precisaria|vai|irá|deverá|terá)\b/gi; },
      desc: "Verbos modais transmitem certeza sem evidências"
    },
    rc_subj_imp_verbs_density: {
      regex: function() { return /\b(veja|confira|saiba|acredite|descubra|compartilhe|repasse|corra|denuncie|exija|leia|assista|clique)\b/gi; },
      desc: "Verbos no imperativo induzem ação emocional"
    },
    rc_pron_1_2_sing_density: {
      regex: function() { return /\b(eu|me|mim|comigo|você|tu|te|ti|contigo)\b/gi; },
      desc: "Narração em 1.ª pessoa mistura relato com opinião"
    },
    rc_pron_1_plur_density: {
      regex: function() { return /\b(nós|nos|conosco|nosso|nossa|nossos|nossas)\b/gi; },
      desc: "Linguagem coletiva pode criar falsa identificação"
    },
    trunc_adj_density: {
      regex: function() { return /\b(especialistas?|pesquisadores?|cientistas?|autoridades?|fontes?|analistas?|peritos?|experts?|médicos?)\b/gi; },
      desc: "Referência vaga a autoridades sem identificação"
    },
    trunc_noun_density: {
      regex: function() { return /\b\d[\d.,]*\s*(?:mil|milhões?|bilhões?|%)?\s+(?:pessoas|casos|vítimas|mortes|jovens|adultos|crianças|pacientes|usuários|alunos|brasileiros)\b/gi; },
      desc: "Número sem fonte ou metodologia citada"
    }
  };

  /* ---- padrões de destaque positivo (faixa Confiavel) ---- */
  var PADROES_POSITIVOS = [
    {
      regex: function() { return /\b(?:segundo|de acordo com|afirmou|disse|relatou|informou|confirmou)\s+(?:o\s+|a\s+|os\s+|as\s+)?[A-Z][a-zA-Záéíóúâêîôûãõç]+/g; },
      desc: "Declaração atribuída a uma fonte identificada"
    },
    {
      regex: function() { return /\b(?:IBGE|OMS|ONU|ANVISA|SUS|Inep|Ipea|Fiocruz|CNJ|STF|TSE|Banco\s+Central|Receita\s+Federal|Agência\s+Brasil)\b/g; },
      desc: "Referência a órgão oficial verificável"
    }
  ];

  var FONTES_CONFIAVEIS = [
    "agenciabrasil.ebc.com.br", "g1.globo.com", "cnnbrasil.com.br",
    "folha.uol.com.br", "estadao.com.br", "bbc.com", "reuters.com",
    "apnews.com", "nexojornal.com.br", "aosfatos.org", "agencia.fapesp.br",
    "cartacapital.com.br", "correiobraziliense.com.br"
  ];

  /* ---- aplica destaques inline no snippet (retorna HTML seguro) ---- */
  function aplicarDestaques(snippet, metricas, faixa) {
    var html = escapeHtml(snippet);

    if (faixa === "Confiavel") {
      PADROES_POSITIVOS.forEach(function(p) {
        html = html.replace(p.regex(), function(m) {
          return '<span class="destaque-positivo">' + m + '</span>';
        });
      });
      return html;
    }

    Object.keys(PADROES_HIGHLIGHT).forEach(function(chave) {
      if (!sinalAtivo(chave, metricas[chave])) return;
      var info = PADROES_HIGHLIGHT[chave];
      html = html.replace(info.regex(), function(m) {
        return '<span class="destaque-suspeito">' + m + '</span>';
      });
    });

    return html;
  }

  /* ---- extrai pares {palavra, desc, tipo} para as pills ---- */
  function extrairMatchesPills(snippet, metricas, faixa) {
    var pills = [];
    var vistos = {};

    function addPill(palavra, desc, tipo) {
      var key = palavra.toLowerCase().trim();
      if (vistos[key]) return;
      vistos[key] = true;
      pills.push({ palavra: palavra.trim(), desc: desc, tipo: tipo });
    }

    if (faixa === "Confiavel") {
      PADROES_POSITIVOS.forEach(function(p) {
        var matches = snippet.match(p.regex()) || [];
        matches.slice(0, 2).forEach(function(m) { addPill(m, p.desc, "positivo"); });
      });

      if (dominioAtual && FONTES_CONFIAVEIS.some(function(f) { return dominioAtual.indexOf(f) !== -1; })) {
        addPill(dominioAtual, "Veículo com histórico de jornalismo verificável", "positivo");
      }

      return pills.slice(0, 6);
    }

    Object.keys(PADROES_HIGHLIGHT).forEach(function(chave) {
      if (!sinalAtivo(chave, metricas[chave])) return;
      var info = PADROES_HIGHLIGHT[chave];
      var matches = snippet.match(info.regex()) || [];
      var cnt = 0;
      matches.forEach(function(m) {
        if (cnt >= 2) return;
        addPill(m, info.desc, "suspeito");
        cnt++;
      });
    });

    return pills.slice(0, 8);
  }

  /* ---- renderiza seção de destaques ---- */
  function renderHighlights(metricas, snippetPlain, faixa) {
    destaquesGrid.innerHTML = "";

    if (!metricas || typeof metricas !== "object") {
      destaquesGrid.innerHTML = "<p style='color:#6B6F80;font-size:13px;'>Nenhuma métrica disponível.</p>";
      return;
    }

    var pills = extrairMatchesPills(snippetPlain, metricas, faixa);

    // fallback: exibe pills genéricas de métrica quando nenhuma palavra foi capturada
    if (!pills.length) {
      var ativas = [];
      Object.keys(metricas).forEach(function(chave) {
        var info = MAPA[chave];
        var forca = intensidadeSinal(chave, metricas[chave]);
        if (!info || forca <= Z_SINAL) return;
        ativas.push({ val: forca, info: info });
      });
      ativas.sort(function(a, b) { return b.val - a.val; });

      if (!ativas.length) {
        destaquesGrid.innerHTML = "<p style='color:#6B6F80;font-size:13px;'>Nenhuma característica suspeita predominante.</p>";
        return;
      }

      ativas.slice(0, 4).forEach(function(item) {
        var row = document.createElement("div");
        row.className = "pill-row";
        var t = document.createElement("span");
        t.className = "pill pill-destaque";
        t.textContent = item.info.label;
        var d = document.createElement("span");
        d.className = "pill pill-categoria";
        d.textContent = item.info.desc;
        row.appendChild(t);
        row.appendChild(d);
        destaquesGrid.appendChild(row);
      });
      return;
    }

    pills.forEach(function(p) {
      var row = document.createElement("div");
      row.className = "pill-row";
      var t = document.createElement("span");
      t.className = "pill " + (p.tipo === "positivo" ? "pill-positivo" : "pill-destaque");
      t.textContent = p.palavra;
      var d = document.createElement("span");
      d.className = "pill pill-categoria";
      d.textContent = p.desc;
      row.appendChild(t);
      row.appendChild(d);
      destaquesGrid.appendChild(row);
    });
  }

  /* =============================================================
     API
     ============================================================= */
  function analisar(texto, url, numLinks) {
    return fetch("http://localhost:8000/analisar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto: texto, url: url || "", num_links: numLinks })
    }).then(function(r) {
      if (r.ok) return r.json();
      // Mostra o motivo devolvido pela API (ex.: texto com menos de 30 palavras)
      return r.json().catch(function() { return {}; }).then(function(corpo) {
        var detalhe = corpo && corpo.detail;
        throw new Error(typeof detalhe === "string" ? detalhe : "HTTP " + r.status);
      });
    });
  }

  function obterClienteId() {
    if (CLIENTE_ID) return Promise.resolve(CLIENTE_ID);
    return new Promise(function(resolve) {
      chrome.storage.local.get("miopia_client_id", function(r) {
        var id = r && r.miopia_client_id;
        if (!id) {
          id = crypto.randomUUID(); // 36 caracteres, cabe em client_id String(36); não identifica a pessoa
          chrome.storage.local.set({ miopia_client_id: id });
        }
        CLIENTE_ID = id;
        resolve(id);
      });
    });
  }

  function enviarVoto(noticiaId, avaliacao) {
    return obterClienteId().then(function(clienteId) {
      return fetch("http://localhost:8000/avaliar", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          noticia_id: noticiaId,
          client_id: clienteId,
          avaliacao: avaliacao
        })
      });
    }).then(function(r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  /* =============================================================
     PREENCHER análise com dados da API
     ============================================================= */
  function preencheAnalise(dados, tituloExtraido, texto) {
    noticiaId = dados.id;
    var faixa = dados.faixa || "Atencao";

    // título
    var titulo = dados.title || tituloExtraido || "Análise de notícia";
    if (tituloNoticia) tituloNoticia.textContent = titulo;

    // snippet com destaques inline
    var snippet = extrairSnippet(texto || "");
    if (corpoNoticia) {
      if (snippet) {
        corpoNoticia.innerHTML = aplicarDestaques(snippet, dados.metricas || {}, faixa);
      } else {
        corpoNoticia.textContent = dados.orientacao || "";
      }
    }

    // pills → palavras capturadas
    renderHighlights(dados.metricas, snippet, faixa);
  }

  /* =============================================================
     EXTRAÇÃO DE TEXTO DA PÁGINA
     ============================================================= */
  function consultaPaginaAtiva() {
    chrome.tabs.query({ active: true, currentWindow: true }, function(tabs) {
      if (!tabs || !tabs.length) { mostraErro("Nenhuma aba ativa encontrada."); return; }
      var tab = tabs[0];
      try { defineSite(new URL(tab.url).hostname); } catch(_) { defineSite(tab.url || ""); }
      tentaExtrairViaContentScript(tab);
    });
  }

  function tentaExtrairViaContentScript(tab) {
    chrome.tabs.sendMessage(tab.id, { action: "READ_NEWS" }, function(resposta) {
      if (chrome.runtime.lastError || !resposta || !resposta.success) {
        tentaExtrairViaScripting(tab);
        return;
      }
      if (!resposta.text) { mostraErro("Não foi possível extrair texto da página."); return; }
      enviaParaAPI(resposta.text, tab.url, resposta.title, resposta.links);
    });
  }

  function tentaExtrairViaScripting(tab) {
    if (!chrome.scripting || !chrome.scripting.executeScript) {
      mostraErro("Não foi possível comunicar com a página. Recarregue a aba e tente novamente.");
      return;
    }

    chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: function() {
        var el = document.querySelector("article") || document.querySelector("main") || document.body;
        if (!el) return { success: false, text: "" };

        var clone = el.cloneNode(true);
        var noise = ["header","footer","nav","aside","script","style","noscript","iframe","svg","form","button","audio","video"];
        noise.forEach(function(tag) {
          try { clone.querySelectorAll(tag).forEach(function(e) { e.remove(); }); } catch(_) {}
        });

        var ps = Array.from(clone.querySelectorAll("p"));
        var textos = [];
        ps.forEach(function(p) {
          var t = (p.innerText || p.textContent || "").trim();
          if (t.length > 20) textos.push(t);
        });

        var links = Array.from(clone.querySelectorAll("a[href]")).filter(function(a) {
           var href = a.getAttribute("href") || "";
           return href.length > 0 && !href.startsWith("#") && !href.startsWith("javascript:");
        }).length;

        var texto = textos.join("\n\n") || (clone.innerText || clone.textContent || "").trim();
        return {
          success: true,
          title: (document.querySelector("h1") ? document.querySelector("h1").innerText.trim() : document.title) || "",
          text: texto,
          words: texto.split(/\s+/).filter(Boolean).length,
          paragraphs: textos.length,
          links: links
        };
      }
    }, function(results) {
      if (chrome.runtime.lastError || !results || !results.length || !results[0].result || !results[0].result.success) {
        mostraErro("Não foi possível comunicar com a página. Recarregue a aba e tente novamente.");
        return;
      }
      var d = results[0].result;
      if (!d.text) { mostraErro("Não foi possível extrair texto da página."); return; }
      enviaParaAPI(d.text, tab.url, d.title, d.links);
    });
  }

  function enviaParaAPI(texto, url, tituloExtraido, links) {
    analisar(texto, url, links).then(function(dados) {
      esconde(loadingState);
      esconde(errorState);
      mostra(conteudoAnalise);
      preencheAnalise(dados, tituloExtraido, texto);
    }).catch(function(err) {
      mostraErro(err.message.indexOf("HTTP") === 0 || err.message === "Failed to fetch"
        ? "Erro ao analisar: " + err.message
        : err.message);
    });
  }

  /* =============================================================
     VOTO / AVALIAÇÃO
     ============================================================= */
  function setupBotoesAvaliacao() {
    if (!botoesAvaliacao) return;

    var botoes = botoesAvaliacao.querySelectorAll(".botao-rating");
    botoes.forEach(function(botao) {
      botao.addEventListener("click", function() {
        botoes.forEach(function(b) { b.classList.remove("selecionado"); });
        botao.classList.add("selecionado");
        valorSelecionado = botao.dataset.valor;
        if (botaoEnviar) botaoEnviar.hidden = false;
      });
    });
  }

  function valorParaInt(v) {
    if (v === "verdadeira") return 0;
    if (v === "duvidosa")   return 1;
    return 2;
  }

  function setupBotaoEnviar() {
    if (!botaoEnviar) return;

    botaoEnviar.addEventListener("click", function() {
      if (!valorSelecionado) return;

      var valores = ["verdadeira", "duvidosa", "falsa"];
      botoesFinalEl.innerHTML = "";
      valores.forEach(function(v) {
        var b = document.createElement("button");
        b.className = "botao-rating" + (v === valorSelecionado ? " selecionado" : "");
        b.dataset.valor = v;
        b.disabled = true;
        b.textContent = v.charAt(0).toUpperCase() + v.slice(1);
        botoesFinalEl.appendChild(b);
      });

      if (modal) modal.classList.add("girando");

      if (noticiaId != null) {
        enviarVoto(noticiaId, valorParaInt(valorSelecionado)).catch(function() {});
      }

      setTimeout(function() {
        if (conteudoAnalise) conteudoAnalise.style.display = "none";
        if (telaPosEnvio) telaPosEnvio.hidden = false;
        if (modal) modal.classList.remove("girando");
      }, 320);
    });
  }

  /* =============================================================
     INIT
     ============================================================= */
  function mostraErro(msg) {
    esconde(loadingState);
    mostra(errorState);
    if (errorMessage) errorMessage.textContent = msg || "Ocorreu um erro inesperado.";
    else if (errorState) errorState.textContent = msg || "Ocorreu um erro inesperado.";
  }

  var retryBtn = document.getElementById("retryBtn");
  if (retryBtn) {
    retryBtn.addEventListener("click", function() {
      esconde(errorState);
      mostra(loadingState);
      consultaPaginaAtiva();
    });
  }

  function init() {
    setupBotoesAvaliacao();
    setupBotaoEnviar();
    consultaPaginaAtiva();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
