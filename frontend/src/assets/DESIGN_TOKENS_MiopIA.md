# Design Tokens — Miop.IA (extraídos de `design.png` por amostragem de pixel)

> Valores extraídos programaticamente (PIL, amostragem do pixel mais saturado em
> cada região, pra evitar diluição por anti-aliasing). Confiança alta nas cores;
> os espaçamentos são estimados visualmente a partir da grade de coordenadas e
> devem ser tratados como ponto de partida, não valor absoluto de Figma.

## Cores

```css
:root {
  /* Fundo */
  --cor-fundo-pagina: #E5E5E5;      /* fundo fora do card (preview/figma) */
  --cor-fundo-card: #FFFFFF;         /* fundo do card/modal */
  --cor-fundo-secao: #F6F7F9;        /* fundo dos blocos ENTENDA e OBRIGADA */

  /* Texto */
  --cor-texto-titulo: #16171D;       /* headline, texto do pill cinza */
  --cor-texto-corpo: #6B6F80;        /* parágrafos, labels uppercase (ENTENDA, PENSE etc.) */

  /* Marca */
  --cor-logo-laranja: #F2600A;       /* "Miop" — amostra bruta deu #FF5000, suavizado p/ uso em texto grande */
  --cor-logo-azul: #012252;          /* ".IA" */

  /* Destaque / sinal de alerta no texto */
  --cor-destaque: #FF0047;           /* palavras marcadas no parágrafo + pill vermelho (borda e texto) */

  /* Bloco PENSE */
  --cor-bullet-pense: #4E00E8;       /* quadrado/marcador roxo-azulado */

  /* Avaliação do usuário */
  --cor-verdadeira: #00855E;         /* verde — borda/texto no estado não selecionado */
  --cor-verdadeira-fundo-selecionado: #00855E; /* mesmo tom, preenchido no estado selecionado */
  --cor-duvidosa: #BC7B1B;           /* âmbar/dourado */
  --cor-falsa: #D20029;              /* vermelho */

  /* Bordas e divisores */
  --cor-borda-caixa: #E3E5EA;        /* borda da caixa "Antes de compartilhar" */
  --cor-borda-pill-cinza: #24252B;   /* borda do pill cinza (mais escura do que parece à primeira vista) */

  /* Badge do site de origem */
  --cor-badge-fundo: #D5D7DC;
}
```

## Tipografia

- Família: sans-serif geométrica/grotesca (ex.: `Inter, system-ui, sans-serif` —
  assumido pela aparência; confirmar contra o arquivo Figma real se possível).
- Labels uppercase (ENTENDA, O QUE CADA DESTAQUE REVELA, PENSE, ANTES DE
  COMPARTILHAR, SUA AVALIAÇÃO, OBRIGADA POR AVALIAR): caixa alta, letter-spacing
  aumentado (~0.05em), peso 600, tamanho pequeno (~11-12px).
- Headline ("Estudo aponta relação..."): peso 700 (bold), ~18-20px.
- Corpo de parágrafo: peso 400, ~14-15px, line-height confortável (~1.6).
- Botões de avaliação: peso 600, ~14px.

## Formas e espaçamento (estimados)

- Raio de borda do card/modal: ~16-20px.
- Raio de borda das caixas internas (ENTENDA, ANTES DE COMPARTILHAR, OBRIGADA):
  ~12px.
- Botões de avaliação e pills: totalmente arredondados (pill shape,
  `border-radius: 9999px`).
- Padding interno do card: ~24px.
- Largura do modal: as duas telas no Figma aparentam larguras ligeiramente
  diferentes (~570px vs ~600px) — recomenda-se fixar **uma única largura** na
  implementação (sugestão: 420-480px, compatível com um painel/modal injetado
  na página — essa largura é grande demais para o popup nativo padrão de
  extensão, então confirmar que o design pressupõe uma modal injetada via
  content script, não o popup de ação da extensão).

## Mapeamento cor → significado (útil para o agente entender a lógica, não só copiar valor)

| Cor | Onde aparece | Significado no produto |
|---|---|---|
| Vermelho/rosa (`--cor-destaque`) | Palavras destacadas no texto + pill da esquerda | Trecho específico que acionou um sinal de alerta |
| Cinza escuro (pill da direita) | Pill explicativo ao lado do destaque | Categoria/tipo do problema (ex.: "Número sem fonte") — neutro, não é alarme |
| Roxo/azul (`--cor-bullet-pense`) | Marcadores da lista "PENSE" | Convite à reflexão, não acusação |
| Verde / âmbar / vermelho (botões) | Botões de avaliação | Escala de confiança do próprio usuário na notícia — NÃO é o resultado do modelo |
