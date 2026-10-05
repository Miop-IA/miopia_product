-- Extensão necessária para geração e validação de UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Tabela principal: Notícias analisadas pela IA
CREATE TABLE IF NOT EXISTS noticias (
    id BIGSERIAL PRIMARY KEY,
    url VARCHAR(2048),
    hash_texto CHAR(64) NOT NULL UNIQUE,
    texto TEXT NOT NULL,
    prob_suspeita REAL,
    faixa VARCHAR(32),
    modelo_f1 REAL DEFAULT 0.961,

    -- Indicadores estilométricos truncados (dataset 11)
    trunc_pausality REAL,
    trunc_emotiveness REAL,
    trunc_diversity REAL,
    trunc_upper_case_density REAL,
    trunc_verb_density REAL,
    trunc_noun_density REAL,
    trunc_adj_density REAL,
    trunc_adv_density REAL,
    trunc_pron_density REAL,
    link_density REAL,

    -- Indicadores recalculados da master (spaCy + SpellChecker)
    rc_spelling_errors REAL,
    rc_modal_verbs_density REAL,
    rc_subj_imp_verbs_density REAL,
    rc_pron_1_2_sing_density REAL,
    rc_pron_1_plur_density REAL,

    criada_em TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Tabela de feedback dos utilizadores: Avaliações / Palpites (0 = Verdadeiro, 1 = Duvidoso, 2 = Falso)
CREATE TABLE IF NOT EXISTS avaliacoes (
    id BIGSERIAL PRIMARY KEY,
    noticia_id BIGINT NOT NULL REFERENCES noticias(id) ON DELETE CASCADE,
    client_id UUID NOT NULL,
    avaliacao SMALLINT NOT NULL CHECK (avaliacao IN (0, 1, 2)),
    criada_em TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_noticia_cliente UNIQUE (noticia_id, client_id)
);

-- Índices de consulta e desempenho
CREATE INDEX IF NOT EXISTS idx_noticias_hash ON noticias(hash_texto);
CREATE INDEX IF NOT EXISTS idx_avaliacoes_client ON avaliacoes(client_id);