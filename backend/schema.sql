-- Extensão necessária para geração e validação de UUIDs
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS modelos (
    id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(50) NOT NULL,
    pipeline_version VARCHAR(50) NOT NULL,
    dataset_version VARCHAR(100) NOT NULL,
    f1 REAL NOT NULL,
    threshold REAL NOT NULL,
    criado_em TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_model_pipeline UNIQUE (model_version, pipeline_version)
);

-- Tabela principal: Notícias analisadas pela IA
CREATE TABLE IF NOT EXISTS noticias (
    id BIGSERIAL PRIMARY KEY,
    url VARCHAR(2048),
    hash_texto CHAR(64) NOT NULL,
    texto TEXT NOT NULL,
    texto_truncado TEXT,
    modelo_id BIGINT NOT NULL REFERENCES modelos(id) ON DELETE RESTRICT,
    prob_suspeita REAL NOT NULL,
    faixa VARCHAR(32) NOT NULL,

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

    criada_em TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_hash_modelo UNIQUE (hash_texto, modelo_id)
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