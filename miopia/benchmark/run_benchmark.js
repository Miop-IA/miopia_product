const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const FIXTURES_DIR = path.join(__dirname, 'fixtures');
const SCRIPTS_DIR = path.join(__dirname, '../scripts');

// Carrega os scripts originais da extensão
const readabilityScript = fs.readFileSync(path.join(SCRIPTS_DIR, 'readability.js'), 'utf8');
const contentScript = fs.readFileSync(path.join(SCRIPTS_DIR, 'content.js'), 'utf8');

function runExtractionOnHtml(htmlContent, url) {
    // Configura o JSDOM para simular o navegador
    const dom = new JSDOM(htmlContent, {
        url: url,
        runScripts: "dangerously" // Permite execução de scripts dentro do DOM
    });

    // Injeta os scripts no contexto do JSDOM
    const window = dom.window;
    const document = window.document;

    // Simula a API do Chrome
    const mockChromeScript = document.createElement('script');
    mockChromeScript.textContent = `
        window.chrome = {
            runtime: {
                onMessage: {
                    addListener: function() {}
                }
            }
        };
    `;
    document.head.appendChild(mockChromeScript);

    // Executa os scripts no window
    const scriptEl1 = document.createElement('script');
    scriptEl1.textContent = readabilityScript;
    document.head.appendChild(scriptEl1);

    const scriptEl2 = document.createElement('script');
    scriptEl2.textContent = contentScript;
    document.head.appendChild(scriptEl2);

    // Chama a função principal de extração mapeada pelo content.js
    let result = null;
    try {
        result = window.extractNewsContent();
    } catch (e) {
        return { error: e.message };
    }

    return result;
}

function runBenchmark() {
    console.log("==========================================");
    console.log("   Miop.IA - Benchmark de Extração        ");
    console.log("==========================================\n");

    if (!fs.existsSync(FIXTURES_DIR)) {
        fs.mkdirSync(FIXTURES_DIR);
        console.log("📁 Diretório 'fixtures' criado. Adicione seus arquivos de teste (.html e .json).");
        return;
    }

    const files = fs.readdirSync(FIXTURES_DIR);
    const htmlFiles = files.filter(f => f.endsWith('.html'));

    if (htmlFiles.length === 0) {
        console.log("⚠️  Nenhum arquivo .html encontrado em 'benchmark/fixtures/'.");
        console.log("   Para rodar o benchmark, adicione pares de arquivos como:");
        console.log("   - g1_noticia.html");
        console.log("   - g1_noticia.json (contendo os resultados esperados)\n");
        return;
    }

    let totalTests = 0;
    let passedTests = 0;

    htmlFiles.forEach(htmlFile => {
        const baseName = htmlFile.replace('.html', '');
        const jsonFile = `${baseName}.json`;
        
        console.log(`\n▶ Testando portal/fixture: ${baseName}`);
        
        const htmlPath = path.join(FIXTURES_DIR, htmlFile);
        const jsonPath = path.join(FIXTURES_DIR, jsonFile);

        const htmlContent = fs.readFileSync(htmlPath, 'utf8');
        let expected = {};
        
        if (fs.existsSync(jsonPath)) {
            expected = JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
        } else {
            console.warn(`  [Aviso] Arquivo ${jsonFile} não encontrado. Usando teste modo exploratório.`);
        }

        const url = expected.url || "https://example.com/noticia";
        
        const startTime = performance.now();
        const result = runExtractionOnHtml(htmlContent, url);
        const endTime = performance.now();

        if (result.error) {
            console.error(`  ❌ Falha na execução: ${result.error}`);
            totalTests++;
            return;
        }

        console.log(`  - Método de extração: ${result.method}`);
        console.log(`  - Palavras extraídas: ${result.words}`);
        console.log(`  - Tempo de execução: ${(endTime - startTime).toFixed(2)} ms`);

        // Validação de métricas
        totalTests++;
        let isPass = true;

        if (expected.words) {
            const diff = Math.abs(result.words - expected.words);
            const tolerance = expected.tolerance || 5; // Tolerância de divergência em palavras
            if (diff <= tolerance) {
                console.log(`  ✅ Palavras extraídas dentro da tolerância (${result.words} vs ${expected.words})`);
            } else {
                console.log(`  ❌ Falso Positivo/Negativo de palavras: Esperado ${expected.words}, Obtido ${result.words}`);
                isPass = false;
            }
        }

        if (expected.expected_method && result.method && !result.method.includes(expected.expected_method)) {
            console.log(`  ❌ Método incorreto: Esperado '${expected.expected_method}', Obtido '${result.method}'`);
            isPass = false;
        }

        if (expected.must_include && expected.must_include.length > 0) {
            expected.must_include.forEach(phrase => {
                if (!result.text.includes(phrase)) {
                    console.log(`  ❌ Falso Negativo (Corte indevido): Trecho ausente -> "${phrase}"`);
                    isPass = false;
                }
            });
        }

        if (expected.must_not_include && expected.must_not_include.length > 0) {
            expected.must_not_include.forEach(phrase => {
                if (result.text.includes(phrase)) {
                    console.log(`  ❌ Falso Positivo (Ruído vazado): Trecho presente -> "${phrase}"`);
                    isPass = false;
                }
            });
        }

        if (isPass) {
            passedTests++;
            console.log(`  🎉 Teste passou.`);
        }
    });

    console.log(`\n==========================================`);
    console.log(`Resumo: ${passedTests}/${totalTests} testes bem sucedidos.`);
    if (passedTests !== totalTests) {
        process.exit(1);
    }
}

runBenchmark();
