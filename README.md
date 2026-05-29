# Laboratório 10 — O Pipeline Definitivo
## RAG + QLoRA + FlashAttention-2 + KV Cache

## Estrutura do Repositório

```
lab10/
├── lab10_pipeline_definitivo.ipynb   # Notebook principal com todo o código
├── benchmark_lab10.png               # Gráfico comparativo gerado automaticamente
└── README.md                         # Este arquivo
```

---

## Como Executar

1. **Ambiente recomendado:** Google Colab com GPU T4 (gratuito) ou A100 (Colab Pro)
   - `Runtime → Change runtime type → T4 GPU`

2. **Instalar dependências** (célula 0 do notebook):
   ```bash
   pip install transformers accelerate bitsandbytes
   pip install flash-attn --no-build-isolation
   ```

3. Executar as células em ordem do Passo 1 ao Passo 4.

---

## Métricas de Benchmark

> ⚠️ Os valores abaixo são referência para GPU T4 (16 GB VRAM).
> Substitua pelos valores reais medidos ao executar o notebook.

| Configuração | VRAM ao Carregar | Tempo (100 tokens) | Pico VRAM Geração |
|---|---|---|---|
| **Passo 1** – Modelo 4-bits QLoRA | ~780 MB | — | — |
| **Passo 3** – Sem KV Cache | — | ~38 s | ~4.200 MB |
| **Passo 4** – KV Cache + FlashAttention-2 | — | ~8 s | ~1.900 MB |

**Speedup de tempo:** ~4,7× mais rápido  
**Redução de VRAM:** ~55% menos memória no pico de geração

---

## Passo 5 — Parecer Técnico Arquitetural

### Parte A — Como QLoRA + KV Cache + FlashAttention "salvaram" o Transformer

O primeiro problema de um sistema RAG com contextos massivos é simplesmente *caber na GPU*. Carregar um modelo Llama em Float16 pleno consumiria entre 2–7 GB de VRAM apenas com os pesos, antes de processar qualquer token. A quantização QLoRA em 4-bits com `bitsandbytes` (NF4 + double quantization) resolve isso ao comprimir cada parâmetro para 4 bits enquanto mantém a precisão de cômputo em Float16, reduzindo o footprint do modelo em aproximadamente 4×. Com o modelo caindo de ~2 GB para ~780 MB, sobra VRAM para o contexto e para a geração.

O segundo problema — e o que causou o travamento em produção — é a complexidade **O(n²)** do Self-Attention durante a decodificação autorregressiva. Sem KV Cache, a cada novo token gerado o Transformer precisa recalcular as matrizes de chave (K) e valor (V) para **todos os tokens anteriores** desde o início do contexto, repetindo bilhões de operações já realizadas. Com `use_cache=True`, essas matrizes são armazenadas em memória após o primeiro cálculo (prefill), e cada passo de geração subsequente realiza apenas O(1) novas operações de atenção — o novo token consulta o cache, sem reprocessar o contexto inteiro. Já o **FlashAttention-2** ataca o gargalo de largura de banda de memória: o kernel padrão de atenção materializa as matrizes N×N de scores na DRAM (HBM) da GPU, gerando tráfego massivo de dados. O FlashAttention reescreve o algoritmo para operar em blocos diretamente na SRAM on-chip (muito mais rápida), reduzindo as transferências HBM em ~5–10× e eliminando a necessidade de armazenar a matriz de atenção completa. A combinação das três técnicas — QLoRA (modelo cabe na GPU), KV Cache (decodificação O(n) no número de novos tokens), FlashAttention-2 (prefill IO-Aware) — é o que torna viável processar 10.000–15.000 tokens em GPUs de consumo sem OOM.

---

### Parte B — Por que 2 milhões de tokens quebrariam até o FlashAttention, e por que a indústria precisa do Mamba

O FlashAttention resolve o gargalo de *bandwidth* de memória ao operar em blocos na SRAM, mas não elimina a **complexidade quadrática de cômputo** O(n²) do mecanismo de atenção. Para 2 milhões de tokens, o número de operações de produto interno cresce para 4×10¹² por camada, tornando o custo computacional proibitivo mesmo com hardware de ponta. Além disso, o KV Cache, que economiza reprocessamento durante a decodificação, passa a ser o novo vilão de memória: armazenar os pares K e V para 2M de tokens em todos os heads e camadas exigiria centenas de gigabytes de VRAM — inviável em qualquer GPU individual. O problema não é mais otimização de acesso à memória, mas a escala intrínseca da arquitetura Transformer.

É exatamente por isso que arquiteturas baseadas em **State Space Models (SSM)**, como o **Mamba**, emergiram como alternativa estrutural. O Mamba substitui o Self-Attention por uma recorrência aprendida que comprime todo o contexto histórico em um **estado latente de dimensão fixa** (o estado oculto da SSM). Isso resulta em complexidade de memória **O(1)** em relação ao comprimento da sequência durante a inferência — independentemente de processar 15.000 ou 2 milhões de tokens, o footprint de memória do estado permanece constante. O custo de treinamento é linear O(n) ao invés de quadrático O(n²), e a inferência autorregressiva é intrinsecamente eficiente porque não há cache crescente. A desvantagem é que o estado comprimido pode perder informação de longo alcance com alta precisão que a atenção captura naturalmente, sendo este um campo de pesquisa ativo com arquiteturas híbridas (como Jamba, que intercala blocos Mamba e Transformer) buscando o melhor dos dois mundos para contextos extremamente longos.

---

## Referências

- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314
- Dao et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022
- Dao (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. arXiv:2307.08691
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752
- Hugging Face. *BitsAndBytes Integration for 4-bit Quantization*. https://huggingface.co/docs/transformers/quantization
