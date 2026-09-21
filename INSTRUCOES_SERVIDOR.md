# Instruções para o servidor — reproduzir o recálculo de FID/KID do paper

> **Para o Claude que for executar isto:** siga os passos na ordem. Não altere o paper
> (`paper_tex/sn-article.tex`) nem a carta (`paper_tex/rev2.tex`) por conta própria. Se algum número não bater,
> pare e reporte as diferenças ao usuário (passo 7).

## Contexto

Os números atuais do paper (Tabela 2, Tabela 3, Fig. 3 e o texto da Seção 4) vieram de um recálculo feito em
17/09/2026 num Mac, a partir das imagens publicadas no dataset do Hugging Face
`FelipeMahlow/auto-tune-generated-images`. Esse recálculo mudou três coisas em relação à rodada original no H200
(`results/dreambooth`, `results/textual_inversion`):

1. **IC95% de FID/KID:** o bootstrap agora mantém fixas as 20 imagens reais de cada conceito e reamostra só as
   geradas. O bootstrap antigo reamostrava também as reais, o que inflava FID/KID; vários intervalos nem continham o
   valor pontual (ex.: Paçoca FID 227,8 [232,7; 260,9]).
2. **Fig. 3:** FID e KID calculados em 100 imagens em **todos** os pontos, incluindo os passos 0 e 3000 (média de
   50 subamostras de 100 das 1000 imagens), porque FID/KID dependem do tamanho da amostra.
3. **Textual Inversion:** CLIP, FID e KID calculados só nas imagens que passaram no filtro de segurança, que são as
   publicadas no HF (entre 511 e 987 por conceito).

Na Tabela 2 (DreamBooth), o **CLIP vem da rodada original** (`results/dreambooth/metrics_dreambooth.csv`);
só FID/KID foram recalculados.

Os scripts daquele recálculo se perderam com a pasta temporária da sessão. Eles foram reconstruídos da transcrição e
estão em `scripts/recompute/`. **O objetivo aqui é rodá-los de novo, conferir que reproduzem o paper e versionar o
resultado**, para que todo número do paper tenha fonte no repositório.

| Script | O que faz | Saída |
|---|---|---|
| `extract_features.py` | Baixa os tars do HF um a um, extrai features InceptionV3 (resize 256, uint8) e CLIP do TI, apaga o tar | `scripts/recompute/work/features/*.npy` |
| `compute_metrics.py` | FID/KID + IC95% (real fixo, 1000 reamostragens, seed 20260917) + FID/KID em 100 imagens | `results/recomputed/metrics_recomputed.csv` |
| `make_tables.py` | Imprime as linhas LaTeX das Tabelas 2 e 3 e os números por grupo | stdout |
| `plot_fig3.py` | Refaz a Fig. 3 | `paper_tex/fig_3_paper.pdf` (+ prévia PNG em `work/`) |

---

## Passo 0 — ATENÇÃO antes do `git pull`: as imagens reais serão apagadas

O commit `2ed8c24` tirou do git as pastas de imagens de treino (`chamanto/`, `chaneques/`, `cuscuz/`, `jian/`,
`lokum/`, `paçoca/`, `patuá/`, `saci/`) por direitos autorais. **Um `git pull` num clone que ainda as tem vai
apagá-las do disco.** Elas são necessárias como conjunto real de referência. Restaure-as depois do pull, fora do
repositório, a partir do último commit que as continha:

```bash
cd /workspace/Auto-Tune
git pull
mkdir -p /workspace/real_images
git archive beff35d chamanto chaneques cuscuz jian lokum paçoca patuá saci | tar -x -C /workspace/real_images
for d in /workspace/real_images/*/; do echo "$(basename "$d") $(ls "$d" | wc -l)"; done   # esperado: 20 em cada
export REAL_IMAGES_ROOT=/workspace/real_images
```

Não copie essas pastas de volta para dentro do repositório nem faça commit delas.

## Passo 1 — Ambiente

Pode reaproveitar o venv dos experimentos. Pacotes necessários:
```bash
pip install torch torchvision "torchmetrics[image]" torch-fidelity "transformers<5" huggingface_hub numpy scipy matplotlib pillow
python -c "import torch; print(torch.cuda.is_available())"   # deve ser True
```
`transformers>=5` quebra o `CLIPScore` do torchmetrics; `transformers` muito antiga com numpy recente dá
`AttributeError: module 'numpy' has no attribute 'typeDict'`. Use `transformers<5` recente.

## Passo 2 — Extrair features (download ~36 GB do HF)

```bash
cd /workspace/Auto-Tune
export RECOMPUTE_WORK=/workspace/recompute_work      # opcional; padrão: scripts/recompute/work (ignorado pelo git)
nohup python -u scripts/recompute/extract_features.py > /workspace/recompute_extract.log 2>&1 &
tail -f /workspace/recompute_extract.log
```
- São 16 tars (DreamBooth e TI de cada conceito). Cada tar é apagado depois de processado, então o disco de pico
  fica em ~4 GB mais as features.
- O script é retomável: pastas com `.npy` pronto são puladas.
- Ao final, `work/features/` deve ter as features reais (`real_<conceito>.npy`), as do DreamBooth (base, 3000 e
  checkpoints 500–2500), as do TI (`..._with_baseline.npy` e `..._with_baseline_clip.npy`) e, para o Saci, as
  imagens do modelo base geradas pelo TI (`..._without_finetuning_ti.npy`).

## Passo 3 — Calcular as métricas

```bash
python scripts/recompute/compute_metrics.py | tail -3     # deve terminar com "wrote .../metrics_recomputed.csv (65 rows)"
```

## Passo 4 — Conferir as Tabelas 2 e 3

```bash
python scripts/recompute/make_tables.py > /workspace/recompute_tables.txt
```
Compare **linha a linha** com o paper:
- Tabela 2 = bloco com `\label{tab:clip-fid-results}` em `paper_tex/sn-article.tex`
- Tabela 3 = bloco com `\label{tab:ti-comparison}`

Os valores esperados são os que estão hoje no paper, por exemplo:
- Tabela 2, Paçoca: `16.45 & 25.40 [25.04, 25.74] & 329.4 & 227.7 [223.7, 232.0] & 0.213 & 0.052 [0.048, 0.056]`
- Tabela 3, Chamanto: `30.36 & 20.35 & +10.01 & 204.2 & 311.8 & −107.6 & 0.047 & 0.118 & −0.071 & 960`

**Tolerância:** o recálculo original extraiu as features em MPS (Apple) e aqui será CUDA, então diferenças de
**±1 na última casa exibida** são aceitáveis. Qualquer diferença maior deve ser reportada.

## Passo 5 — Refazer e conferir a Fig. 3

```bash
python scripts/recompute/plot_fig3.py
```
Isso sobrescreve `paper_tex/fig_3_paper.pdf`. Abra a prévia (`$RECOMPUTE_WORK/fig_3_preview.png`) e compare com
a figura anterior (`git show HEAD:paper_tex/fig_3_paper.pdf > /tmp/fig3_old.pdf`). As curvas devem ser as mesmas.

Confira também, a partir do CSV, as afirmações do texto que dependem dele:

| Onde (`sn-article.tex`) | Afirmação |
|---|---|
| Seção de métricas (~l. 304) | FID em 100 imagens é maior que em 1000 por **1,7 a 5,0 pontos**, conforme o conceito |
| Após Tabela 2 (~l. 371) | IC com no máximo **±0,35** CLIP, **±6** FID, **±0,005** KID; nenhum IC pós-treino contém o valor pré-treino; KID menor após o treino em 7 de 8 conceitos (exceção: Saci, 0,145 vs 0,132) |
| Análise por checkpoint (~l. 393) | CLIP máximo em 3000 passos para 5 conceitos (Lokum, Paçoca, Chaneques, Jian, Chamanto) e em 2500 para 3 (diferença ≤ 1,1); FID mínimo em 3000 para 4 conceitos, com margens de 2,0 a 6,1; KID mínimo em 3000 só para Jian e Chamanto, em 2500 para Cuscuz, Paçoca e Patuá |
| Comparação com TI (~l. 400) | DB melhor em CLIP e FID nos 8 conceitos (média +9,0 CLIP, −115 FID); KID melhor em 6 de 8 (TI melhor em Saci 0,131 vs 0,145 e Lokum 0,056 vs 0,069); nenhum IC dos dois métodos se sobrepõe |
| Diretrizes (~l. 449) | KID em 3000 passos claramente pior que num checkpoint anterior para 3 conceitos |

Verificação extra: em todas as linhas de 1000 imagens do CSV, o IC de FID e de KID deve **conter o valor pontual**.

## Passo 6 — Versionar

Se tudo bateu (dentro da tolerância):
1. Acrescente ao `README.md`, na parte dos resultados, uma nota curta: os IC de FID/KID em `results/dreambooth` e
   `results/textual_inversion` vêm do bootstrap original, que reamostrava também as imagens reais; os números do
   paper vêm de `results/recomputed/metrics_recomputed.csv` (gerado por `scripts/recompute/`); o
   `experiment_utils.py` já foi corrigido para manter o conjunto real fixo.
2. Commit e push:
```bash
git add results/recomputed/metrics_recomputed.csv paper_tex/fig_3_paper.pdf README.md
git commit -m "Add recomputed FID/KID metrics that back the paper's Tables 2-3 and Figure 3"
git push origin main
```
Não faça commit de `scripts/recompute/work/` nem das imagens reais.

## Passo 7 — Se algo não bater

Não edite o paper. Reporte ao usuário:
- a lista de valores diferentes (conceito, métrica, valor no paper, valor recalculado);
- quais afirmações da tabela do passo 5 deixaram de valer;
- a sua avaliação da causa provável (ex.: diferença MPS × CUDA, imagem real diferente, tar diferente).

O usuário decide se o paper deve ser atualizado.

## Depois disto, ainda pendente (não faz parte destas instruções)

- Compilar o paper no Overleaf.
- As imagens raspadas continuam no **histórico** do GitHub (saíram só dos commits novos); o README do dataset no HF
  ainda diz que os dados de treino estão no GitHub; a coluna "Real" da Fig. 4 reproduz imagens de terceiros
  (permissão da Springer).
