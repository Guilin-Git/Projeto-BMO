"""Audit script to analyze Gold data quality gaps."""
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

GOLD = r"C:\Users\PC\Projeto-BMO\bmo_brain\knowledge\gold"

for cat in ["episodes", "characters", "places", "items"]:
    cat_dir = os.path.join(GOLD, cat)
    total = 0
    sem_descricao = 0
    muito_curtos = 0
    incompletos = 0
    tamanhos = []
    secoes = {}

    for f in sorted(os.listdir(cat_dir)):
        if not f.endswith(".md"):
            continue
        total += 1

        with open(os.path.join(cat_dir, f), "r", encoding="utf-8") as fh:
            content = fh.read()

        tamanhos.append(len(content))

        if "incompleto: true" in content[:500].lower():
            incompletos += 1

        # Separa frontmatter do corpo
        parts = content.split("---", 2)
        corpo = parts[2].strip() if len(parts) >= 3 else content

        # Corpo muito curto
        texto_real = re.sub(r"[#>\n\r\s*_\-]", "", corpo)
        if len(texto_real) < 100:
            muito_curtos += 1

        # Conta seções presentes
        secs = re.findall(r"^## (.+)$", corpo, re.MULTILINE)
        for s in secs:
            s = s.strip()
            secoes[s] = secoes.get(s, 0) + 1

        # Verifica se tem descricao com conteudo
        has_content = False
        if "## Descrição" in corpo:
            match = re.search(r"## Descrição\s*\n(.*?)(?=\n## |\Z)", corpo, re.DOTALL)
            if match and len(match.group(1).strip()) > 20:
                has_content = True
        if "## Sinopse" in corpo or "## Enredo" in corpo:
            has_content = True
        if not has_content and len(texto_real) > 200:
            has_content = True

        if not has_content:
            sem_descricao += 1

    print(f"\n=== {cat.upper()} ({total} arquivos) ===")
    print(f"  Incompletos:           {incompletos}")
    print(f"  Sem conteúdo útil:     {sem_descricao}")
    print(f"  Muito curtos (<100ch): {muito_curtos}")
    if tamanhos:
        print(f"  Tamanho: min={min(tamanhos)}B  max={max(tamanhos)}B  media={sum(tamanhos)//len(tamanhos)}B")
    print(f"  Seções encontradas:")
    for sec, cnt in sorted(secoes.items(), key=lambda x: -x[1]):
        pct = cnt * 100 // total
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"    {sec:30s} {cnt:3d}/{total} ({pct:2d}%) {bar}")
