import re
import sys
from pathlib import Path

def audit_paper():
    paper_dir = Path(r"c:\Users\Dell\Documents\AI_projects\projects\Rana\paper")
    tex_path = paper_dir / "main.tex"
    bib_path = paper_dir / "references.bib"

    print("=" * 60)
    print("Q1 ACADEMIC PAPER & HUMANIZER AUDIT REPORT")
    print("=" * 60)

    if not tex_path.exists():
        print(f"Error: {tex_path} not found.")
        sys.exit(1)
    if not bib_path.exists():
        print(f"Error: {bib_path} not found.")
        sys.exit(1)

    tex_content = tex_path.read_text(encoding="utf-8")
    bib_content = bib_path.read_text(encoding="utf-8")

    # 1. Check for Em dashes (—) and En dashes (–) in prose
    print("\n[CHECK 1] Scanning for Em dashes (—) and En dashes (–)...")
    # Exclude BibTeX and pure math/TikZ lines
    em_dashes = [m.start() for m in re.finditer(r"—", tex_content)]
    en_dashes = [m.start() for m in re.finditer(r"–", tex_content)]
    double_hyphens_outside_bib = []
    
    if em_dashes:
        print(f"FAILED: Found {len(em_dashes)} em-dashes (—) in main.tex!")
    else:
        print("PASSED: Zero em-dashes (—) found in main.tex.")

    if en_dashes:
        print(f"FAILED: Found {len(en_dashes)} en-dashes (–) in main.tex!")
    else:
        print("PASSED: Zero en-dashes (–) found in main.tex.")

    # 2. Check for banned AI vocabulary words
    print("\n[CHECK 2] Scanning for banned AI vocabulary words...")
    banned_words = [
        "delve", "delves", "tapestry", "landscape", "pivotal", "crucial",
        "vital", "testament", "underscore", "underscores", "showcase", "showcases",
        "intricate", "intricacies", "multifaceted", "holistic", "foster", "fostering",
        "garner", "garnering", "cutting-edge", "vibrant", "revolutionize", "seamlessly"
    ]
    
    # Strip comments from tex before checking
    clean_tex = re.sub(r"%.*$", "", tex_content, flags=re.MULTILINE)
    
    found_banned = {}
    for word in banned_words:
        matches = list(re.finditer(rf"\b{word}\b", clean_tex, flags=re.IGNORECASE))
        if matches:
            found_banned[word] = len(matches)

    if found_banned:
        print(f"FAILED: Found banned AI words: {found_banned}")
    else:
        print("PASSED: Zero banned AI vocabulary words detected.")

    # 3. Check citations consistency
    print("\n[CHECK 3] Cross-referencing Citations between main.tex and references.bib...")
    # Extract citations from tex
    cited_keys = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", clean_tex):
        keys = [k.strip() for k in m.group(1).split(",")]
        cited_keys.update(keys)

    # Extract bib keys
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib_content))

    missing_in_bib = cited_keys - bib_keys
    uncited_in_tex = bib_keys - cited_keys

    print(f"Total citations in text: {len(cited_keys)}")
    print(f"Total entries in BibTeX: {len(bib_keys)}")

    if missing_in_bib:
        print(f"FAILED: Citations in text missing from BibTeX: {missing_in_bib}")
    else:
        print("PASSED: All citations in main.tex exist in references.bib.")

    if uncited_in_tex:
        print(f"WARNING: BibTeX entries not cited in main.tex: {uncited_in_tex}")
    else:
        print("PASSED: All BibTeX entries are actively cited in main.tex.")

    # 4. Check LaTeX environments matching
    print("\n[CHECK 4] Checking LaTeX environment balance (\begin{...} / \end{...})...")
    begins = re.findall(r"\\begin\{([^}]+)\}", clean_tex)
    ends = re.findall(r"\\end\{([^}]+)\}", clean_tex)
    
    begin_counts = {}
    end_counts = {}
    for b in begins: begin_counts[b] = begin_counts.get(b, 0) + 1
    for e in ends: end_counts[e] = end_counts.get(e, 0) + 1

    env_mismatch = False
    all_envs = set(begin_counts.keys()).union(set(end_counts.keys()))
    for env in all_envs:
        b_cnt = begin_counts.get(env, 0)
        e_cnt = end_counts.get(env, 0)
        if b_cnt != e_cnt:
            print(f"FAILED: Environment '{env}' mismatch: {b_cnt} \\begin vs {e_cnt} \\end")
            env_mismatch = True

    if not env_mismatch:
        print("PASSED: All LaTeX environments are perfectly balanced.")

    # 5. Check Labels and References
    print("\n[CHECK 5] Checking Internal Cross-References (\\ref vs \\label)...")
    labels = set(re.findall(r"\\label\{([^}]+)\}", clean_tex))
    refs = set(re.findall(r"\\ref\{([^}]+)\}", clean_tex))
    missing_labels = refs - labels

    print(f"Total labels defined: {len(labels)}")
    print(f"Total references called: {len(refs)}")
    if missing_labels:
        print(f"FAILED: Missing labels for references: {missing_labels}")
    else:
        print("PASSED: All \\ref calls have matching \\label definitions.")

    # 6. Word count and Page Budget Estimation
    print("\n[CHECK 6] Estimating Word Count and Page Budget...")
    words = re.findall(r"\b[A-Za-z0-9_-]+\b", clean_tex)
    total_words = len(words)
    # In IEEEtran 2-column format, each page holds ~850-950 words of body text,
    # or ~500-600 words with equations, algorithms, tables, and figures.
    est_pages = round(total_words / 600, 1)
    print(f"Total word count: {total_words} words")
    print(f"Estimated length in IEEEtran format (with figures, tables, math): ~{est_pages} pages (Budget: <= 12 pages)")
    if est_pages <= 12:
        print("PASSED: Page budget within IEEE limits (<= 12 pages).")
    else:
        print("WARNING: May exceed 12 pages.")

    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    audit_paper()
