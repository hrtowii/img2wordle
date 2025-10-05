#!/usr/bin/env python3
"""
wordle_art_export.py
Usage: python3 wordle_art_export.py input.png words.txt n

Produces:
 - results.json   (structured data of solved blocks, target, guesses, patterns)
 - wordle_art.html (self-contained viewer that draws the Wordle tiles)
"""

import sys
import json
from pathlib import Path

import numpy as np
from PIL import Image


# -----------------------------
# WORDLE LOGIC (binary: 1=green, 0=black)
# -----------------------------
def feedback(guess: str, target: str):
    """Return binary pattern list: 1 if same letter at same pos, else 0."""
    return [1 if g == t else 0 for g, t in zip(guess, target)]


def match_pattern(word: str, target: str, pattern):
    return feedback(word, target) == pattern


def solve_block(block, wordlist, used_words):
    """
    block: np.array shape (5,5) entries 0/1
    wordlist: list of 5-letter uppercase words
    used_words: set of words already used anywhere
    Return (target, guesses_list, patterns) or (None, None, None)
    - guesses_list is a list of 5 guess words (top->bottom)
    - patterns is list of 5 lists corresponding to rows (same as block rows)
    """
    # iterate targets in deterministic order but skip used ones
    for target in wordlist:
        if target in used_words:
            continue
        guesses = []
        valid = True
        for row in block:
            pat = row.tolist()
            # candidates must be unused and match pattern for this target
            candidates = [
                w
                for w in wordlist
                if (w not in used_words) and match_pattern(w, target, pat)
            ]
            if not candidates:
                valid = False
                break
            # choose candidate deterministically - pick the first
            guesses.append(candidates[0])
        if valid:
            # mark used (target + guesses)
            used_words.add(target)
            used_words.update(guesses)
            return target, guesses, [row.tolist() for row in block]
    return None, None, None


# -----------------------------
# IMAGE -> BLOCKS
# -----------------------------
def image_to_blocks(img_path, n):
    """
    Resize the image to exactly (5n, 5n), threshold to binary, and
    return list-of-list of 5x5 np arrays: blocks[row][col]
    """
    img = Image.open(img_path).convert("L")  # grayscale
    img = img.resize((5 * n, 5 * n), Image.NEAREST)
    arr = np.array(img)
    thresh = arr.mean()  # global threshold; you can customize
    binary = (arr < thresh).astype(
        int
    )  # darker pixels -> 1 (green), lighter -> 0 (black)

    blocks = []
    for by in range(n):
        row_blocks = []
        for bx in range(n):
            block = binary[by * 5 : (by + 1) * 5, bx * 5 : (bx + 1) * 5]
            row_blocks.append(block)
        blocks.append(row_blocks)
    return blocks


HTML_TEMPLATE = """<!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Wordle Art Viewer</title>
    <style>
      body {{ font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; padding: 18px; background:#111; color:#eee; }}
      .grid {{ display: grid; grid-gap: 18px; }}
      .block {{ display: inline-grid; grid-template-columns: repeat(5, 36px); grid-auto-rows: 36px; gap:6px; padding:8px; background:#222; border-radius:8px; }}
      .tile {{ width:36px; height:36px; display:flex; align-items:center; justify-content:center; font-weight:700; border-radius:6px; color:#fff; font-size:14px; }}
      .tile.black {{ background:#222; border: 2px solid #333; color:#888; }}
      .tile.green {{ background:#6aaa64; border: 2px solid #4c8f4a; }}
      .block-label {{ margin-top:6px; font-size:12px; color:#bbb; text-align:center; }}
      .container {{ display:flex; gap:18px; flex-direction:column; align-items:flex-start;}}
      .row {{ display:flex; gap:8px; align-items:flex-start; }}
      .meta {{ margin-bottom:8px; color:#bbb; font-size:14px; }}
    </style>
    </head>
    <body>
    <div class="container">
      <div class="meta">Wordle Art Viewer — grid: <strong>{n}×{n}</strong> blocks — block size: 5×5 — wordlist size: {wordlist_count}</div>
      <div id="canvas" class="grid" style="grid-template-columns: repeat({n}, auto);"></div>
      <details style="margin-top:12px; color:#ccc;">
        <summary>Raw results JSON</summary>
        <pre id="raw" style="white-space:pre-wrap;"></pre>
      </details>
    </div>

    <script>
    window.WORDLE_ART = {json_data};

    function draw() {{
      const data = window.WORDLE_ART;
      const canvas = document.getElementById('canvas');
      canvas.innerHTML = '';

      data.blocks.forEach(row => {{
        row.forEach(block => {{
          const blockEl = document.createElement('div');
          blockEl.className = 'block';
          blockEl.style.width = 'auto';
          const guesses = block.guesses;
          const patterns = block.patterns;
          for (let r = 0; r < 5; r++) {{
            if (!guesses) {{
              for (let c = 0; c < 5; c++) {{
                const t = document.createElement('div');
                t.className = 'tile black';
                t.textContent = (patterns ? (patterns[r][c] ? '•' : '') : '');
                blockEl.appendChild(t);
              }}
            }} else {{
              const word = guesses[r];
              const pat = patterns[r];
              for (let c = 0; c < 5; c++) {{
                const t = document.createElement('div');
                t.className = 'tile ' + (pat[c] ? 'green' : 'black');
                t.textContent = word[c];
                blockEl.appendChild(t);
              }}
            }}
          }}
          const label = document.createElement('div');
          label.className = 'block-label';
          label.textContent = block.target ? ('T:' + block.target) : 'UNSOLVED';
          const wrapper = document.createElement('div');
          wrapper.appendChild(blockEl);
          wrapper.appendChild(label);
          canvas.appendChild(wrapper);
        }});
      }});

      document.getElementById('raw').textContent = JSON.stringify(data, null, 2);
    }}

    draw();
    </script>
    </body>
    </html>
    """


def export_results_json(outpath: Path, metadata, blocks_result):
    data = {"meta": metadata, "blocks": blocks_result}
    outpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def export_html(outpath: Path, data, n, wordlist_count):
    html = HTML_TEMPLATE.format(
        json_data=json.dumps(data), n=n, wordlist_count=wordlist_count
    )
    outpath.write_text(html, encoding="utf-8")


# -----------------------------
# MAIN
# -----------------------------
def main(img_path, wordlist_path, n):
    # load wordlist (uppercase)
    with open(wordlist_path, encoding="utf-8") as f:
        wordlist = [w.strip().upper() for w in f if len(w.strip()) == 5]

    if not wordlist:
        print("Wordlist empty or no 5-letter words found.")
        return

    blocks = image_to_blocks(img_path, n)
    used_words = set()
    blocks_result = []  # will be list of rows; each row is list of block dicts

    total_blocks = n * n
    solved_count = 0

    for brow in blocks:
        row_result = []
        for block in brow:
            target, guesses, patterns = solve_block(block, wordlist, used_words)
            if target is None:
                # unsolved: we still include the pattern so HTML shows it
                row_result.append(
                    {
                        "target": None,
                        "guesses": None,
                        "patterns": patterns or [r.tolist() for r in block],
                    }
                )
            else:
                row_result.append(
                    {
                        "target": target,
                        "guesses": guesses,
                        "patterns": patterns,
                    }
                )
                solved_count += 1
        blocks_result.append(row_result)

    meta = {
        "image": str(img_path),
        "n": n,
        "total_blocks": total_blocks,
        "solved_blocks": solved_count,
        "wordlist_count": len(wordlist),
    }

    # write outputs
    json_path = Path("results.json")
    html_path = Path("wordle_art.html")
    data = export_results_json(json_path, meta, blocks_result)
    export_html(html_path, data, n, len(wordlist))
    print(f"Wrote {json_path} and {html_path}.")
    print(
        f"Solved {solved_count}/{total_blocks} blocks. Uniqueness enforced across the whole canvas."
    )


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 wordle_art_export.py input.png words.txt n")
        sys.exit(1)
    img_path = sys.argv[1]
    wordlist_path = sys.argv[2]
    n = int(sys.argv[3])
    main(img_path, wordlist_path, n)
