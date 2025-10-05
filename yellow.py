#!/usr/bin/env python3
"""
wordle_art_export.py
Usage: python3 wordle_art_export.py input.png words.txt n [target_word]

Produces:
 - results.json   (structured data of solved blocks, target, guesses, patterns)
 - wordle_art.html (self-contained viewer that draws the Wordle tiles)
"""

import sys
import json
from pathlib import Path
import time
from collections import defaultdict
import itertools

import numpy as np
from PIL import Image, ImageEnhance
import random
from multiprocessing import Pool, cpu_count


# -----------------------------
# WORDLE LOGIC (0=black, 1=yellow, 2=green)
# -----------------------------
def feedback(guess: str, target: str):
    """Return Wordle feedback: 0=black, 1=yellow, 2=green."""
    guess = guess.upper()
    target = target.upper()
    pattern = [0] * 5
    target_counts = {}

    # First pass: mark greens and count non-green target letters
    for i in range(5):
        if guess[i] == target[i]:
            pattern[i] = 2
        else:
            target_counts[target[i]] = target_counts.get(target[i], 0) + 1

    # Second pass: mark yellows (respecting letter counts)
    for i in range(5):
        if (
            pattern[i] == 0
            and guess[i] in target_counts
            and target_counts[guess[i]] > 0
        ):
            pattern[i] = 1
            target_counts[guess[i]] -= 1

    return pattern


def match_pattern(word: str, target: str, pattern):
    """Return True if feedback(word, target) equals pattern (list of ints)."""
    return feedback(word, target) == pattern


# -----------------------------
# OPTIMIZED PATTERN MATCHING
# -----------------------------
def build_pattern_index(wordlist):
    """
    Build an index of words by their letter patterns.
    This is much faster than precomputing all word pairs.
    """
    print("Building pattern index...")
    start_time = time.time()

    # Group words by their letter patterns (e.g., "ABCDE", "AABCD", etc.)
    pattern_groups = defaultdict(list)

    for word in wordlist:
        # Create a pattern based on repeated letters
        # For example, "HELLO" -> "ABCBD"
        pattern = []
        letter_map = {}
        next_char = "A"

        for letter in word:
            if letter not in letter_map:
                letter_map[letter] = next_char
                next_char = chr(ord(next_char) + 1)
            pattern.append(letter_map[letter])

        pattern_str = "".join(pattern)
        pattern_groups[pattern_str].append(word)

    print(f"Pattern index built in {time.time() - start_time:.2f} seconds")
    return pattern_groups


def find_words_for_pattern(target, pattern, wordlist, pattern_groups):
    """
    Find words that match the given pattern against the target.
    Uses the pattern index to narrow down candidates.
    """
    # First, determine what letter pattern we need in the guess
    # based on the feedback pattern
    needed_positions = {}
    excluded_positions = set()
    required_letters = {}

    for i, p in enumerate(pattern):
        if p == 2:  # Green - letter must be in this position
            needed_positions[i] = target[i]
        elif p == 1:  # Yellow - letter must be in the word but not in this position
            excluded_positions.add(i)
            if target[i] not in required_letters:
                required_letters[target[i]] = 0
            required_letters[target[i]] += 1
        # Black (0) doesn't give us specific requirements beyond the above

    # Now find candidate words
    candidates = []

    # Get all words with the right letter pattern
    # This is a heuristic - it narrows down the search space
    for pattern_str, words in pattern_groups.items():
        # Skip patterns that don't match our requirements
        # This is a quick filter before doing the full feedback check

        # Check if this pattern could possibly match our requirements
        valid_pattern = True

        # Convert pattern_str back to a list for easier processing
        pattern_list = list(pattern_str)

        # Check green requirements
        for pos, letter in needed_positions.items():
            # All words in this group have the same letter pattern
            # So we can check if the pattern allows the right letter in this position
            if pattern_list[pos] != letter_map_for_word(target, pos):
                valid_pattern = False
                break

        if not valid_pattern:
            continue

        # Now check each word in this group
        for word in words:
            # Quick check: does the word have the right letters in the right positions?
            # This is faster than computing the full feedback
            valid = True

            # Check green requirements
            for pos, letter in needed_positions.items():
                if word[pos] != letter:
                    valid = False
                    break

            if not valid:
                continue

            # Check yellow requirements
            letter_count = {}
            for pos, letter in enumerate(word):
                if pos in excluded_positions and word[pos] == target[pos]:
                    valid = False
                    break
                letter_count[letter] = letter_count.get(letter, 0) + 1

            if not valid:
                continue

            # Check if we have the right number of each required letter
            for letter, count in required_letters.items():
                if letter_count.get(letter, 0) < count:
                    valid = False
                    break

            if not valid:
                continue

            # If we passed all the quick checks, do the full feedback check
            if feedback(word, target) == pattern:
                candidates.append(word)

    return candidates


def letter_map_for_word(word, pos):
    """Helper function to get the pattern character for a position in a word."""
    # This is a simplified version of what we did in build_pattern_index
    # It's not exact but serves as a heuristic
    return word[pos]


# -----------------------------
# SOLVER: find target + guesses for a 5x5 block of states 0/1/2
# -----------------------------
def solve_block_optimized(args):
    """
    Optimized version of solve_block for multiprocessing.
    args: (block, wordlist, pattern_groups, target_word, block_id)
    Returns: (block_id, target, guesses, patterns) or (block_id, None, None, None)
    """
    block, wordlist, pattern_groups, target_word, block_id = args

    # normalize block to integer lists
    block = np.array(block, dtype=int)

    # If a target word is specified, use it
    if target_word:
        target = target_word.upper()

        chosen = []
        patterns = []

        # For all 5 rows, find candidate guesses whose feedback equals the row pattern
        for r in range(5):
            desired_pat = block[r].tolist()

            # Find all words that match this pattern with the target
            candidates = find_words_for_pattern(
                target, desired_pat, wordlist, pattern_groups
            )

            if candidates:
                # Use a word that perfectly matches the desired pattern
                chosen.append(candidates[0])
                patterns.append(desired_pat)  # Use the desired pattern
            else:
                # If no perfect match, find the closest word and use its actual feedback
                best_word = None
                best_score = -1

                # Try to find a word that matches as many positions as possible
                for word in wordlist:
                    actual_pat = feedback(word, target)
                    score = sum(1 for i in range(5) if actual_pat[i] == desired_pat[i])
                    if score > best_score:
                        best_score = score
                        best_word = word

                if best_word:
                    chosen.append(best_word)
                    patterns.append(
                        feedback(best_word, target)
                    )  # Use the actual feedback pattern
                else:
                    # Fallback to random word
                    random_word = random.choice(wordlist)
                    chosen.append(random_word)
                    patterns.append(
                        feedback(random_word, target)
                    )  # Use the actual feedback pattern

        # Always return the guesses and their actual patterns
        return block_id, target, chosen, patterns

    # If no target word is specified, we can't solve the block
    return block_id, None, None, None


def image_to_blocks(img_path, min_cols):
    """
    Resize the image to maintain aspect ratio, with minimum columns specified.
    Return list-of-list of 5x5 np arrays: blocks[row][col]

    The three levels are:
      - 0 => black
      - 1 => yellow
      - 2 => green
    """
    img = Image.open(img_path).convert("L")  # grayscale

    # Get original dimensions
    orig_width, orig_height = img.size
    aspect_ratio = orig_height / orig_width

    # Calculate dimensions based on minimum columns
    cols = min_cols
    rows = int(cols * aspect_ratio)

    # Calculate new image dimensions (each block is 5x5 pixels)
    new_width = 5 * cols
    new_height = 5 * rows

    # Resize the image
    img = img.resize((new_width, new_height), Image.NEAREST)

    # Enhance contrast to make edges more defined
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)  # Double the contrast

    # Convert to numpy array
    arr = np.array(img)

    # Use more aggressive thresholding to create more black/outlines
    # Much lower thresholds to make the image darker with more contrast
    thresh1 = np.percentile(arr, 30)  # Lower threshold for yellow
    thresh2 = np.percentile(arr, 60)  # Much lower threshold for green

    # Create three-level array with more contrast
    three_level = np.zeros_like(arr, dtype=int)
    three_level[(arr > thresh1) & (arr <= thresh2)] = 1  # yellow
    three_level[arr > thresh2] = 2  # green
    # black remains 0 (largest portion now)

    # Optional: Apply a simple edge enhancement by making adjacent differences more pronounced
    # This creates more black borders between different regions
    for i in range(1, rows * 5 - 1):
        for j in range(1, cols * 5 - 1):
            # Check if current pixel is different from its neighbors
            current = three_level[i, j]
            neighbors = [
                three_level[i - 1, j],
                three_level[i + 1, j],
                three_level[i, j - 1],
                three_level[i, j + 1],
            ]

            # If current pixel is different from most neighbors, make it black (outline)
            if sum(1 for n in neighbors if n != current) >= 3:
                three_level[i, j] = 0

    blocks = []
    for by in range(rows):
        row_blocks = []
        for bx in range(cols):
            block = three_level[by * 5 : (by + 1) * 5, bx * 5 : (bx + 1) * 5]
            row_blocks.append(block)
        blocks.append(row_blocks)

    return blocks, cols, rows


# -----------------------------
# HTML template (use str.format). CSS braces are escaped with double {{ }}.
# -----------------------------
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
  .tile.black {{ background:#222; border: 2px solid #333; color:#ccc; }}
  .tile.yellow {{ background:#c9b458; border: 2px solid #a7943a; color:#111; }}
  .tile.green {{ background:#6aaa64; border: 2px solid #4c8f4a; }}
  .block-label {{ margin-top:6px; font-size:12px; color:#bbb; text-align:center; }}
  .container {{ display:flex; gap:18px; flex-direction:column; align-items:flex-start;}}
  .meta {{ margin-bottom:8px; color:#bbb; font-size:14px; }}
</style>
</head>
<body>
<div class="container">
  <div class="meta">Wordle Art Viewer — grid: <strong>{cols}×{rows}</strong> blocks — block size: 5×5 — wordlist size: {wordlist_count}</div>
  <div id="canvas" class="grid" style="grid-template-columns: repeat({cols}, auto);"></div>
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
            let cls = 'black';
            if (patterns && patterns[r] && patterns[r][c] === 1) cls = 'yellow';
            else if (patterns && patterns[r] && patterns[r][c] === 2) cls = 'green';
            t.className = 'tile ' + cls;
            t.textContent = (patterns ? (patterns[r][c] ? '•' : '') : '');
            blockEl.appendChild(t);
          }}
        }} else {{
          const word = guesses[r];
          const pat = patterns[r];
          for (let c = 0; c < 5; c++) {{
            const t = document.createElement('div');
            let cls = 'black';
            if (pat[c] === 1) cls = 'yellow';
            else if (pat[c] === 2) cls = 'green';
            t.className = 'tile ' + cls;
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


def export_html(outpath: Path, data, cols, rows, wordlist_count):
    # For safety in HTML, escape the closing script tag sequence inside JSON
    safe_json = json.dumps(data).replace("</", "<\\/")
    html = HTML_TEMPLATE.format(
        json_data=safe_json, cols=cols, rows=rows, wordlist_count=wordlist_count
    )
    outpath.write_text(html, encoding="utf-8")


# -----------------------------
# MAIN
# -----------------------------
def main(
    img_path,
    wordlist_path,
    min_cols,
    target_word=None,
    output_dir=".",
    output_prefix="wordle",
    max_processes=None,
    verbose=True,
):
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # load wordlist (uppercase)
    with open(wordlist_path, encoding="utf-8") as f:
        wordlist = [w.strip().upper() for w in f if len(w.strip()) == 5]

    if not wordlist:
        if verbose:
            print("Wordlist empty or no 5-letter words found.")
        return

    if verbose:
        print(f"Loaded {len(wordlist)} words from wordlist")

    # If no target word is specified, pick a random one
    if not target_word:
        target_word = random.choice(wordlist)
        if verbose:
            print(f"No target word specified, using random word: {target_word}")
    else:
        target_word = target_word.upper()
        if target_word not in wordlist:
            if verbose:
                print(
                    f"Warning: Target word '{target_word}' not in wordlist, using random word instead"
                )
            target_word = random.choice(wordlist)

    # Build pattern index instead of precomputing all pattern matches
    pattern_groups = build_pattern_index(wordlist)

    if verbose:
        print(
            f"Image processing started at {time.strftime('%H:%M:%S', time.localtime())}"
        )
    blocks, cols, rows = image_to_blocks(img_path, min_cols)
    if verbose:
        print(
            f"Image processing completed at {time.strftime('%H:%M:%S', time.localtime())}"
        )

    # Flatten blocks for multiprocessing
    flat_blocks = []
    block_positions = []
    for by in range(rows):
        for bx in range(cols):
            flat_blocks.append(blocks[by][bx])
            block_positions.append((by, bx))

    # Prepare arguments for multiprocessing
    block_args = []
    for i, block in enumerate(flat_blocks):
        block_args.append((block, wordlist, pattern_groups, target_word, i))

    if verbose:
        print(f"Solving started at {time.strftime('%H:%M:%S', time.localtime())}")

    # Use multiprocessing for solving
    if max_processes is None:
        max_processes = min(
            cpu_count(), 8
        )  # Limit to 8 processes to avoid excessive memory usage

    with Pool(processes=max_processes) as pool:
        results = pool.map(solve_block_optimized, block_args)

    if verbose:
        print(f"Solving completed at {time.strftime('%H:%M:%S', time.localtime())}")

    # Initialize blocks_result with results
    blocks_result = [[None for _ in range(cols)] for _ in range(rows)]
    solved_count = 0

    for i, (block_id, target, guesses, patterns) in enumerate(results):
        by, bx = block_positions[i]
        if target is not None:
            blocks_result[by][bx] = {
                "target": target,
                "guesses": guesses,
                "patterns": patterns,
            }
            solved_count += 1
        else:
            blocks_result[by][bx] = {
                "target": None,
                "guesses": None,
                "patterns": [r.tolist() for r in flat_blocks[i]],
            }

    meta = {
        "image": str(img_path),
        "cols": cols,
        "rows": rows,
        "total_blocks": cols * rows,
        "solved_blocks": solved_count,
        "wordlist_count": len(wordlist),
        "target_word": target_word,
    }

    # write outputs
    json_path = output_dir / f"{output_prefix}_results.json"
    html_path = output_dir / f"{output_prefix}_art.html"
    data = export_results_json(json_path, meta, blocks_result)
    export_html(html_path, data, cols, rows, len(wordlist))

    total_time = time.time() - start_time
    if verbose:
        print(f"Wrote {json_path} and {html_path}.")
        print(
            f"Solved {solved_count}/{cols * rows} blocks with target word: {target_word}"
        )
        print(f"Total processing time: {total_time:.2f} seconds")

    return {
        "json_path": str(json_path),
        "html_path": str(html_path),
        "solved_count": solved_count,
        "total_blocks": cols * rows,
        "target_word": target_word,
        "processing_time": total_time,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Wordle art from images")
    parser.add_argument("input_image", help="Input image path")
    parser.add_argument("wordlist", help="Path to wordlist file")
    parser.add_argument("min_cols", type=int, help="Minimum number of columns")
    parser.add_argument("--target-word", help="Target word (random if not specified)")
    parser.add_argument(
        "--output-dir", default=".", help="Output directory (default: current)"
    )
    parser.add_argument(
        "--output-prefix", default="wordle", help="Output file prefix (default: wordle)"
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        help="Maximum number of processes (default: min(cpu_count, 8))",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    main(
        args.input_image,
        args.wordlist,
        args.min_cols,
        args.target_word,
        args.output_dir,
        args.output_prefix,
        args.max_processes,
        not args.quiet,
    )
