#!/usr/bin/env python3
"""
wordle_art_export_optimized.py
Usage: python3 wordle_art_export_optimized.py input.png words.txt n [target_word]

Optimized version with:
- Vectorized image processing
- Improved caching and pattern matching
- Better parallelization for large grids
- Memory-efficient batch processing
- Persistent caching based on target word
- Automatic screenshot functionality

Produces:
 - results.json   (structured data of solved blocks, target, guesses, patterns)
 - wordle_art.html (self-contained viewer that draws the Wordle tiles)
 - screenshot.png (automatic screenshot of the HTML output)
"""

import sys
import json
from pathlib import Path
import time
from collections import defaultdict
import itertools
from functools import lru_cache
import concurrent.futures
import threading
import gc
import pickle
import hashlib
import queue

import numpy as np
from PIL import Image, ImageEnhance
import random

# Selenium imports for screenshot functionality
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


# Global cache for pattern matching - thread-safe
_pattern_cache = {}
_cache_lock = threading.Lock()

# Persistent cache for target word mappings
_persistent_cache = {}
_cache_file_lock = threading.Lock()
CACHE_DIR = Path.home() / ".img2wordle_cache"
CACHE_DIR.mkdir(exist_ok=True)

# Screenshot queue system
_screenshot_queue = queue.Queue()
_screenshot_threads = []
_screenshot_shutdown = threading.Event()

# HTML template for export
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

# -----------------------------
# WORDLE LOGIC (0=black, 1=yellow, 2=green)
# -----------------------------


def feedback(guess, target):
    """Return pattern for guess against target (0=black, 1=yellow, 2=green)."""
    # Use cache for frequently computed patterns
    cache_key = (guess, target)
    with _cache_lock:
        if cache_key in _pattern_cache:
            return _pattern_cache[cache_key]

    pattern = [0] * 5
    target_chars = list(target)

    # First pass: mark correct positions (green)
    for i in range(5):
        if guess[i] == target[i]:
            pattern[i] = 2
            target_chars[i] = None  # Mark as used

    # Second pass: mark yellow positions
    for i in range(5):
        if pattern[i] == 0 and guess[i] in target_chars:
            pattern[i] = 1
            target_chars[target_chars.index(guess[i])] = None

    with _cache_lock:
        if len(_pattern_cache) < 10000:  # Limit cache size
            _pattern_cache[cache_key] = pattern

    return pattern


def match_pattern(pattern1, pattern2):
    """Check if two patterns are identical."""
    return pattern1 == pattern2


@lru_cache(maxsize=1000)
def get_word_letter_pattern(word):
    """Get the letter pattern for a word (cached)."""
    pattern = []
    letter_map = {}
    next_char = "A"

    for letter in word:
        if letter not in letter_map:
            letter_map[letter] = next_char
            next_char = chr(ord(next_char) + 1)
        pattern.append(letter_map[letter])

    return "".join(pattern)


def build_pattern_index_optimized(wordlist):
    """
    Build an optimized index of words by their letter patterns.
    Pre-compute common patterns and create lookup tables.
    """
    print("Building optimized pattern index...")
    start_time = time.time()

    # Group words by their letter patterns
    pattern_groups = defaultdict(list)

    # Use batch processing for better memory efficiency
    batch_size = 1000
    for i in range(0, len(wordlist), batch_size):
        batch = wordlist[i : i + batch_size]
        for word in batch:
            pattern_str = get_word_letter_pattern(word)
            pattern_groups[pattern_str].append(word)

    # Pre-compute letter frequency maps for faster filtering
    # Convert to regular dict to avoid lambda functions that can't be pickled
    letter_position_index = {}
    for word in wordlist:
        for pos, letter in enumerate(word):
            if pos not in letter_position_index:
                letter_position_index[pos] = {}
            if letter not in letter_position_index[pos]:
                letter_position_index[pos][letter] = []
            letter_position_index[pos][letter].append(word)

    print(f"Optimized pattern index built in {time.time() - start_time:.2f} seconds")
    return pattern_groups, letter_position_index


# ------------------------------------------------------------------
#  GREEDY-SET-COVER word picker
# ------------------------------------------------------------------
# private to this module – one dict per worker process
_greedy_state = {}  # pattern-tuple -> word that covers it
_greedy_counter = defaultdict(int)  # how many times we reused a covering word


def find_words_for_pattern_greedy(
    target, pattern, wordlist, pattern_groups, letter_pos_index
):
    """
    Greedy-set-cover variant:
    1. if we already chose a word for this pattern → return it immediately
    2. otherwise pick the *first* word that produces the pattern (cheap)
    3. store the choice so every future identical pattern reuses the word
    """
    pat = tuple(pattern)

    # 1.  already covered by earlier greedy choice
    if pat in _greedy_state:
        _greedy_counter[pat] += 1
        return [_greedy_state[pat]]

    # 2.  fall back to the original optimised scanner
    candidates = find_words_for_pattern_greedy(
        target, desired_pat, wordlist, pattern_groups, letter_position_index
    )

    # 3.  first word becomes the permanent covering set for this pattern
    if candidates:
        word = candidates[0]
        _greedy_state[pat] = word
        return [word]

    # 4.  no exact match → use old approximate fallback
    return []


# ------------------------------------------------------------------
#  optional: diagnostics
# ------------------------------------------------------------------
def greedy_stats():
    total = sum(_greedy_counter.values())
    unique = len(_greedy_state)
    return {"greedy_hits": total, "unique_patterns_covered": unique}


def find_words_for_pattern_optimized(
    target, pattern, wordlist, pattern_groups, letter_position_index
):
    """
    Optimized word finding using pre-computed indices and vectorized operations.
    """
    # Extract requirements from pattern
    green_positions = {}
    yellow_letters = {}
    black_positions = set()
    pat = tuple(pattern)
    for i, p in enumerate(pattern):
        if p == 2:  # Green
            green_positions[i] = target[i]
        elif p == 1:  # Yellow
            if target[i] not in yellow_letters:
                yellow_letters[target[i]] = []
            yellow_letters[target[i]].append(i)
        else:  # Black
            black_positions.add(i)

    # Start with words that have correct letters in green positions
    candidates = set(wordlist)

    # Filter by green positions first (most restrictive)
    for pos, letter in green_positions.items():
        if pos in letter_position_index and letter in letter_position_index[pos]:
            candidates &= set(letter_position_index[pos][letter])
        else:
            return []  # No words match this requirement

    # Convert to list for faster iteration
    candidates = list(candidates)

    # Batch process remaining candidates
    valid_candidates = []
    batch_size = 100

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        for word in batch:
            # Quick validation before expensive feedback computation
            if not _quick_validate_word(
                word, target, green_positions, yellow_letters, black_positions
            ):
                continue

            # Only compute full feedback for promising candidates
            if feedback(word, target) == pattern:
                valid_candidates.append(word)

        # ----------  HYBRID HOOK  ----------
        if len(valid_candidates) == 0 and not _preindex_ready.is_set():
            # trigger lazy build once we are “close” to threshold
            if len(_pattern_cache) >= _PREINDEX_THRESHOLD:
                _maybe_build_preindex(target, wordlist)
                # retry with pre-index
                if pat in _preindex_map and _preindex_map[pat]:
                    return _preindex_map[pat]
        return valid_candidates


def _quick_validate_word(
    word, target, green_positions, yellow_letters, black_positions
):
    """Quick validation before expensive feedback computation."""
    # Check green positions
    for pos, letter in green_positions.items():
        if word[pos] != letter:
            return False

    # Check yellow letters are present but not in forbidden positions
    word_letter_count = {}
    for letter in word:
        word_letter_count[letter] = word_letter_count.get(letter, 0) + 1

    for letter, positions in yellow_letters.items():
        if letter not in word_letter_count:
            return False
        # Check it's not in the yellow positions
        for pos in positions:
            if word[pos] == letter:
                return False

    return True


def solve_block_batch_optimized(block_batch):
    """
    Solve multiple blocks in a single process to reduce overhead.
    """
    results = []

    for block_data in block_batch:
        (
            block,
            wordlist,
            pattern_groups,
            letter_position_index,
            target_word,
            block_id,
        ) = block_data

        # Normalize block to integer array
        block = np.array(block, dtype=int)

        if target_word:
            target = target_word.upper()
            chosen = []
            patterns = []

            # Process all 5 rows for this block
            for r in range(5):
                desired_pat = block[r].tolist()

                # Find candidates using optimized search
                candidates = _fast_lookup(
                    target, desired_pat, wordlist, pattern_groups, letter_position_index
                )

                if candidates:
                    chosen.append(candidates[0])
                    patterns.append(desired_pat)
                else:
                    # Fallback logic - find best approximate match
                    best_word = _find_best_approximate_match(
                        target, desired_pat, wordlist
                    )
                    chosen.append(best_word)
                    patterns.append(feedback(best_word, target))

            results.append((block_id, target, chosen, patterns))
        else:
            results.append((block_id, None, None, None))

    return results


def _find_best_approximate_match(target, desired_pattern, wordlist):
    """Find the word that best approximates the desired pattern."""
    best_word = None
    best_score = -1

    # Sample a subset for performance on large wordlists
    sample_size = min(500, len(wordlist))
    sample_words = random.sample(wordlist, sample_size)

    for word in sample_words:
        actual_pattern = feedback(word, target)
        score = sum(1 for i in range(5) if actual_pattern[i] == desired_pattern[i])
        if score > best_score:
            best_score = score
            best_word = word

    return best_word if best_word else random.choice(wordlist)


def image_to_blocks_vectorized(img_path, min_cols):
    """
    Vectorized image processing for better performance on large images.
    """
    print("Processing image with vectorized operations...")
    start_time = time.time()

    img = Image.open(img_path).convert("L")

    # Calculate dimensions
    orig_width, orig_height = img.size
    aspect_ratio = orig_height / orig_width
    cols = min_cols
    rows = int(cols * aspect_ratio)
    new_width = 5 * cols
    new_height = 5 * rows

    # Resize image
    img = img.resize((new_width, new_height), Image.NEAREST)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2.0)

    # Convert to numpy array for vectorized operations
    arr = np.array(img, dtype=np.float32)

    # More lenient vectorized thresholding - use wider percentile ranges
    thresh1 = np.percentile(arr, 20)  # Reduced from 30 to 20
    thresh2 = np.percentile(arr, 70)  # Increased from 60 to 70

    # Create three-level array using vectorized operations
    three_level = np.zeros_like(arr, dtype=np.int8)
    three_level[(arr > thresh1) & (arr <= thresh2)] = 1  # yellow
    three_level[arr > thresh2] = 2  # green

    # More lenient edge enhancement using convolution
    try:
        from scipy import ndimage

        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
        edges = ndimage.convolve(arr, kernel, mode="constant")
        edge_mask = np.abs(edges) > np.percentile(
            np.abs(edges), 90
        )  # Increased from 85 to 90 to be more lenient
        three_level[edge_mask] = 0
    except ImportError:
        # Fallback: manual edge detection for systems without scipy
        print("Using fallback edge detection (scipy not available)")

        # Simple edge detection using numpy operations
        diff_h = np.abs(np.diff(three_level, axis=0))
        diff_v = np.abs(np.diff(three_level, axis=1))

        # Create edge mask with more lenient thresholds
        edge_mask = np.zeros_like(three_level, dtype=bool)
        edge_mask[:-1, :] |= diff_h > 0.8  # Increased from 0.5 to 0.8
        edge_mask[1:, :] |= diff_h > 0.8  # Increased from 0.5 to 0.8
        edge_mask[:, :-1] |= diff_v > 0.8  # Increased from 0.5 to 0.8
        edge_mask[:, 1:] |= diff_v > 0.8  # Increased from 0.5 to 0.8

        # Apply edge enhancement
        three_level[edge_mask] = 0

    # Reshape into blocks more efficiently
    blocks = []
    three_level_reshaped = three_level.reshape(rows, 5, cols, 5)

    for by in range(rows):
        row_blocks = []
        for bx in range(cols):
            block = three_level_reshaped[by, :, bx, :]
            row_blocks.append(block)
        blocks.append(row_blocks)

    print(f"Image processing completed in {time.time() - start_time:.2f} seconds")
    return blocks, cols, rows


def export_results_json(path, meta, blocks_result):
    """Export results to JSON file."""
    data = {"meta": meta, "blocks": blocks_result}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return data


def export_html(path, data, cols, rows, wordlist_count):
    """Export HTML visualization."""
    # For safety in HTML, escape the closing script tag sequence inside JSON
    safe_json = json.dumps(data).replace("</", "<\\/")
    html = HTML_TEMPLATE.format(
        json_data=safe_json, cols=cols, rows=rows, wordlist_count=wordlist_count
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def load_persistent_cache(target_word):
    """Load persistent cache for a specific target word."""
    cache_file = CACHE_DIR / f"cache_{target_word.lower()}.pkl"

    with _cache_file_lock:
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except (pickle.PickleError, EOFError):
                print(
                    f"Warning: Could not load cache file {cache_file}, starting fresh"
                )
                return {}
        return {}


def save_persistent_cache(target_word, cache_data):
    """Save persistent cache for a specific target word."""
    cache_file = CACHE_DIR / f"cache_{target_word.lower()}.pkl"

    with _cache_file_lock:
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except pickle.PickleError as e:
            print(f"Warning: Could not save cache file {cache_file}: {e}")


def get_image_hash(img_path, min_cols):
    """Generate a hash key for an image with processing parameters."""
    # Create hash based on image content and processing parameters
    with open(img_path, "rb") as f:
        img_content = f.read()

    hash_input = f"{hashlib.md5(img_content).hexdigest()}_{min_cols}".encode()
    return hashlib.md5(hash_input).hexdigest()


def setup_selenium_driver(headless=True, window_size=(1920, 1080)):
    """Setup Chrome driver for screenshot functionality."""
    if not SELENIUM_AVAILABLE:
        return None

    chrome_options = Options()
    if headless:
        chrome_options.add_argument("--headless")

    # Performance optimizations for screenshots
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-plugins")
    chrome_options.add_argument("--disable-software-rasterizer")
    chrome_options.add_argument(f"--window-size={window_size[0]},{window_size[1]}")

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_window_size(window_size[0], window_size[1])
        driver.set_page_load_timeout(10)
        driver.implicitly_wait(2)
        return driver
    except Exception as e:
        print(f"Warning: Could not setup Chrome driver for screenshots: {e}")
        return None


def take_screenshot(html_path, screenshot_path, driver=None):
    """Take a screenshot of the HTML file."""
    if not SELENIUM_AVAILABLE:
        print("Warning: Selenium not available, skipping screenshot")
        return False

    driver_created = False
    if driver is None:
        driver = setup_selenium_driver()
        driver_created = True

    if driver is None:
        return False

    try:
        # Load the HTML file
        html_url = f"file://{Path(html_path).absolute()}"
        driver.get(html_url)

        # Wait for content to load
        try:
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CLASS_NAME, "grid"))
            )
        except:
            time.sleep(2)  # Fallback wait

        # Get the total size of the page content
        total_width = driver.execute_script(
            "return Math.max(document.body.scrollWidth, document.body.offsetWidth, document.documentElement.clientWidth, document.documentElement.scrollWidth, document.documentElement.offsetWidth);"
        )
        total_height = driver.execute_script(
            "return Math.max(document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);"
        )

        # Set the window size to the full content size
        driver.set_window_size(total_width, total_height)
        time.sleep(1)  # Allow resize to complete

        # Take screenshot
        driver.save_screenshot(str(screenshot_path))
        return True

    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return False
    finally:
        if driver_created and driver:
            driver.quit()


def screenshot_worker_thread(worker_id, verbose=False):
    """Worker thread that processes screenshots from the queue."""
    driver = setup_selenium_driver()
    if driver is None and verbose:
        print(f"Screenshot worker {worker_id}: Could not initialize driver")
        return

    try:
        while not _screenshot_shutdown.is_set():
            try:
                # Get task from queue with timeout
                task = _screenshot_queue.get(timeout=1.0)
                if task is None:  # Shutdown signal
                    break

                html_path, screenshot_path, callback = task

                if verbose:
                    print(
                        f"Screenshot worker {worker_id}: Processing {Path(html_path).name}"
                    )

                success = False
                if driver:
                    success = take_screenshot(html_path, screenshot_path, driver)

                # Call callback with result
                if callback:
                    callback(success, html_path, screenshot_path)

                _screenshot_queue.task_done()

            except queue.Empty:
                continue  # Check shutdown flag and try again
            except Exception as e:
                if verbose:
                    print(f"Screenshot worker {worker_id} error: {e}")
                _screenshot_queue.task_done()

    finally:
        if driver:
            driver.quit()


def start_screenshot_workers(num_workers=2, verbose=False):
    """Start screenshot worker threads."""
    global _screenshot_threads
    _screenshot_shutdown.clear()

    for i in range(num_workers):
        thread = threading.Thread(
            target=screenshot_worker_thread, args=(i, verbose), daemon=True
        )
        thread.start()
        _screenshot_threads.append(thread)

    if verbose:
        print(f"Started {num_workers} screenshot worker threads")


def stop_screenshot_workers(verbose=False):
    """Stop screenshot worker threads."""
    global _screenshot_threads

    # Signal shutdown
    _screenshot_shutdown.set()

    # Add shutdown signals to queue
    for _ in _screenshot_threads:
        _screenshot_queue.put(None)

    # Wait for threads to finish
    for thread in _screenshot_threads:
        thread.join(timeout=5.0)

    _screenshot_threads.clear()

    if verbose:
        print("Stopped screenshot worker threads")


def queue_screenshot(html_path, screenshot_path, callback=None):
    """Queue a screenshot task."""
    _screenshot_queue.put((html_path, screenshot_path, callback))


def main(
    img_path,
    wordlist_path,
    min_cols,
    target_word=None,
    output_dir=".",
    output_prefix="wordle",
    max_workers=None,
    batch_size=None,
    verbose=False,
    take_screenshot_flag=True,
    screenshot_workers=2,
):
    """
    Main function with optimized processing for large grids and caching.
    Uses threaded queue system for screenshot processing.
    """
    start_time = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load wordlist
    with open(wordlist_path, encoding="utf-8") as f:
        wordlist = [w.strip().upper() for w in f if len(w.strip()) == 5]

    if not wordlist:
        if verbose:
            print("Wordlist empty or no 5-letter words found.")
        return

    if verbose:
        print(f"Loaded {len(wordlist)} words from wordlist")

    # Handle target word
    if not target_word:
        target_word = random.choice(wordlist)
        if verbose:
            print(f"Using random target word: {target_word}")
    else:
        target_word = target_word.upper()
        if target_word not in wordlist:
            if verbose:
                print(f"Target word '{target_word}' not in wordlist, using random word")
            target_word = random.choice(wordlist)

    # Load persistent cache for this target word
    img_hash = get_image_hash(img_path, min_cols)
    cache_key = f"{img_hash}_{target_word}"
    persistent_cache = load_persistent_cache(target_word)

    if verbose:
        print(f"Using target word: {target_word}")
        if cache_key in persistent_cache:
            print(f"Found cached results for this image and target word combination")
        else:
            print(f"No cached results found, will compute fresh")

    # Check if we have cached results for this exact combination
    if cache_key in persistent_cache:
        cached_data = persistent_cache[cache_key]
        if verbose:
            print(
                f"Using cached results (saved {cached_data.get('processing_time', 0):.2f}s of processing)"
            )

        # Export cached results
        json_path = output_dir / f"{output_prefix}_results.json"
        html_path = output_dir / f"{output_prefix}_art.html"
        screenshot_path = output_dir / f"{output_prefix}_screenshot.png"

        # Export the cached data
        data = export_results_json(
            json_path, cached_data["meta"], cached_data["blocks"]
        )
        export_html(
            html_path,
            data,
            cached_data["meta"]["cols"],
            cached_data["meta"]["rows"],
            len(wordlist),
        )

        # Take screenshot using queue system
        screenshot_success = False
        if take_screenshot_flag:
            # Start screenshot workers for this operation
            start_screenshot_workers(screenshot_workers, verbose)

            # Track screenshot completion
            screenshot_result = {"success": False}

            def screenshot_callback(success, html_path, screenshot_path):
                screenshot_result["success"] = success
                if verbose and success:
                    print(f"Screenshot saved: {screenshot_path}")

            # Queue the screenshot
            queue_screenshot(html_path, screenshot_path, screenshot_callback)

            # Wait for screenshot to complete
            _screenshot_queue.join()
            screenshot_success = screenshot_result["success"]

            # Stop workers
            stop_screenshot_workers(verbose)

        return {
            "json_path": str(json_path),
            "html_path": str(html_path),
            "screenshot_path": str(screenshot_path) if screenshot_success else None,
            "solved_count": cached_data["meta"]["solved_blocks"],
            "total_blocks": cached_data["meta"]["total_blocks"],
            "target_word": target_word,
            "processing_time": 0.1,  # Minimal time for cache retrieval
            "from_cache": True,
        }

    # Build optimized pattern index
    pattern_groups, letter_position_index = build_pattern_index_optimized(wordlist)

    # Process image with vectorized operations
    blocks, cols, rows = image_to_blocks_vectorized(img_path, min_cols)
    total_blocks = cols * rows

    if verbose:
        print(f"Processing {total_blocks} blocks ({cols}x{rows})")

    # Prepare blocks for batch processing
    flat_blocks = []
    block_positions = []
    for by in range(rows):
        for bx in range(cols):
            flat_blocks.append(blocks[by][bx])
            block_positions.append((by, bx))

    # Determine optimal batch size and worker count
    if batch_size is None:
        batch_size = max(1, min(50, total_blocks // 20))  # Adaptive batch size

    if max_workers is None:
        import os

        max_workers = min(os.cpu_count(), 8)  # Reasonable default

    if verbose:
        print(f"Using {max_workers} workers with batch size {batch_size}")

    # Create batches
    block_batches = []
    for i in range(0, len(flat_blocks), batch_size):
        batch_end = min(i + batch_size, len(flat_blocks))
        batch = []
        for j in range(i, batch_end):
            batch.append(
                (
                    flat_blocks[j],
                    wordlist,
                    pattern_groups,
                    letter_position_index,
                    target_word,
                    j,
                )
            )
        block_batches.append(batch)

    if verbose:
        print(f"Solving {len(block_batches)} batches...")
        solve_start = time.time()

    # Process batches in parallel
    all_results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(solve_block_batch_optimized, batch): batch
            for batch in block_batches
        }

        completed = 0
        for future in concurrent.futures.as_completed(future_to_batch):
            batch_results = future.result()
            all_results.extend(batch_results)
            completed += 1

            if verbose and completed % max(1, len(block_batches) // 10) == 0:
                progress = (completed / len(block_batches)) * 100
                print(f"Progress: {progress:.1f}%")

    if verbose:
        solve_time = time.time() - solve_start
        print(f"Solving completed in {solve_time:.2f} seconds")

    # Reconstruct results grid
    blocks_result = [[None for _ in range(cols)] for _ in range(rows)]
    solved_count = 0

    for block_id, target, guesses, patterns in all_results:
        by, bx = block_positions[block_id]
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
                "patterns": [r.tolist() for r in flat_blocks[block_id]],
            }

    # Generate metadata
    meta = {
        "image": str(img_path),
        "cols": cols,
        "rows": rows,
        "total_blocks": total_blocks,
        "solved_blocks": solved_count,
        "wordlist_count": len(wordlist),
        "target_word": target_word,
        "processing_time": time.time() - start_time,
    }

    # Save to persistent cache for future use
    cache_data = {
        "meta": meta,
        "blocks": blocks_result,
    }
    persistent_cache[cache_key] = cache_data
    save_persistent_cache(target_word, persistent_cache)

    if verbose:
        print(f"Cached results for future use with key: {cache_key[:16]}...")

    # Export results
    json_path = output_dir / f"{output_prefix}_results.json"
    html_path = output_dir / f"{output_prefix}_art.html"
    screenshot_path = output_dir / f"{output_prefix}_screenshot.png"

    data = export_results_json(json_path, meta, blocks_result)
    export_html(html_path, data, cols, rows, len(wordlist))

    # Take screenshot automatically using queue system
    screenshot_success = False
    if take_screenshot_flag:
        if verbose:
            print("Queuing screenshot...")

        # Start screenshot workers
        start_screenshot_workers(screenshot_workers, verbose)

        # Track screenshot completion
        screenshot_result = {"success": False}

        def screenshot_callback(success, html_path, screenshot_path):
            screenshot_result["success"] = success
            if verbose and success:
                print(f"Screenshot saved: {screenshot_path}")
            elif verbose:
                print("Screenshot failed or skipped")

        # Queue the screenshot
        queue_screenshot(html_path, screenshot_path, screenshot_callback)

        # Wait for screenshot to complete
        _screenshot_queue.join()
        screenshot_success = screenshot_result["success"]

        # Stop workers
        stop_screenshot_workers(verbose)

    total_time = time.time() - start_time
    if verbose:
        print(f"Wrote {json_path} and {html_path}")
        print(f"Solved {solved_count}/{total_blocks} blocks with target: {target_word}")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Average time per block: {(total_time / total_blocks) * 1000:.2f}ms")

    # Clean up cache periodically
    with _cache_lock:
        if len(_pattern_cache) > 5000:
            _pattern_cache.clear()

    gc.collect()  # Force garbage collection for large grids

    return {
        "json_path": str(json_path),
        "html_path": str(html_path),
        "screenshot_path": str(screenshot_path) if screenshot_success else None,
        "solved_count": solved_count,
        "total_blocks": total_blocks,
        "target_word": target_word,
        "processing_time": total_time,
        "from_cache": False,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Wordle art from images (optimized version)"
    )
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
        "--max-workers",
        type=int,
        help="Maximum number of worker processes (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for processing blocks (default: adaptive)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument(
        "--no-screenshot", action="store_true", help="Skip automatic screenshot"
    )
    parser.add_argument(
        "--screenshot-workers",
        type=int,
        default=2,
        help="Number of screenshot worker threads (default: 2)",
    )

    args = parser.parse_args()

    main(
        args.input_image,
        args.wordlist,
        args.min_cols,
        args.target_word,
        args.output_dir,
        args.output_prefix,
        args.max_workers,
        args.batch_size,
        not args.quiet,
        take_screenshot_flag=not args.no_screenshot,
        screenshot_workers=args.screenshot_workers,
    )

# ------------------------------------------------------------------
#  HYBRID PRE-INDEX  (lazy, shared across batches)
# ------------------------------------------------------------------
_PREINDEX_PATH = None  # set later when target_word is known
_preindex_ready = threading.Event()  # flipped when file is ready
_preindex_map = {}  # pattern-tuple -> list[word]

_PREINDEX_THRESHOLD = 8_000  # build after LRU saw 8 k unique guesses


def _get_preindex_path(target_word):
    """Get the preindex file path for the given target word."""
    return CACHE_DIR / f"preindex_{target_word.lower()}.json"


def _maybe_build_preindex(target: str, wordlist: list[str]):
    """Called inside the LRU path once threshold is crossed."""
    global _preindex_map
    if _preindex_ready.is_set():  # already built
        return
    preindex_path = _get_preindex_path(target)
    with _cache_file_lock:  # one builder
        if preindex_path.exists():  # another worker built it
            _load_preindex(target)
            return
        try:
            if verbose:
                print("[hybrid] building pre-index …")
        except NameError:
            print("[hybrid] building pre-index …")
        tmp = defaultdict(list)
        for w in wordlist:
            tmp[tuple(feedback(w, target))].append(w)
        # atomic write
        preindex_path.write_text(
            json.dumps({",".join(map(str, k)): v for k, v in tmp.items()})
        )
        _load_preindex(target)


def _load_preindex(target_word):
    """Lightning-fast load: pure dict, no pickle."""
    global _preindex_map
    preindex_path = _get_preindex_path(target_word)
    raw = json.loads(preindex_path.read_text())
    _preindex_map = {tuple(map(int, k.split(","))): v for k, v in raw.items()}
    _preindex_ready.set()
    print(f"[hybrid] pre-index ready ({len(_preindex_map)} patterns)")


def _fast_lookup(
    target: str,
    pattern: tuple,
    wordlist: list[str],
    pattern_groups,
    letter_position_index,
):
    """LRU → pre-index → fallback scan."""
    # 1.  LRU hit  (O(1))
    pat = tuple(pattern)
    if pat in _preindex_map and _preindex_map[pat]:
        return _preindex_map[pat][0]
    # 2.  still not ready → scan like before
    return find_words_for_pattern_optimized(
        target, pattern, wordlist, pattern_groups, letter_position_index
    )
