"""
long_prompt segmentation: first sentence + several segments starting with ['tag', ...];
randomly pick one segment by probability and concatenate it with the first sentence.
Consistent with preprocess_dataset Step3 / training-time random_idx_list.
"""
from __future__ import annotations

import ast
import random
import re
from typing import Any, Dict, List, Optional, Tuple

BracketSegment = Dict[str, Any]  # keys: tags: List[str], text: str


def parse_long_prompt_bracket_segments(long_prompt_text: str) -> Tuple[str, List[BracketSegment]]:
    """
    The first sentence is the text before the first [' ;
    each following segment is a ['...'] block plus the body text until the next [' or end of string.
    """
    if not long_prompt_text or not isinstance(long_prompt_text, str):
        return "", []

    idx = long_prompt_text.find("['")
    if idx == -1:
        s = long_prompt_text.strip()
        return (s if s.endswith(".") else s + ".", [])

    first_sentence = long_prompt_text[:idx].strip()
    rest = long_prompt_text[idx:]

    segments: List[BracketSegment] = []
    tag_pattern = re.compile(r"\[([^\]]+)\]")
    pos = 0
    while pos < len(rest):
        m = tag_pattern.match(rest, pos)
        if not m:
            break
        bracket_full = m.group(0)
        try:
            tags = ast.literal_eval(bracket_full)
        except Exception:
            tags = [m.group(1).strip().strip("'\"")]
        if not isinstance(tags, list):
            tags = [str(tags)]
        end_tag = m.end()
        next_m = tag_pattern.search(rest, end_tag)
        if next_m:
            text = rest[end_tag : next_m.start()].strip()
            pos = next_m.start()
        else:
            text = rest[end_tag:].strip()
            pos = len(rest)
        segments.append({"tags": [str(t) for t in tags], "text": text})
    return first_sentence, segments

def select_segment_weighted(
    segments: List[BracketSegment]
) -> Optional[BracketSegment]:
    """
    1) Ignore overview
    2) Uniform random sample among remaining segments
    If none remain (e.g. all are overview), fall back to uniform sample over all segments.
    """
    if not segments:
        return None

    def is_overview(seg: BracketSegment) -> bool:
        return any(t.lower() == "overview" for t in seg["tags"])

    candidates = [s for s in segments if not is_overview(s)]
    if not candidates:
        candidates = list(segments)
    return random.choice(candidates)

def compose_two_part_prompt(first_sentence: str, segment: BracketSegment) -> str:
    a = first_sentence.strip()
    if not a.endswith("."):
        a = a + "."
    b = segment["text"].strip()
    if not b.endswith("."):
        b = b + "."
    return f"{a} {b}"


def process_long_prompt_segmented(
    long_prompt_text: str,
    relative_image_path: str,
) -> Tuple[str, List[str]]:
    """
    Returns:
        processed_prompt: first sentence + one randomly chosen segment
        selected_tags: tag list of the chosen segment (for writing random_idx_list.txt)
    """
    _ = relative_image_path

    first, segments = parse_long_prompt_bracket_segments(long_prompt_text)
    if not segments:
        out = first.strip()
        if out and not out.endswith("."):
            out = out + "."
        return out, []

    chosen = select_segment_weighted(segments)
    if chosen is None:
        chosen = random.choice(segments)

    prompt = compose_two_part_prompt(first, chosen)
    tags = list(chosen["tags"])
    return prompt, tags


def tags_to_mask_idx(tags: List[str]) -> int:
    """
    Backward-compatible mapper kept for dataset import compatibility.
    """
    if not tags:
        return -1
    tl = [str(t).lower() for t in tags]
    if "overview" in tl:
        return -1
    if "hair" in tl:
        return -1
    if any(x in tl for x in ("l_eye", "r_eye")):
        return 1
    if any(x in tl for x in ("l_brow", "r_brow")):
        return 2
    if any(x in tl for x in ("u_lip", "l_lip", "mouth")):
        return 3
    if "nose" in tl:
        return 4
    if any(x in tl for x in ("l_ear", "r_ear", "ear_r")):
        return 5
    return -1


def format_random_list_line(relative_path: str, tags: List[str]) -> str:
    """Same as the example format: path: ['u_lip', 'l_lip', 'mouth'] (repr)"""
    return f"{relative_path.strip()}: {repr(tags)}"
