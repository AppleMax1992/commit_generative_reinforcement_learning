import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import pandas as pd

# -----------------------------
# Lightweight ChangeScribe-like generator
# Input : raw `git diff` text (unified diff)
# Output: a template string that contains:
#   <header> ChangeScribeStart
#   Summarized Code Changes:
#   ...
#   End change part
#
# Design goals:
# - No repo checkout / no AST needed.
# - Works purely on diff text in CSV `code_diff`.
# - Produces stable, readable, condensed summaries.
# - Heuristics cover common patterns:
#   - method rename (token-level)
#   - method invocation rename
#   - identifier rename (fallback)
#   - parameter add/remove (diff-based)
#   - condition change (if/while/for)
#   - added/removed function calls
#   - statement add/remove (generic)
# -----------------------------

RE_DIFF_GIT = re.compile(r"^diff --git a/(.*?) b/(.*?)$", re.M)
RE_FILE_OLD = re.compile(r"^--- a/(.*)$", re.M)
RE_FILE_NEW = re.compile(r"^\+\+\+ b/(.*)$", re.M)
RE_HUNK = re.compile(r"^@@ .*? @@(?P<context>.*)$", re.M)

# Exclude file marker lines when collecting +/- lines
def _is_real_add(line: str) -> bool:
    return line.startswith("+") and not line.startswith("+++")
def _is_real_del(line: str) -> bool:
    return line.startswith("-") and not line.startswith("---")

def _strip_prefix(line: str) -> str:
    return line[1:] if line and line[0] in "+-" else line

def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def _short_path(path: str) -> str:
    # Keep last 3 components for readability
    parts = path.split("/")
    return "/".join(parts[-3:]) if len(parts) > 3 else path


@dataclass
class Hunk:
    header: str
    added: List[str]
    deleted: List[str]


@dataclass
class FileDiff:
    path: str
    hunks: List[Hunk]


def parse_unified_diff(diff_text: str) -> List[FileDiff]:
    """
    Parse unified diff into file->hunks with added/deleted lines.
    """
    s = diff_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = s.split("\n")

    filediffs: List[FileDiff] = []
    cur_file: Optional[FileDiff] = None
    cur_hunk: Optional[Hunk] = None

    def flush_hunk():
        nonlocal cur_hunk, cur_file
        if cur_file is not None and cur_hunk is not None:
            cur_file.hunks.append(cur_hunk)
        cur_hunk = None

    def flush_file():
        nonlocal cur_file
        if cur_file is not None:
            filediffs.append(cur_file)
        cur_file = None

    for line in lines:
        m = RE_DIFF_GIT.match(line)
        if m:
            flush_hunk()
            flush_file()
            a_path, b_path = m.group(1), m.group(2)
            cur_file = FileDiff(path=b_path or a_path, hunks=[])
            continue

        if line.startswith("@@ "):
            flush_hunk()
            cur_hunk = Hunk(header=line.strip(), added=[], deleted=[])
            continue

        if cur_hunk is None:
            continue

        if _is_real_add(line):
            cur_hunk.added.append(_strip_prefix(line))
        elif _is_real_del(line):
            cur_hunk.deleted.append(_strip_prefix(line))

    flush_hunk()
    flush_file()
    return filediffs


# -----------------------------
# Heuristics: detect condensed change statements
# -----------------------------

RE_METHOD_SIG = re.compile(
    r"""^\s*
    (public|protected|private)?\s*
    (static\s+)?(final\s+)?(synchronized\s+)?(abstract\s+)?\s*
    ([\w\<\>\[\],\s\.\?]+?)\s+   # return type-ish
    (?P<name>[A-Za-z_]\w*)\s*
    \((?P<params>[^\)]*)\)\s*
    (\{|\;)?\s*$
    """,
    re.VERBOSE,
)

RE_CALL = re.compile(r"(?P<recv>[A-Za-z_]\w*\.)?(?P<name>[A-Za-z_]\w*)\s*\(")
RE_IF = re.compile(r"^\s*if\s*\((?P<cond>.*)\)\s*\{?\s*$")
RE_WHILE = re.compile(r"^\s*while\s*\((?P<cond>.*)\)\s*\{?\s*$")
RE_FOR = re.compile(r"^\s*for\s*\((?P<cond>.*)\)\s*\{?\s*$")
RE_RETURN = re.compile(r"^\s*return\s+(?P<expr>.*)\s*;\s*$")

def _extract_method_sig(line: str) -> Optional[Tuple[str, str]]:
    """
    Return (name, params) if the line looks like a method signature.
    """
    m = RE_METHOD_SIG.match(line)
    if not m:
        return None
    name = m.group("name")
    params = _norm_ws(m.group("params"))
    return name, params

def _extract_calls(line: str) -> List[str]:
    """
    Extract possible call names from a line.
    """
    calls = []
    for m in RE_CALL.finditer(line):
        calls.append(m.group("name"))
    # de-dup preserving order
    seen = set()
    out = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def _token_renames(deleted_lines: List[str], added_lines: List[str]) -> List[Tuple[str, str, int]]:
    """
    Identify token-level renames by aligning similar lines (very lightweight).
    Returns (old, new, count).
    """
    # Collect candidate identifiers from +/- lines
    # Use simple word tokens; ignore Java keywords
    keywords = {
        "public","private","protected","static","final","class","interface","enum","return",
        "if","else","for","while","switch","case","break","continue","new","throw","throws",
        "try","catch","finally","void","int","long","double","float","boolean","char","byte",
        "short","null","true","false","this","super","extends","implements","package","import"
    }

    def tokens(s: str) -> List[str]:
        ts = re.findall(r"[A-Za-z_]\w*", s)
        return [t for t in ts if t not in keywords]

    # Build a mapping by pairing most similar lines (Jaccard)
    pairs: List[Tuple[str, str]] = []
    used_add = set()
    for d in deleted_lines:
        td = set(tokens(d))
        if not td:
            continue
        best = None
        best_score = 0.0
        for i, a in enumerate(added_lines):
            if i in used_add:
                continue
            ta = set(tokens(a))
            if not ta:
                continue
            inter = len(td & ta)
            union = len(td | ta) or 1
            score = inter / union
            if score > best_score:
                best_score = score
                best = (i, a)
        if best and best_score >= 0.5:  # reasonably similar
            used_add.add(best[0])
            pairs.append((d, best[1]))

    # From aligned pairs, infer renames as tokens that changed
    rename_counts: Dict[Tuple[str, str], int] = {}
    for d, a in pairs:
        td = tokens(d)
        ta = tokens(a)
        # If lines are similar, differences often represent rename
        # Take tokens that appear in one but not the other
        sd, sa = set(td), set(ta)
        removed = list(sd - sa)
        added = list(sa - sd)
        # Heuristic: single token replacement
        if len(removed) == 1 and len(added) == 1:
            key = (removed[0], added[0])
            rename_counts[key] = rename_counts.get(key, 0) + 1

    return [(old, new, cnt) for (old, new), cnt in sorted(rename_counts.items(), key=lambda x: -x[1])]


def summarize_file_diff(fd: FileDiff, max_items: int = 12) -> List[str]:
    """
    Produce condensed summary lines for one file.
    """
    summaries: List[str] = []
    file_short = _short_path(fd.path)

    # Gather all added/deleted lines across hunks for rename detection
    all_added = []
    all_deleted = []
    for h in fd.hunks:
        all_added.extend(h.added)
        all_deleted.extend(h.deleted)

    # 1) Method signature rename detection
    # Pair deleted and added method signatures
    del_sigs = [(name, params, line) for line in all_deleted if (tmp := _extract_method_sig(line)) for name, params in [tmp]]
    add_sigs = [(name, params, line) for line in all_added if (tmp := _extract_method_sig(line)) for name, params in [tmp]]

    # Detect same params but different name
    for dn, dp, _ in del_sigs:
        for an, ap, _ in add_sigs:
            if dp == ap and dn != an and dp != "":
                summaries.append(f"Rename method `{dn}({dp})` to `{an}({ap})` in {file_short}")
                break

    # 2) Token-level renames (covers call rename like hasException->hasThrowable)
    for old, new, cnt in _token_renames(all_deleted, all_added)[:5]:
        # Avoid trivial renames on common words
        if len(old) <= 2 or len(new) <= 2:
            continue
        summaries.append(f"Rename identifier `{old}` to `{new}` ({cnt} occurrence{'s' if cnt>1 else ''}) in {file_short}")

    # 3) Condition changes (if/while/for)
    # Very lightweight: if a deleted condition line aligns with an added condition line
    cond_pairs = []
    for d in all_deleted:
        md = RE_IF.match(d) or RE_WHILE.match(d) or RE_FOR.match(d)
        if not md:
            continue
        for a in all_added:
            ma = RE_IF.match(a) or RE_WHILE.match(a) or RE_FOR.match(a)
            if ma and type(md) == type(ma):
                if _norm_ws(md.group("cond")) != _norm_ws(ma.group("cond")):
                    cond_pairs.append((_norm_ws(md.group("cond")), _norm_ws(ma.group("cond"))))
                    break
    for before, after in cond_pairs[:3]:
        summaries.append(f"Modify conditional expression from `{before}` to `{after}` in {file_short}")

    # 4) Parameter add/remove (diff-based heuristic)
    # Look for lines with method calls where arguments list length changed
    # Pair similar calls and compare inside parentheses.
    def call_sig(line: str) -> Optional[Tuple[str, str]]:
        m = re.search(r"(?P<name>[A-Za-z_]\w*)\s*\((?P<args>[^\)]*)\)", line)
        if not m:
            return None
        return m.group("name"), _norm_ws(m.group("args"))

    del_calls = [call_sig(x) for x in all_deleted]
    add_calls = [call_sig(x) for x in all_added]
    del_calls = [c for c in del_calls if c]
    add_calls = [c for c in add_calls if c]

    for (dn, da) in del_calls:
        for (an, aa) in add_calls:
            if dn == an and da != aa:
                # Compare argument counts
                dcnt = 0 if da == "" else len([p.strip() for p in da.split(",")])
                acnt = 0 if aa == "" else len([p.strip() for p in aa.split(",")])
                if dcnt != acnt:
                    summaries.append(f"Modify arguments list when calling `{dn}` ({dcnt}→{acnt} args) in {file_short}")
                break

    # 5) Added/removed calls (very rough)
    del_call_names = set(sum([_extract_calls(x) for x in all_deleted], []))
    add_call_names = set(sum([_extract_calls(x) for x in all_added], []))
    removed_calls = sorted(list(del_call_names - add_call_names))
    added_calls = sorted(list(add_call_names - del_call_names))
    for c in removed_calls[:3]:
        summaries.append(f"Remove method call `{c}()` in {file_short}")
    for c in added_calls[:3]:
        summaries.append(f"Add method call `{c}()` in {file_short}")

    # 6) Generic fallback: statement add/remove counts
    add_n = len(all_added)
    del_n = len(all_deleted)
    if not summaries and (add_n + del_n) > 0:
        summaries.append(f"Modify {file_short} ({del_n} deletion{'s' if del_n!=1 else ''}, {add_n} addition{'s' if add_n!=1 else ''})")

    # Cap
    # De-dup preserving order
    seen = set()
    out = []
    for s in summaries:
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def lightweight_changescribe(
    diff_text: str,
    repo_name: str = "UnknownRepo",
    change_type: str = "Change",
    include_end: bool = True,
    max_files: int = 20,
    max_items_per_file: int = 12,
) -> str:
    """
    Generate ChangeScribe-like template from raw git diff text.
    """
    filediffs = parse_unified_diff(diff_text)
    filediffs = filediffs[:max_files]

    lines: List[str] = []
    lines.append(f"{repo_name} [{change_type}] ChangeScribeStart")
    lines.append("Summarized Code Changes:")

    if not filediffs:
        lines.append(" - (No parsable changes found)")
    else:
        for fd in filediffs:
            file_short = _short_path(fd.path)
            summaries = summarize_file_diff(fd, max_items=max_items_per_file)
            lines.append(f"File: {file_short}")
            for s in summaries:
                lines.append(f" - {s}")

    if include_end:
        lines.append("End change part")

    return "\n".join(lines).strip()


# -----------------------------
# CSV helper: add condensed template and optionally core_diff
# -----------------------------
def add_lightweight_changescribe_columns(
    in_csv: str,
    out_csv: str,
    code_diff_col: str = "code_diff",
    repo_col: Optional[str] = None,
    change_type_col: Optional[str] = None,
):
    """
    Read CSV, generate:
      - changescribe_text: full template (header + start + summaries + end)
      - core_diff: content between ChangeScribeStart and End change part (exclusive)
    """
    df = pd.read_csv(in_csv, low_memory=False)
    if code_diff_col not in df.columns:
        raise KeyError(f"Missing column {code_diff_col!r}; got columns: {list(df.columns)[:50]}...")

    def _gen(row) -> str:
        repo = row[repo_col] if (repo_col and repo_col in row and pd.notna(row[repo_col])) else "UnknownRepo"
        ctype = row[change_type_col] if (change_type_col and change_type_col in row and pd.notna(row[change_type_col])) else "Change"
        return lightweight_changescribe(row[code_diff_col], repo_name=str(repo), change_type=str(ctype))

    df["changescribe_text"] = df.apply(_gen, axis=1)

    # core_diff extraction
    core = []
    for t in df["changescribe_text"].astype(str).tolist():
        m = re.search(r"ChangeScribeStart\s*\n(.*?)\nEnd change part\b", t, flags=re.DOTALL)
        core.append(m.group(1).strip() if m else "")
    df["core_diff"] = core

    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote: {out_csv}")


# -----------------------------
# Quick demo (uses your earlier sample diff)
# -----------------------------
if __name__ == "__main__":
    sample = r"""diff --git a/rxjava-core/src/main/java/rx/Notification.java b/rxjava-core/src/main/java/rx/Notification.java
index ad6b81c0..866ed064 100644
--- a/rxjava-core/src/main/java/rx/Notification.java
+++ b/rxjava-core/src/main/java/rx/Notification.java
@@ -92,5 +92,5 @@ public class Notification<T> {
-    public boolean hasException() {
+    public boolean hasThrowable() {
         return isOnError() && throwable != null;
     }
@@ -126,5 +126,5 @@ public class Notification<T> {
-        if (hasException())
+        if (hasThrowable())
             str.append(" ").append(getThrowable().getMessage());
         str.append("]");
@@ -137,5 +137,5 @@ public class Notification<T> {
-        if (hasException())
+        if (hasThrowable())
             hash = hash * 31 + getThrowable().hashCode();
         return hash;
@@ -155,5 +155,5 @@ public class Notification<T> {
-        if (hasException() && !getThrowable().equals(notification.getThrowable()))
+        if (hasThrowable() && !getThrowable().equals(notification.getThrowable()))
             return false;
         return true;
"""
    print(lightweight_changescribe(sample, repo_name="rxjava", change_type="Refactor"))