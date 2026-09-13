"""Freeze one candidate-only consumer arm without changing the checkout.

Both production facts are selected in an isolated measurement commit. No
baseline arm is created and this measurement-only commit must not be merged.
Run only after the matching kernel gate qualifies the implementation base.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile


def git(*args, input=None, env=None):
    return subprocess.check_output(["git", *args], input=input, text=True, env=env).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", required=True)
    ap.add_argument("--branch", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    if not args.branch.startswith("codex/"):
        ap.error("measurement branch must use the codex/ prefix")
    git("check-ref-format", "refs/heads/" + args.branch)
    base = git("rev-parse", args.base + "^{commit}")
    path = "engine/profiles/glm53/boot.py"
    source = git("show", f"{base}:{path}") + "\n"
    old, new = "deferred_kda=0, terminal_mhc=0", "deferred_kda=1, terminal_mhc=1"
    if source.count(old) != 1:
        raise ValueError("base must contain exactly one default-off production contract")
    with tempfile.TemporaryDirectory(prefix="st-terminal-arm-") as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        blob = git("hash-object", "-w", "--stdin", input=source.replace(old, new))
        git("read-tree", base, env=env)
        git("update-index", "--cacheinfo", "100644", blob, path, env=env)
        tree = git("write-tree", env=env)
    found = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "refs/heads/" + args.branch],
                           capture_output=True, text=True)
    if found.returncode == 0:
        sha = found.stdout.strip()
        if git("rev-parse", sha + "^{tree}") != tree or git("rev-parse", sha + "^") != base:
            raise ValueError("branch already names another arm; choose a new branch")
    else:
        sha = git("commit-tree", tree, "-p", base,
                  input="measurement only: accepted KDA and terminal feature consumer\n\nDo not merge this arm commit.\n")
        git("update-ref", "refs/heads/" + args.branch, sha, "0" * 40)
    assert git("diff", "--name-only", base, sha) == path
    report = dict(scope="candidate-only measurement arm; no GPU admission and no default promotion",
                  base=base, sha=sha, branch=args.branch, changed_path=path,
                  production_facts=dict(deferred_kda=1, terminal_mhc=1),
                  consumer=dict(boots=1, c1_runs=2, c1_contexts=[32000, 128000],
                                c4_runs=1, c4_contexts=[32000], kv_gib=5,
                                completion_budget=3072, reasoning_budget=2048,
                                individual_max_tokens=1024, fixed_decode_reps=0,
                                verdict_metrics=["decode_steps", "generated_tok_s", "acceptance"]))
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
