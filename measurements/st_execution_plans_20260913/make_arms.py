"""Create immutable measurement branches; never change the checkout or push.

Production-shaped fleet brackets select committed arms, not environment knobs.
Only the three execution facts in boot.py differ. These branches must not be
merged: the main experiment implementation keeps production defaults off.
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
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("--prefix", default="codex/gb10-execution-r1")
    args = ap.parse_args()
    base = git("rev-parse", args.base + "^{commit}")
    path = "engine/profiles/glm53/boot.py"
    source = git("show", f"{base}:{path}") + "\n"
    old = "execution_overlap=0, early_observe=0, prefill_tiles=1"
    if source.count(old) != 1:
        raise ValueError("base must keep exactly one default-off production contract")
    arms = {"baseline": dict(sha=base, values=[0, 0, 1])}
    variants = {"tp": (1, 0, 1), "early": (0, 1, 1), "prefill2": (0, 0, 2),
                "prefill4": (0, 0, 4), "combined": (1, 1, 2)}
    with tempfile.TemporaryDirectory(prefix="st-execution-index-") as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        for name, (overlap, early, tiles) in variants.items():
            branch = f"{args.prefix}-{name}"
            git("check-ref-format", "refs/heads/" + branch)
            replacement = f"execution_overlap={overlap}, early_observe={early}, prefill_tiles={tiles}"
            blob = git("hash-object", "-w", "--stdin", input=source.replace(old, replacement))
            git("read-tree", base, env=env)
            git("update-index", "--cacheinfo", "100644", blob, path, env=env)
            tree = git("write-tree", env=env)
            found = subprocess.run(["git", "rev-parse", "--verify", "--quiet", "refs/heads/" + branch],
                                   capture_output=True, text=True)
            if found.returncode == 0:
                sha = found.stdout.strip()
                if git("rev-parse", sha + "^{tree}") != tree or git("rev-parse", sha + "^") != base:
                    raise ValueError(f"{branch} belongs to a different arm; choose a new prefix")
            else:
                sha = git("commit-tree", tree, "-p", base,
                          input=f"measurement only: GB10 {name} execution arm\n\nDo not merge or deploy as a default.\n")
                git("update-ref", "refs/heads/" + branch, sha, "0" * 40)
            arms[name] = dict(sha=sha, branch=branch, values=[overlap, early, tiles])
    report = dict(scope="isolated measurement commits; unqualified, never merge these arm-only commits",
                  base=base, changed_path=path, value_order=["execution_overlap", "early_observe", "prefill_tiles"],
                  arms=arms)
    output = Path(__file__).with_name("arms.json")
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
