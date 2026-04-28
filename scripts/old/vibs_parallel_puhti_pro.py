#!/usr/bin/env python3
import subprocess
import time
import sys
import re
from pathlib import Path
from collections import defaultdict

AIMS_BIN = "/projappl/project_2001912/aims.250822.scalapack.mpi.x"
ACCOUNT = "project_2001912"
PARTITION = "small"
MAX_JOBS_LIMIT = 390
GRACE_SECONDS = 120
group_size = 4

# ---------------- Slurm helpers ----------------

def run(cmd):
    return subprocess.check_output(cmd, text=True).strip()


def sbatch(jobfile):
    out = run(["sbatch", jobfile])
    return out.split()[-1]


def job_running(jid):
    out = subprocess.run(
        ["squeue", "-h", "-j", str(jid)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()

    return bool(out)


def active_jobs(user):
    return int(run(["bash", "-lc", f"squeue -u {user} -h | wc -l"]))


# ---------------- Filesystem logic ----------------

def is_finished(d: Path):
    out = d / "aims.out"
    if not out.exists():
        return False
    txt = out.read_text(errors="ignore")
    return bool(re.search(r"Have a nice day|Invalid ovlp_type", txt))


def leaf_dirs(base: Path):
    res = []
    for d in base.rglob("*"):
        if not d.is_dir():
            continue
        if any(p.is_dir() for p in d.iterdir()):
            continue
        
        res.append(d)
    return res


def classify(d: Path):
    s = str(d)
    if "negative_displacement/zero_field" in s:
        return "neg/zero"
    if "negative_displacement/field_on" in s:
        return "neg/on"
    if "positive_displacement/zero_field" in s:
        return "pos/zero"
    if "positive_displacement/field_on" in s:
        return "pos/on"
    return None


# ---------------- Jobfile writer ----------------

def write_job(jobfile: Path, jobname: str, dirs):
    with jobfile.open("w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --account={ACCOUNT}
#SBATCH -p {PARTITION}
#SBATCH --time=6:00:00
#SBATCH -J {jobname}
#SBATCH -o out/{jobname}_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40

module load StdEnv csc-tools python-data/3.8-22.10 intel-oneapi-mkl/2022.1.0 intel-oneapi-compilers/2022.1.0 intel-oneapi-mpi/2021.6.0
ulimit -s unlimited
""")
        for d in dirs:
            d = Path(d).as_posix()
            f.write(f"""
echo "Running calculation in {d}"
cd "{d}"
srun {AIMS_BIN} >> aims.out 2>> aims.err
if [ $? -ne 0 ]; then
    echo "Calculation unsuccessful in {d}, stopping all calculations for this group"
    exit 1
fi
echo "Calculation successful in {d}"
cd - >/dev/null
""")


# ---------------- Restart linking ----------------

def link_restarts(src_dirs, remaining):
    for src in src_dirs:
        src_label = classify(src)
        if src_label is None:
            continue

        csc = next(src.glob("*.csc"), None)
        if csc is None:
            continue
        csc = csc.resolve()

        for d in remaining:
            if classify(d) != src_label:
                continue

            dst = d / csc.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()

            dst.symlink_to(csc)
            print(f"Linked {dst} → {src}")
        
def group_leaf_dirs_by_base(mode: str):
    # Find all mode_* base directories
    base_dirs = [p for p in Path(".").glob(f"{mode}/mode_*") if p.is_dir()]
    # Dictionary to hold groups: {base_dir: [leaf_dirs]}
    groups = defaultdict(list)
    
    for base in base_dirs:
        for leaf in leaf_dirs(base):
            groups[base].append(leaf)
    
    return groups

def group_leaf_dirs_by_base(mode: str):
    """
    Collect all leaf directories under each base, and build a full group dict
    including placeholders for first_groups, remaining, finished, stage IDs.
    """
    base_dirs = [p for p in Path(".").glob(f"{mode}/mode_*") if p.is_dir()]

    base_groups = {}

    for base in base_dirs:
        leaves = leaf_dirs(base)  # collect all leaves
        finished = [d for d in leaves if is_finished(d)]

        base_groups[base] = {
            "leaves": leaves,           # all leaves
            "first_groups": [],         # to be filled later
            "remaining": [],     # to be filled later
            "finished": finished,       # already finished
            "stage1_ids": [],
            "stage2_done": False,
            "stage3_ids": []
        }

    return base_groups


def chunks(lst, n):
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# ---------------- Main workflow ----------------

def main():
    if len(sys.argv) != 2:
        print("Usage: launcher.py [nmodes|ters1d|ters2d]")
        sys.exit(1)
    start_time = time.time()
    print(f"Launcher started at {time.ctime(start_time)}")

    mode = sys.argv[1]
    out = Path("out"); out.mkdir(exist_ok=True)
    jobs = Path("job_launchers"); jobs.mkdir(exist_ok=True)

    base_groups = group_leaf_dirs_by_base(mode)
    job_counter = 0

    for base, info in base_groups.items():
        leaves = info["leaves"]

        # --- Split leaves into first_groups and remaining per base ---
        first_groups = []
        remaining = []
        seen = set()
        for d in leaves:
            t = classify(d)
            if t and t not in seen and len(first_groups) < 4:  # 4 types max per base
                first_groups.append(d)
                seen.add(t)
            else:
                remaining.append(d)

        # ---- Update base_groups dict ----
        info.update({
            "first_groups": first_groups,
            "remaining": remaining
        })

        # --- Print info before updating dict ---
        print(f"Base {base}:")
        print(f"  first_groups dirs:")
        for d in first_groups:
            print(f"    {d}")
        print(f"  remaining dirs count: {len(remaining)}")


        # ---- Stage 1 ----
        stage1_ids = []

        for d in first_groups:
            if is_finished(d):
                print(f"{d} is already finished → skipping Stage 1 submission")
                continue  # skip this directory

            ctrl = d / "control.in"
            if ctrl.exists():
                txt = ctrl.read_text()
                ctrl.write_text("elsi_restart write scf_converged\n" + txt)

            jobfile = jobs / f"job_group_{job_counter}_single.sh"
            write_job(jobfile, f"group_{job_counter}_single", [d])
            stage1_ids.append(sbatch(jobfile))
            print(f"{d} submitted")

            job_counter += 1

        
        info["stage1_ids"] = stage1_ids

    while True:
        all_done = True

        for base, info in base_groups.items():
            remaining = [d for d in info["remaining"] if not is_finished(d)]
            # Track Stage 1
            if not info["stage2_done"]:
                all_done = False
                if all(not job_running(j) for j in info["stage1_ids"]):
                    # Stage 1 finished for this base → do Stage 2
                    print(f"[{time.ctime()}] Stage 1 of base {base} complete")
                    link_restarts(info["first_groups"], remaining)
                    info["stage2_done"] = True
                    print(f"[{time.ctime()}] Stage 2 of base {base} complete")

                    # Submit Stage 3 jobs for this base
                    stage3_ids = []
                    for group_dirs in chunks(remaining, group_size):
                        for d in group_dirs:
                            ctrl = d / "control.in"
                            if ctrl.exists():
                                txt = ctrl.read_text()
                                ctrl.write_text("elsi_restart read\n" + txt)

                        jobfile = jobs / f"{base.name}_job_{job_counter}.sh"
                        write_job(jobfile, f"{base.name}_group_{job_counter}", group_dirs)

                        while active_jobs("planelle") >= MAX_JOBS_LIMIT:
                            print(f"Job limit reached, waiting to launch more jobs in {base}...")
                            time.sleep(60)
                        stage3_ids.append(sbatch(jobfile))
                        print(f"Group submitted: {', '.join(d.as_posix() for d in group_dirs)}")
                        job_counter += 1

                    info["stage3_ids"] = stage3_ids

            # Track Stage 3
            info["stage3_ids"] = [j for j in info["stage3_ids"] if job_running(j)]
            if info["stage3_ids"]:
                all_done = False
            elif info["stage2_done"]:
                print(f"[{time.ctime()}] Stage 3 of base {base} complete")

        if all_done:
            break

        time.sleep(10)

    end_time = time.time()
    total_time = end_time - start_time
    total_mins = total_time / 60

    print("Waiting for all Stage 3 jobs to finish...")
    print(f"Last job ended at {time.ctime(end_time)}")
    print(f"Total running time: {total_mins:.2f} minutes")


if __name__ == "__main__":
    main()

