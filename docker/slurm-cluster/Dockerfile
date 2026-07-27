# Slurm simulation node: Debian-native Slurm 22.05 (bookworm) layered on the
# shared HPC base image, so the Python stack is identical to the Dask
# rig and to what the sbatch templates expect on a real cluster.
#
# Requires deap-hpc-base:latest to exist locally — built by the test
# fixture / scripts/hpc_validate.* from docker/hpc-base/Dockerfile.
#
# No slurmdbd/mysql/accounting and no systemd: daemons run in the foreground
# (-D) under a plain entrypoint, which is what keeps this working on Docker
# Desktop/WSL2 (systemd-based Slurm images fight the container runtime).
FROM deap-hpc-base:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        slurmctld slurmd slurm-client munge \
    && rm -rf /var/lib/apt/lists/*

# Bake one munge key into the image: every container is started from the
# same image, so controller and nodes trivially share it. Test-only rig --
# a real cluster provisions its own key, never a baked one.
RUN dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 \
    && chown munge:munge /etc/munge/munge.key \
    && chmod 0400 /etc/munge/munge.key

# CPU torch + AgileRL on every node: co-evolution evaluation workers must
# unpickle FrozenRLPolicyWrapper CPU actor snapshots (EA sides evaluating
# against a live RL opponent), so the CPU compute nodes need the RL stack
# too — mirroring a real cluster with [rl-baselines] installed uniformly.
# The cpu wheel index keeps this image CUDA-free; the GPU node image
# (Dockerfile.gpu) swaps in the CUDA build. AgileRL pins pettingzoo~=1.23.1
# (conflicts with mpe2's pettingzoo>=1.24.3 already in the base image), so
# it is installed without deps and its real runtime deps are added EXCEPT
# pettingzoo — the same recipe as .github/workflows/ci.yml. The explicit
# torch pin inside the final resolve stops tensordict/accelerate from
# upgrading torch behind our back.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch==2.6.0" \
    && pip install --no-cache-dir --no-deps "agilerl==2.6.1" \
    && pip install --no-cache-dir "torch==2.6.0" packaging accelerate dill \
        flatten-dict google-cloud-storage h5py hydra-core minari omegaconf \
        redis tensordict termcolor tqdm wandb fastrand pymunk

COPY docker/slurm-cluster/slurm.conf /etc/slurm/slurm.conf
COPY docker/slurm-cluster/cgroup.conf /etc/slurm/cgroup.conf
COPY docker/slurm-cluster/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Role (slurmctld | slurmd) is passed as the container command per service
# in docker-compose.yml.
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
