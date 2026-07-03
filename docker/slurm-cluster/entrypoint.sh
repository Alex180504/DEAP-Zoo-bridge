#!/bin/bash
# Per-role entrypoint for the Slurm simulation rig: start munged, then run
# the requested Slurm daemon in the foreground (-D). No systemd, no init --
# the daemon is the container.
set -euo pipefail

# NB: no braces in the message -- a } inside ${1:?...} would end the
# expansion early and corrupt the role string.
ROLE="${1:?usage: entrypoint.sh (slurmctld|slurmd)}"

# /run is a tmpfs, so munge's socket dir has to be recreated on every
# container start; the key/seed dirs are baked into the image but re-chown
# defensively (named-volume mounts can shadow ownership).
mkdir -p /run/munge
chown -R munge:munge /run/munge /etc/munge /var/lib/munge /var/log/munge
# munged refuses to run as root; its packaged account has a nologin shell,
# hence the explicit -s.
su -s /bin/sh munge -c /usr/sbin/munged

# Fail fast (and loudly, in docker logs) if munge auth is broken -- every
# Slurm RPC would otherwise fail with opaque "protocol authentication"
# errors later.
for _ in $(seq 1 20); do
    [[ -S /run/munge/munge.socket.2 ]] && break
    sleep 0.5
done
munge -n | unmunge > /dev/null

case "$ROLE" in
    slurmctld)
        mkdir -p /var/spool/slurmctld
        chown slurm:slurm /var/spool/slurmctld
        exec /usr/sbin/slurmctld -D
        ;;
    slurmd)
        mkdir -p /var/spool/slurmd
        exec /usr/sbin/slurmd -D
        ;;
    *)
        echo "Unknown role: '$ROLE' (expected slurmctld or slurmd)" >&2
        exit 1
        ;;
esac
