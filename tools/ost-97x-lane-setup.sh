#!/usr/bin/env bash
# ost-97x joins the tailnet as its own node, so the single-GPU lane can reach its 5050.
#
# The lane (bench/fleet_single.py, probes/run_engine_probe.sh) asks one thing of a box of
# its own: an ssh alias that answers, on a host with docker, the NVIDIA runtime and an
# x86_64 image. This box is the operator's Windows PC, and its GPU answers inside WSL2
# (/usr/lib/wsl/lib/libcuda.so is mounted there, nvidia-smi sees the 5050), so the lane's
# host is that WSL2 distro -- not Windows, which has no sshd and no /proc/meminfo for the
# lane's evidence to read.
#
# Reaching the distro is the part Windows would otherwise own. WSL2 sits behind a NAT
# (eth0 172.26.x) that the Windows host's tailnet address does not cross, and the two ways
# to bridge it are both Windows-wide: a portproxy chasing an address WSL2 redraws every
# boot, or mirrored networking, which rewrites the network for every session on this box.
# Neither is needed. Tailscale is already installed inside WSL2 and running there with a
# kernel TUN -- only logged out. Logging THAT daemon in gives the distro its own node and
# its own 100.x address; the Windows side forwards nothing and keeps its own node.
#
# Run in WSL2 on ost-97x, as choiceoh, NOT under sudo -- step 1 reads srv4 over the
# invoking user's ssh key, and the steps that need root call sudo themselves:
#
#   bash tools/ost-97x-lane-setup.sh             # ssh + the tailnet node
#   bash tools/ost-97x-lane-setup.sh --docker    # and docker with the NVIDIA runtime
#
# Every step checks before it changes anything, so a second run reports and does nothing.
set -euo pipefail

want_docker=0
for arg in "$@"; do
  case $arg in
    --docker) want_docker=1;;
    -h|--help) sed -n '2,25p' "$0"; exit 0;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 2;;
  esac
done

[ "$(id -u)" != 0 ] || { echo "run as choiceoh, not root: step 1 needs the invoking user's ssh key" >&2; exit 2; }
grep -qi microsoft /proc/version || { echo "this is the WSL2 side of ost-97x; /proc/version says otherwise" >&2; exit 2; }

NODE=${OST97X_NODE:-ost-97x}          # the tailnet name; the controller's ~/.ssh/config owns the alias
# The controller's alias stays ost-97x whatever the tailnet calls this box: it is the check lane's default host
# (FLEET_CHECK_GPU_HOST) and the key of this box's facts in bench/fleet_single.py HOSTS (floor, budget, image, flashinfer).
ALIAS=ost-97x
KEYS_FROM=${OST97X_KEYS_FROM:-srv4}   # the fleet box that already lists both controllers' keys
# Which keys to take, by their comment in $KEYS_FROM's authorized_keys. Deliberately not
# written down here: this repository is public, and which keypairs reach which box is the
# tailnet's business, not the repository's.
: "${OST97X_CONTROLLER_KEYS:?set it to the controller key comments, space separated -- see bench/OST_97X_LANE.md}"
read -r -a CONTROLLERS <<< "$OST97X_CONTROLLER_KEYS"
step() { printf '\n== %s\n' "$*"; }

# ---- 1. the two controllers' keys ------------------------------------------------------
# Not ssh-copy-id: that needs an sshd here to copy INTO, and there is no password to open
# it with (step 2 turns password auth off, and it is off by default for a fresh account).
# $KEYS_FROM already lists both controllers' keys in its own authorized_keys, so the keys
# come from there, by comment, and nothing else in that file is copied.
step "authorized keys for ${CONTROLLERS[*]}"
install -d -m 700 ~/.ssh
touch ~/.ssh/authorized_keys
theirs=$(mktemp) && trap 'rm -f "$theirs"' EXIT
if ! timeout 20 ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$KEYS_FROM" 'cat ~/.ssh/authorized_keys' > "$theirs"; then
  echo "  cannot read $KEYS_FROM:~/.ssh/authorized_keys -- add the two keys by hand and re-run" >&2
  exit 1
fi
cp -a ~/.ssh/authorized_keys ~/.ssh/authorized_keys.bak."$(date +%Y%m%d%H%M%S)"
for comment in "${CONTROLLERS[@]}"; do
  line=$(grep -F " $comment" "$theirs" | head -1 || true)
  if [ -z "$line" ]; then
    echo "  NOT ON $KEYS_FROM: $comment -- that controller will not get in until its key is here"
    continue
  fi
  key=$(printf '%s' "$line" | awk '{print $2}')
  if grep -qF "$key" ~/.ssh/authorized_keys; then
    echo "  already authorized: $comment"
  else
    printf '%s\n' "$line" >> ~/.ssh/authorized_keys
    echo "  ADDED: $comment"
  fi
done
chmod 600 ~/.ssh/authorized_keys

# ---- 2. sshd ---------------------------------------------------------------------------
# Keys only, this account only. The distro's eth0 is NAT behind Windows and nothing routes
# to it from outside, so the tailnet node from step 3 is the only way in; ListenAddress is
# left alone rather than pinned to an address a re-auth could redraw.
step "sshd: keys only, choiceoh only"
conf=/etc/ssh/sshd_config.d/10-ost-97x.conf
if [ -f "$conf" ]; then
  echo "  $conf already written"
else
  sudo tee "$conf" >/dev/null <<'CONF'
# The single-GPU lane's door (tools/ost-97x-lane-setup.sh). The lane speaks ssh with
# BatchMode=yes and scp/rsync: no password, no keyboard-interactive, one account.
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
AllowUsers choiceoh
CONF
  echo "  wrote $conf"
fi
sudo sshd -t || { echo "  sshd config is invalid -- nothing enabled" >&2; exit 1; }
# Ubuntu 24.04 ships socket activation: ssh.socket owns :22 and starts ssh.service on the
# first connection, so on an idle box `is-active ssh` reads inactive while the door is in
# fact open and enabled at boot. `enable --now ssh.service` on top of that live socket is
# how this step aborted the whole script on 2026-09-15 -- sshd cannot bind a port the
# socket already holds -- and the tailnet login below never ran. Take the listener that is
# already there; only a box with neither gets the service enabled.
if [ "$(systemctl is-enabled ssh.socket 2>/dev/null)" = enabled ]; then
  sudo systemctl start ssh.socket
  [ "$(systemctl is-active ssh.service)" != active ] || sudo systemctl reload ssh.service
  echo "  ssh.socket owns :22 (socket activation, enabled at boot); drop-in applied"
elif [ "$(systemctl is-active ssh.service)" = active ]; then
  sudo systemctl reload ssh.service
  echo "  ssh.service running; drop-in applied"
else
  sudo systemctl enable --now ssh.service
  echo "  ssh.service enabled and started"
fi

# ---- 3. the tailnet node ---------------------------------------------------------------
# --accept-dns=false: WSL2's /etc/resolv.conf is generated by Windows and already resolves
# MagicDNS names through the Windows node (this box ssh's to srv4 by name today). Letting a
# second tailscaled rewrite it is how that stops working.
step "tailnet node '$NODE'"
if tailscale status >/dev/null 2>&1; then
  echo "  already up as $(tailscale status --self --peers=false 2>/dev/null | awk '{print $2}') -- $(tailscale ip -4 | head -1)"
else
  echo "  logging this distro's tailscaled in; authenticate in the browser it prints"
  sudo tailscale up --hostname="$NODE" --accept-dns=false
fi

# ---- 4. docker with the NVIDIA runtime -------------------------------------------------
if [ "$want_docker" = 1 ]; then
  step "docker with the NVIDIA runtime"
  . /etc/os-release
  # Ask whether docker RUNS, not whether a path exists. On this box /usr/bin/docker was a
  # dangling symlink into /mnt/wsl/docker-desktop/ left by an uninstalled Docker Desktop:
  # `command -v` happens to answer no for a dangling link, so the install below ran -- but
  # let that mount come back without a working daemon behind it and the path test would say
  # yes, skip the install, and hand a broken CLI to nvidia-ctk and systemctl under `set -e`.
  if docker --version >/dev/null 2>&1; then
    echo "  docker already installed: $(docker --version)"
  else
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $VERSION_CODENAME stable" \
      | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
  fi
  if dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    echo "  nvidia-container-toolkit already installed"
  else
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
      | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
  fi
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl enable --now docker
  sudo systemctl restart docker
  id -nG "$USER" | tr ' ' '\n' | grep -qx docker || {
    sudo usermod -aG docker "$USER"
    echo "  added $USER to the docker group -- 'wsl.exe --shutdown' from Windows, then reopen, for it to count"
  }
  # The runtime mounts the driver's own tools into the container, so a stock image proves it.
  step "does a container see the 5050?"
  sudo docker run --rm --gpus all ubuntu:24.04 nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv \
    || echo "  the container could not reach the GPU -- see bench/OST_97X_LANE.md"
fi

# ---- 5. what the controller still owns -------------------------------------------------
step "done -- on each controller, the side this script does not own"
addr=$(tailscale ip -4 2>/dev/null | head -1 || true)
cat <<CONTROLLER

  ~/.ssh/config:

    Host $ALIAS
        HostName ${addr:-<this node's tailnet address>}
        User $USER
        IdentityFile ~/.ssh/id_ed25519
        IdentitiesOnly yes

  then, for the queue: nothing to export. The alias is $ALIAS even where the tailnet name is
  not ($NODE here): it is the check lane's default host (FLEET_CHECK_GPU_HOST, bench/fleet.sh)
  and the key of this box's floor, budget, image and flashinfer in bench/fleet_single.py HOSTS.
  A check goes there with

    bash bench/fleet.sh run --gpu --check <session> [est] [note] -- bash probes/run_engine_check.sh ...

  and check it:

    ssh $ALIAS 'hostname; nvidia-smi --query-gpu=name --format=csv,noheader'
    python3 bench/fleet_single.py evidence --host $ALIAS --gib 4

CONTROLLER
