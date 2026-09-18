# Rootless Docker on tars/case from a non-login shell: `source ~/.docker_env.sh`
# (this file is a copy of the one installed in the servers' home directories).
export XDG_RUNTIME_DIR=/run/user/$(id -u)
export DOCKER_HOST=unix://$XDG_RUNTIME_DIR/docker.sock
export DBUS_SESSION_BUS_ADDRESS=unix:path=$XDG_RUNTIME_DIR/bus
