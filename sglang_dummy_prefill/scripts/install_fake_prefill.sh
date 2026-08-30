#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/shawlleyw/sglang.git}"
BRANCH="${BRANCH:-fake_prefill}"
ENV_NAME="${ENV_NAME:-sglang-fp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/sglang-fake-prefill}"
MINICONDA_DIR="${MINICONDA_DIR:-$HOME/miniconda3}"

log() {
    printf '[install-fake-prefill] %s\n' "$*"
}

has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

download_file() {
    local url="$1"
    local output="$2"

    if has_cmd curl; then
        curl -fsSL "$url" -o "$output"
        return
    fi

    if has_cmd wget; then
        wget -qO "$output" "$url"
        return
    fi

    log "Neither curl nor wget is installed; cannot download Miniconda."
    exit 1
}

resolve_conda_exe() {
    if has_cmd conda; then
        local conda_base
        conda_base="$(conda info --base)"
        printf '%s\n' "${conda_base}/bin/conda"
        return 0
    fi

    if [ -x "${MINICONDA_DIR}/bin/conda" ]; then
        printf '%s\n' "${MINICONDA_DIR}/bin/conda"
        return 0
    fi

    return 1
}

install_miniconda() {
    local arch
    local installer_name
    local installer_url
    local installer_path

    arch="$(uname -m)"
    case "$arch" in
        x86_64)
            installer_name="Miniconda3-latest-Linux-x86_64.sh"
            ;;
        aarch64|arm64)
            installer_name="Miniconda3-latest-Linux-aarch64.sh"
            ;;
        *)
            log "Unsupported architecture: ${arch}"
            exit 1
            ;;
    esac

    installer_url="https://repo.anaconda.com/miniconda/${installer_name}"
    installer_path="$(mktemp "${TMPDIR:-/tmp}/miniconda-installer.XXXXXX.sh")"

    log "Installing Miniconda to ${MINICONDA_DIR}"
    download_file "$installer_url" "$installer_path"
    bash "$installer_path" -b -p "$MINICONDA_DIR"
    rm -f "$installer_path"
    "${MINICONDA_DIR}/bin/conda" config --set auto_activate_base false >/dev/null 2>&1 || true
}

ensure_conda_env() {
    local conda_exe="$1"

    if "$conda_exe" run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
        log "Conda env ${ENV_NAME} already exists"
    else
        log "Creating conda env ${ENV_NAME} with Python ${PYTHON_VERSION}"
        "$conda_exe" create -y -n "$ENV_NAME" "python=${PYTHON_VERSION}"
    fi
}

ensure_repo() {
    if [ -d "${INSTALL_DIR}/.git" ]; then
        log "Using existing repo at ${INSTALL_DIR}; fetching latest refs"
        git -C "$INSTALL_DIR" fetch --all --prune
    else
        if [ -e "$INSTALL_DIR" ] && [ -n "$(ls -A "$INSTALL_DIR" 2>/dev/null || true)" ]; then
            log "Install directory exists and is not empty: ${INSTALL_DIR}"
            exit 1
        fi

        log "Cloning ${REPO_URL} into ${INSTALL_DIR}"
        git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
    fi

    if ! git -C "$INSTALL_DIR" checkout "$BRANCH"; then
        git -C "$INSTALL_DIR" checkout -b "$BRANCH" "origin/$BRANCH"
    fi

    if [ -n "$(git -C "$INSTALL_DIR" status --porcelain)" ]; then
        log "Repository has local changes; skipping pull"
    else
        git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH"
    fi
}

install_project() {
    local conda_exe="$1"

    log "Upgrading pip inside ${ENV_NAME}"
    "$conda_exe" run -n "$ENV_NAME" python -m pip install --upgrade pip

    log "Installing project editable from ${INSTALL_DIR}"
    "$conda_exe" run -n "$ENV_NAME" python -m pip install -e "$INSTALL_DIR"
}

verify_install() {
    local conda_exe="$1"

    "$conda_exe" run -n "$ENV_NAME" python -c "import sglang; print('sglang', getattr(sglang, '__version__', 'unknown'))"
    "$conda_exe" run -n "$ENV_NAME" sglang --help >/dev/null
    log "Verification passed (import + CLI)"
}

main() {
    local conda_exe
    local conda_base

    if conda_exe="$(resolve_conda_exe)"; then
        log "Using existing conda: ${conda_exe}"
    else
        install_miniconda
        conda_exe="${MINICONDA_DIR}/bin/conda"
        log "Using freshly installed conda: ${conda_exe}"
    fi

    conda_base="$(dirname "$(dirname "$conda_exe")")"

    ensure_conda_env "$conda_exe"
    ensure_repo
    install_project "$conda_exe"
    verify_install "$conda_exe"

    log "Done."
    log "Activate env with: source \"${conda_base}/etc/profile.d/conda.sh\" && conda activate ${ENV_NAME}"
}

main "$@"
