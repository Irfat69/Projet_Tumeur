"""Expose les bibliotheques NVIDIA de pip a TensorFlow dans le venv Linux/WSL2."""

import importlib.util
import os
from pathlib import Path
import sys
import sysconfig


def ensure_link(source: Path, destination: Path) -> int:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return 0
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"Chemin existant conserve, verifier manuellement : {destination}")
    destination.symlink_to(os.path.relpath(source, destination.parent))
    return 1


def main():
    if sys.platform != "linux" or sys.prefix == sys.base_prefix:
        raise SystemExit("Executer ce script dans le venv du projet sous Linux/WSL2.")

    spec = importlib.util.find_spec("tensorflow")
    nvidia_dir = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    libraries = sorted(nvidia_dir.glob("*/lib/*.so*"))
    ptxas = nvidia_dir / "cuda_nvcc" / "bin" / "ptxas"
    if spec is None or not spec.origin or not libraries or not ptxas.is_file():
        raise SystemExit("Installer d'abord : python -m pip install -r requirements/gpu.txt")

    tensorflow_dir = Path(spec.origin).parent
    created = 0
    for library in libraries:
        created += ensure_link(library, tensorflow_dir / library.name)

    # Le RUNPATH de la roue TensorFlow 2.21 cherche cuda_runtime/lib mais omet
    # cusolver/lib. Garder cuSOLVER dans un chemin deja connu du chargeur.
    for library in sorted((nvidia_dir / "cusolver" / "lib").glob("libcusolver.so*")):
        created += ensure_link(library, nvidia_dir / "cuda_runtime" / "lib" / library.name)

    created += ensure_link(ptxas, Path(sys.prefix) / "bin" / "ptxas")
    print(f"Configuration GPU : {created} lien(s) cree(s) dans {sys.prefix}.")
    print("Redemarrer les processus Python / noyaux Jupyter avant de verifier le GPU.")


if __name__ == "__main__":
    main()
