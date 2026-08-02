"""Registration of the model folders used by the llama.cpp nodes.

ComfyUI keeps a global registry (``folder_paths.folder_names_and_paths``) that
maps a folder key to the directories it is searched in and the file extensions
that are considered valid.  GGUF weights are not part of stock ComfyUI, so the
keys are created here on import.
"""

from __future__ import annotations

import os
from typing import List

GGUF_EXTENSIONS = {".gguf"}

# Folder keys created/extended by this pack.  ``llm`` holds the language model
# weights, ``mmproj`` the multimodal projector files used by vision models.
# Both are also allowed to live in the other folder, since projector files are
# routinely shipped next to the model they belong to.
LLM_FOLDER = "llm"
MMPROJ_FOLDER = "mmproj"

# Directory names under ComfyUI/models that are picked up automatically when
# they exist.  Different node packs settled on different spellings, and users
# should not have to move their weights around to switch.
_EXTRA_DIRS = {
    LLM_FOLDER: ("llm", "LLM", "llms", "llama", "gguf", "GGUF"),
    # Projector files are usually downloaded next to the model they belong to.
    MMPROJ_FOLDER: ("mmproj", "llm", "LLM"),
}


def _folder_paths():
    try:
        import folder_paths  # provided by ComfyUI at runtime
    except ImportError:
        return None
    return folder_paths


def _add_extensions(entry) -> None:
    """Make sure ``.gguf`` is accepted for an existing folder key."""
    extensions = entry[1]
    if isinstance(extensions, set):
        extensions.update(GGUF_EXTENSIONS)
    elif isinstance(extensions, list):
        for ext in GGUF_EXTENSIONS:
            if ext not in extensions:
                extensions.append(ext)


def _register(folder_paths, key: str, directories: List[str]) -> None:
    registry = folder_paths.folder_names_and_paths
    if key not in registry:
        registry[key] = ([], set(GGUF_EXTENSIONS))
    entry = registry[key]
    _add_extensions(entry)

    known = entry[0]
    for directory in directories:
        if directory not in known:
            known.append(directory)


def register_model_folders() -> None:
    """Create the ``llm``/``mmproj`` folder keys inside ComfyUI.

    Safe to call more than once and a no-op outside of ComfyUI so the package
    stays importable for tests.
    """
    folder_paths = _folder_paths()
    if folder_paths is None:
        return

    models_dir = getattr(folder_paths, "models_dir", None)
    if not models_dir:
        return

    for key, candidates in _EXTRA_DIRS.items():
        directories = []
        for name in candidates:
            path = os.path.join(models_dir, name)
            # The primary directory is always registered so ComfyUI creates it
            # on demand; the aliases only count when the user actually has them.
            if name == key or os.path.isdir(path):
                directories.append(path)
        _register(folder_paths, key, directories)

    # Create the canonical directory so it shows up in the file manager.
    primary = os.path.join(models_dir, LLM_FOLDER)
    try:
        os.makedirs(primary, exist_ok=True)
    except OSError:
        pass


def list_models(folder: str = LLM_FOLDER) -> List[str]:
    """Return the GGUF files available for a folder key.

    The result always contains at least one entry so the combo widget stays
    usable when no weights are installed yet.
    """
    folder_paths = _folder_paths()
    names: List[str] = []
    if folder_paths is not None:
        try:
            names = list(folder_paths.get_filename_list(folder))
        except Exception:
            names = []
    names = sorted({name for name in names if name.lower().endswith(".gguf")})
    return names or ["<no .gguf files found>"]


def resolve_model_path(folder: str, name: str, override: str = "") -> str:
    """Turn a combo selection (or an override path) into an absolute path."""
    override = (override or "").strip().strip('"')
    if override:
        path = os.path.expanduser(override)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model path does not exist: {path}")
        return path

    if not name or name.startswith("<"):
        raise FileNotFoundError(
            f"No model selected. Put .gguf files into ComfyUI/models/{folder}/ "
            "or set the path override on the loader node."
        )

    folder_paths = _folder_paths()
    if folder_paths is not None:
        path = folder_paths.get_full_path(folder, name)
        if path:
            return path

    if os.path.isfile(name):
        return name
    raise FileNotFoundError(f"Could not find '{name}' in the '{folder}' model folder.")
