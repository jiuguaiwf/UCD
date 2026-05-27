import os
from os.path import abspath, dirname, expanduser, isdir, isfile, join


_QM9_PROCESSED_FILES = ('train.npz', 'valid.npz', 'test.npz')


def _normalize_path(path):
    return abspath(expanduser(path))


def repo_root():
    return dirname(dirname(abspath(__file__)))


def default_qm9_datadir():
    return join(repo_root(), 'qm9', 'temp')


def qm9_datadir_candidates(preferred=None):
    candidates = []
    if preferred:
        candidates.append(preferred)

    env_datadir = os.environ.get('QM9_DATADIR')
    if env_datadir:
        candidates.append(env_datadir)

    candidates.append(default_qm9_datadir())
    candidates.append(join(dirname(repo_root()), 'radm', 'qm9', 'temp'))

    seen = set()
    for candidate in candidates:
        normalized = _normalize_path(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        yield normalized


def is_processed_qm9_datadir(datadir):
    qm9_dir = join(datadir, 'qm9')
    return all(isfile(join(qm9_dir, filename)) for filename in _QM9_PROCESSED_FILES)


def resolve_qm9_datadir(preferred=None, require_processed=False):
    for candidate in qm9_datadir_candidates(preferred):
        if require_processed and is_processed_qm9_datadir(candidate):
            return candidate
        if not require_processed and isdir(candidate):
            return candidate

    if require_processed:
        raise RuntimeError(
            'Could not find a processed QM9 datadir with qm9/train.npz, '
            'qm9/valid.npz, and qm9/test.npz. Pass --datadir /path/to/qm9/temp '
            'or set QM9_DATADIR.'
        )

    fallback = preferred if preferred else default_qm9_datadir()
    return _normalize_path(fallback)


def get_qm9_smiles_cache_path(pickle_name, preferred=None):
    datadir = resolve_qm9_datadir(preferred)
    os.makedirs(datadir, exist_ok=True)
    return join(datadir, f'{pickle_name}_smiles.pickle')
