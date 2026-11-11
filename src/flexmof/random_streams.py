"""Versioned per-draw noise streams for reproducible generation."""

import hashlib


def auxiliary_noise_seed(seed, domain, version=2):
    """Keep archived v1 draws reproducible; v2 separates stream namespaces.

    The historical seed+1 stream overlaps the next draw's initial seed when
    draw seeds are consecutive. A domain-separated hash avoids that systematic
    reuse without changing a draw's recorded initial seed.
    """
    if version == 1:
        return seed+1
    if version != 2:
        raise ValueError('Noise stream version must be 1 or 2')
    payload = f'flexmof-noise-v2:{domain}:{int(seed)}'.encode('ascii')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little') % (2**63)
