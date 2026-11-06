import numpy as np
from brain_tumor.data_audit import connect_similar
from brain_tumor.safe_split import UnionFind


def test_hash_candidate_index_has_no_false_negatives_with_four_changed_bits():
    rng = np.random.default_rng(17)
    original = 0x2A4D87519BCDE123
    hashes = [original]
    for _ in range(80):
        positions = rng.choice(63, 4, replace=False)
        modified = original
        for position in positions:
            modified ^= 1 << int(position)
        hashes.append(modified)
    vectors = np.full((len(hashes), 64), 1 / 8, dtype=np.float32)
    groups = UnionFind(len(hashes))
    connect_similar(np.array(hashes, dtype=np.uint64), vectors, groups, 0.995)
    assert len({groups.find(i) for i in range(len(hashes))}) == 1
