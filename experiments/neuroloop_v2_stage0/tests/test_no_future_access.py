import numpy as np

from run_stage0_v2 import target_ea


def test_strict_ea_prefix_invariance():
    rng = np.random.default_rng(0)
    prefix = rng.standard_normal((4, 22, 1000), dtype=np.float32)
    suffix_a = rng.standard_normal((3, 22, 1000), dtype=np.float32)
    suffix_b = rng.standard_normal((3, 22, 1000), dtype=np.float32)
    first = target_ea(np.concatenate((prefix, suffix_a)), "strict")[: len(prefix)]
    second = target_ea(np.concatenate((prefix, suffix_b)), "strict")[: len(prefix)]
    assert np.array_equal(first, second)


if __name__ == "__main__":
    test_strict_ea_prefix_invariance()
    print("PASS test_strict_ea_prefix_invariance")
