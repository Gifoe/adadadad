import torch

from models.logpower_stem import LogPowerSincStem


def test_logpower_stem_energy_invariants():
    torch.manual_seed(0)
    stem = LogPowerSincStem(channels=22, bands=24, spatial_components=8, d_model=128).eval()
    time = torch.arange(1000, dtype=torch.float32) / 250.0
    def signal(amplitude, phase):
        return (amplitude * torch.sin(2 * torch.pi * 10.0 * time + phase))[None, None].repeat(1, 22, 1)
    with torch.no_grad():
        one = stem.power_patches(signal(1., 0.)).mean((0, 2, 3))
        phase = stem.power_patches(signal(1., torch.pi / 2)).mean((0, 2, 3))
        doubled = stem.power_patches(signal(2., 0.)).mean((0, 2, 3))
        signed = stem.signed_mean_patches(signal(1., 0.)).abs().mean()
    lo, hi = stem.band_edges_hz()
    selected = int(torch.argmin((lo + hi - 20.).abs()))
    assert one[selected] > 0
    assert (one[selected] - phase[selected]).abs() / one[selected] < .10
    assert 3.2 < doubled[selected] / one[selected] < 4.8
    assert signed < one[selected]
    assert stem(signal(1., 0.)).shape == (1, 24, 20, 128)


if __name__ == "__main__":
    test_logpower_stem_energy_invariants()
    print("PASS test_logpower_stem_energy_invariants")
