from pathlib import Path

import pytest
import torch

from fastvideo.optimal_transport import (
    OptimalTransportMap,
    PotentialNetwork,
    extract_ot_map_state_dict,
    load_ot_checkpoint,
    load_optimal_transport_map,
)


def test_paper_network_shapes():
    ot_map = OptimalTransportMap()
    potential = PotentialNetwork()

    features = torch.randn(3, 512)
    assert ot_map(features).shape == (3, 512)
    assert potential(features).shape == (3,)


def test_load_raw_and_structured_checkpoints(tmp_path):
    expected = OptimalTransportMap()
    raw_path = tmp_path / "raw.pt"
    structured_path = tmp_path / "structured.pt"
    torch.save(expected.state_dict(), raw_path)
    torch.save({"step": 12, "T": expected.state_dict()}, structured_path)

    raw_model = load_optimal_transport_map(raw_path)
    structured_model = load_optimal_transport_map(structured_path)

    for key, expected_value in expected.state_dict().items():
        assert torch.equal(raw_model.state_dict()[key], expected_value)
        assert torch.equal(structured_model.state_dict()[key], expected_value)
    assert not any(parameter.requires_grad for parameter in raw_model.parameters())


def test_structured_trainer_checkpoint_round_trip(tmp_path):
    ot_map = OptimalTransportMap()
    potential = PotentialNetwork()
    ot_optimizer = torch.optim.Adam(ot_map.parameters(), lr=1e-4)
    potential_optimizer = torch.optim.Adam(potential.parameters(), lr=1e-4)
    ot_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        ot_optimizer, T_max=10
    )
    potential_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        potential_optimizer, T_max=10
    )

    inputs = torch.randn(2, 512)
    (ot_map(inputs).square().mean() + potential(inputs).square().mean()).backward()
    ot_optimizer.step()
    potential_optimizer.step()
    ot_scheduler.step()
    potential_scheduler.step()

    checkpoint_path = tmp_path / "trainer.pt"
    torch.save(
        {
            "format_version": 1,
            "step": 7,
            "T": ot_map.state_dict(),
            "f": potential.state_dict(),
            "T_opt": ot_optimizer.state_dict(),
            "f_opt": potential_optimizer.state_dict(),
            "sched_T": ot_scheduler.state_dict(),
            "sched_f": potential_scheduler.state_dict(),
        },
        checkpoint_path,
    )

    checkpoint = load_ot_checkpoint(checkpoint_path)
    restored_map = OptimalTransportMap()
    restored_potential = PotentialNetwork()
    restored_ot_optimizer = torch.optim.Adam(restored_map.parameters(), lr=1e-4)
    restored_potential_optimizer = torch.optim.Adam(
        restored_potential.parameters(), lr=1e-4
    )
    restored_ot_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_ot_optimizer, T_max=10
    )
    restored_potential_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        restored_potential_optimizer, T_max=10
    )
    restored_map.load_state_dict(extract_ot_map_state_dict(checkpoint))
    restored_potential.load_state_dict(checkpoint["f"])
    restored_ot_optimizer.load_state_dict(checkpoint["T_opt"])
    restored_potential_optimizer.load_state_dict(checkpoint["f_opt"])
    restored_ot_scheduler.load_state_dict(checkpoint["sched_T"])
    restored_potential_scheduler.load_state_dict(checkpoint["sched_f"])
    assert checkpoint["step"] == 7
    assert restored_map.state_dict().keys() == ot_map.state_dict().keys()
    assert restored_potential.state_dict().keys() == potential.state_dict().keys()
    assert restored_ot_optimizer.state_dict()["state"]
    assert restored_potential_optimizer.state_dict()["state"]
    assert restored_ot_scheduler.last_epoch == ot_scheduler.last_epoch
    assert restored_potential_scheduler.last_epoch == potential_scheduler.last_epoch


def test_extract_ddp_prefixed_state_dict():
    model = OptimalTransportMap()
    prefixed = {
        f"module.{key}": value for key, value in model.state_dict().items()
    }
    extracted = extract_ot_map_state_dict({"T": prefixed})
    assert set(extracted) == set(model.state_dict())


def test_incompatible_checkpoint_has_clear_error(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save(OptimalTransportMap(hidden_dim=64).state_dict(), path)
    with pytest.raises(ValueError, match="incompatible"):
        load_optimal_transport_map(path)


def test_bundled_ot_checkpoint_is_compatible():
    checkpoint_path = (
        Path(__file__).resolve().parents[1]
        / "pretrained"
        / "OT_map_156000.pt"
    )
    if not checkpoint_path.exists():
        pytest.skip("Bundled OT checkpoint is not present in this checkout.")
    model = load_optimal_transport_map(checkpoint_path)
    assert model(torch.randn(1, 512)).shape == (1, 512)
