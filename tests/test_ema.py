import torch
from torch import nn

from ddpm_eurosat.train import checkpoint_payload, update_ema


def test_update_ema_uses_configured_decay():
    model = nn.Linear(1, 1, bias=False)
    ema_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
        ema_model.weight.fill_(0.0)

    update_ema(ema_model, model, decay=0.75)

    assert torch.equal(ema_model.weight, torch.tensor([[0.5]]))


def test_ema_inference_checkpoint_preserves_raw_resume_weights():
    model = nn.Linear(1, 1, bias=False)
    ema_model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
        ema_model.weight.fill_(1.0)
    optimizer = torch.optim.AdamW(model.parameters())

    payload = checkpoint_payload(
        model,
        optimizer,
        {"training": {"ema_decay": 0.999}},
        epoch=3,
        batch_in_epoch=0,
        global_step=12,
        best_validation_loss=0.2,
        best_ema_validation_loss=0.1,
        ema_model=ema_model,
        weights_model=ema_model,
    )

    assert torch.equal(payload["model_state_dict"]["weight"], torch.tensor([[1.0]]))
    assert torch.equal(payload["training_model_state_dict"]["weight"], torch.tensor([[2.0]]))
    assert torch.equal(payload["ema_model_state_dict"]["weight"], torch.tensor([[1.0]]))

    resumed_model = nn.Linear(1, 1, bias=False)
    resumed_ema_model = nn.Linear(1, 1, bias=False)
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters())
    resumed_model.load_state_dict(payload["training_model_state_dict"])
    resumed_ema_model.load_state_dict(payload["ema_model_state_dict"])
    resumed_optimizer.load_state_dict(payload["optimizer_state_dict"])

    assert torch.equal(resumed_model.weight, torch.tensor([[2.0]]))
    assert torch.equal(resumed_ema_model.weight, torch.tensor([[1.0]]))
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu"} | ({"torch_cuda"} if torch.cuda.is_available() else set())


def test_model_only_checkpoint_omits_optimizer_and_rng_state():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.Adam(model.parameters())
    payload = checkpoint_payload(
        model,
        optimizer,
        {"model": {"architecture": "ddpm_cifar_v1"}},
        epoch=2,
        batch_in_epoch=0,
        global_step=10,
        best_validation_loss=0.1,
        include_optimizer=False,
    )
    assert "optimizer_state_dict" not in payload
    assert "rng_state" not in payload
    assert payload["model_state_dict"]
