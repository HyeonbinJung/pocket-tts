"""Architecture options of the flow LM, on small randomly initialized models."""

import torch

from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.modules.attention import StreamingMultiheadAttention
from pocket_tts.modules.rope import RotaryEmbedding
from pocket_tts.modules.stateful_module import StatefulModule, increment_steps, init_states
from pocket_tts.utils.config import CONFIGS_DIR, FlowLMConfig, load_config


def build_flow_lm(**overrides: object) -> FlowLMModel:
    """A 2-layer, 64-wide english flow LM with `overrides` ("transformer.num_kv_heads": 2, ...)."""
    torch.manual_seed(0)
    config = load_config(CONFIGS_DIR / "english.yaml").flow_lm.model_dump()
    config["transformer"].update(d_model=64, num_heads=4, num_layers=2)
    config["lookup_table"]["dim"] = 64
    config["flow"].update(dim=32, depth=1)
    for key, value in overrides.items():
        *path, leaf = key.split(".")
        section = config
        for part in path:
            section = section[part]
        section[leaf] = value
    flow_lm = FlowLMModel.from_pydantic_config(
        FlowLMConfig(**config), latent_dim=32, insert_bos_before_voice=True
    )
    for name, module in flow_lm.named_modules():
        if isinstance(module, StatefulModule):
            module._module_absolute_name = name
    return flow_lm.eval()


@torch.no_grad()
def sample_latents(flow_lm: FlowLMModel, num_frames: int = 3) -> torch.Tensor:
    """[num_frames, ldim] latents generated after a short text, streaming one frame at a time."""
    state = init_states(flow_lm, batch_size=1, sequence_length=32)
    text = flow_lm.conditioner(flow_lm.conditioner.prepare("Hello world."))
    latent = torch.full((1, 1, flow_lm.ldim), float("nan"))
    latents = []
    for _ in range(num_frames):
        latent, _ = flow_lm(latent, text, state, 1, 0.3, None, -4.0)
        latent = latent[:, None]
        increment_steps(flow_lm, state, increment=text.shape[1] + 1)
        text = text[:, :0]
        latents.append(latent[0, 0])
    return torch.stack(latents)


def test_z_dim_projects_backbone_output_for_both_heads():
    flow_lm = build_flow_lm(z_dim=16)
    assert flow_lm.z_proj is not None
    assert flow_lm.z_proj.weight.shape == (16, 64)
    assert flow_lm.out_eos.in_features == 16
    assert flow_lm.flow_net.cond_embed.in_features == 16
    assert sample_latents(flow_lm).isfinite().all()


def test_z_dim_defaults_to_no_projection():
    flow_lm = build_flow_lm()
    assert flow_lm.z_proj is None
    assert not any(k.startswith("z_proj") for k in flow_lm.state_dict())


def test_grouped_query_attention_matches_repeated_kv_heads():
    torch.manual_seed(0)
    rope = RotaryEmbedding()
    gqa = StreamingMultiheadAttention(64, 4, rope, attn_dim=32, num_kv_heads=2)
    mha = StreamingMultiheadAttention(64, 4, rope, attn_dim=32)
    q, k, v = gqa.in_proj.weight.detach().split([32, 16, 16])

    def repeat_heads(w: torch.Tensor) -> torch.Tensor:
        return w.view(2, 8, 64).repeat_interleave(2, dim=0).reshape(32, 64)

    mha.in_proj.weight.data = torch.cat([q, repeat_heads(k), repeat_heads(v)])
    mha.out_proj.weight.data = gqa.out_proj.weight.data
    x = torch.randn(1, 5, 64)
    torch.testing.assert_close(gqa(x, None), mha(x, None))


def test_attn_dim_and_kv_heads_in_the_flow_lm():
    flow_lm = build_flow_lm(**{"transformer.attn_dim": 32, "transformer.num_kv_heads": 2})
    weights = flow_lm.state_dict()
    assert weights["transformer.layers.0.self_attn.in_proj.weight"].shape == (32 + 2 * 16, 64)
    assert weights["transformer.layers.0.self_attn.out_proj.weight"].shape == (64, 32)
    state = init_states(flow_lm, batch_size=1, sequence_length=8)
    assert state["transformer.layers.0.self_attn"]["cache"].shape == (2, 1, 8, 2, 8)
    assert sample_latents(flow_lm).isfinite().all()
