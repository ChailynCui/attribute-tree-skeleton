"""Translate saved parameter paths into the explicit skeleton model hierarchy."""

from collections import OrderedDict

TOKENIZER_PATHS = {'coor_embed': 'segment_encoder.coordinates',
 'length_embed': 'segment_encoder.length',
 'midp_embed': 'segment_encoder.midpoint',
 'direction_embed': 'segment_encoder.direction',
 'angle_embed': 'segment_encoder.angle',
 'project_in': 'segment_encoder.input_projection',
 'init_sage_conv': 'segment_encoder.graph_input',
 'init_encoder_act_and_norm': 'segment_encoder.input_activation',
 'encoders': 'segment_encoder.graph_layers',
 'encoder_attn_blocks': 'segment_encoder.attention',
 'project_dim_codebook': 'segment_encoder.endpoint_projection',
 'decoder_attn_blocks': 'coordinate_decoder.attention',
 'init_decoder_conv': 'coordinate_decoder.input_convolution',
 'decoders': 'coordinate_decoder.residual_layers',
 'to_coor_logits': 'coordinate_decoder.readout'}

GENERATOR_PATHS = {'autoencoder': 'tokenizer',
 'sos_token': 'start_embedding',
 'token_embed': 'code_embedding',
 'species_embed': 'morphology.species',
 'attribute_embeds': 'morphology.buckets',
 'continuous_attribute_embedder': 'morphology.continuous',
 'attribute_prefix_type_embed': 'morphology.attribute_slots',
 'text_condition_projector': 'morphology.text_projection',
 'anonymous_text_slot_embed': 'morphology.text_slots',
 'attribute_film_pooler': 'morphology.pool',
 'attribute_film_gates': 'morphology.gates',
 'coarse_attribute_film': 'morphology.segment_film',
 'fine_attribute_film': 'morphology.endpoint_film',
 'length_predictor': 'planning.predictor',
 'length_progress_embed': 'planning.progress',
 'quantize_level_embed': 'code_level_embedding',
 'vertex_embed': 'endpoint_embedding',
 'abs_pos_emb': 'position_embedding',
 'to_line_tokens': 'segment_stage.summarize',
 'coarse_gateloop_block': 'segment_stage.recurrence',
 'decoder': 'segment_stage.attention',
 'maybe_project_coarse_to_fine': 'endpoint_stage.context_projection',
 'fine_gateloop_block': 'endpoint_stage.recurrence',
 'fine_decoder': 'endpoint_stage.attention',
 'to_logits': 'endpoint_stage.readout'}


def _replace_root(key, paths):
    root, dot, rest = key.partition(".")
    return paths.get(root, root) + (dot + rest if dot else "")


def tokenizer_state(state):
    return _translate(state, lambda key: _replace_root(key, TOKENIZER_PATHS))


def generator_state(state):
    def translate(key):
        key = _replace_root(key, GENERATOR_PATHS)
        if key.startswith("tokenizer."):
            key = "tokenizer." + _replace_root(key[len("tokenizer."):], TOKENIZER_PATHS)
        return key
    return _translate(state, translate)


def _translate(state, translate):
    converted = OrderedDict()
    for key, value in state.items():
        target = translate(key)
        if target in converted:
            raise ValueError(f"Checkpoint has conflicting parameter paths: {target}")
        converted[target] = value
    return converted
