"""The packed policy buffer's field order is a wire format, not a detail.

`packed_npy_inputs` is one flat float buffer that the policy JIT splits at offsets
baked in when it was captured. Both orderings produce the same total size, so a
mismatch does not raise - it silently feeds the model the wrong bytes for every
field after the first. That is what reached the car: traffic_convention and
action_t were being read out of hidden-state activations.
"""

from openpilot.sunnypilot.modeld_v2.compile_modeld import get_policy_npy_shapes


# RDF checkpoint 5's declared input_shapes.
RDF_INPUT_SHAPES = {
  'img': (1, 12, 128, 256),
  'big_img': (1, 12, 128, 256),
  'desire_pulse': (1, 33, 8),
  'traffic_convention': (1, 2),
  'action_t': (1, 2),
  'features_buffer': (1, 32, 512),
}

# Verbatim from get_policy_npy_shapes in commaai/openpilot rdf-driving
# compile_modeld.py, which is what compiled these artifacts.
COMMA_ORDER = ['desire', 'traffic_convention', 'action_t', 'prev_feat']
COMMA_SIZES = [8, 2, 2, 512]


def test_openpilot_layout_matches_comma_field_order():
  shapes, sizes = get_policy_npy_shapes(RDF_INPUT_SHAPES, is_supercombo=True, openpilot_layout=True)

  assert list(shapes.keys()) == COMMA_ORDER
  assert sizes == COMMA_SIZES


def test_openpilot_layout_offsets_place_each_field_where_the_jit_reads_it():
  shapes, sizes = get_policy_npy_shapes(RDF_INPUT_SHAPES, is_supercombo=True, openpilot_layout=True)
  offsets = {}
  cursor = 0
  for name, size in zip(shapes.keys(), sizes, strict=True):
    offsets[name] = (cursor, cursor + size)
    cursor += size

  assert offsets == {
    'desire': (0, 8),
    'traffic_convention': (8, 10),
    'action_t': (10, 12),
    'prev_feat': (12, 524),
  }
  assert cursor == 524


def test_sunnypilot_layout_is_left_alone():
  # sunnypilot compiles its own models against this order; changing it would
  # break every model in the catalog.
  shapes, sizes = get_policy_npy_shapes(RDF_INPUT_SHAPES, is_supercombo=True, openpilot_layout=False)

  assert list(shapes.keys()) == ['desire', 'prev_feat', 'traffic_convention', 'action_t']
  assert sizes == [8, 512, 2, 2]


def test_the_two_layouts_agree_on_total_size_only():
  # Precisely why the mismatch was silent: same buffer size, different meaning.
  _, comma_sizes = get_policy_npy_shapes(RDF_INPUT_SHAPES, is_supercombo=True, openpilot_layout=True)
  _, sp_sizes = get_policy_npy_shapes(RDF_INPUT_SHAPES, is_supercombo=True, openpilot_layout=False)

  assert sum(comma_sizes) == sum(sp_sizes)
  assert comma_sizes != sp_sizes
