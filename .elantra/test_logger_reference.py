"""Logger source/schema contracts only; C++ API execution is a separate gate."""
import ast
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pytest_plugins = ("test_controller_observation",)  # one current isolated schema per process


def source(relative="openpilot/system/loggerd/logger.cc"):
  return (ROOT / relative).read_text()


def function(text, signature):
  start = text.index(signature)
  begin = text.index("{", start)
  depth = 0
  for index in range(begin, len(text)):
    depth += (text[index] == "{") - (text[index] == "}")
    if depth == 0:
      return text[start:index+1]
  raise AssertionError("unterminated source fragment")


def test_default_off_fixed_flag_and_original_native_operation_order():
  text = source()
  gate = function(text, "bool logger_reference_observer_enabled()")
  assert 'getenv("ELANTRA_LOGGER_REFERENCE_OBSERVER")' in gate
  assert "flag[0] == '1'" in gate and "flag[1] == '\\0'" in gate
  assert "nanos_since_boot" not in gate and "Params" not in gate
  build = function(text, "kj::Array<capnp::word> logger_build_init_data(")
  assert build.count("params.readAll()") == build.count("msg.initEvent()") == 1
  assert build.index("params.readAll()") < build.index("init.initParams()") < build.index("logger_attach_reference(")
  assert "setBootlogId" not in text


def test_fixed_serializer_refuses_copy_before_bounds_and_has_no_read_or_hash():
  text = source()
  attach = function(text, "static void logger_attach_reference(")
  assert 'params_map->find("CurrentBootlog")' in attach and '(*params_map)["CurrentBootlog"]' not in attach
  assert attach.index("logger_label_valid(*owned_label)") < attach.index("setRouteLabel(*owned_label)")
  assert attach.index("logger_git_claim_valid(git)") < attach.index("setGitCommitClaim(git)")
  assert not any(re.search(r"\b" + name + r"\s*\(", attach) for name in ("Params", "read_file", "getenv", "sha256", "check_output", "initEvent"))
  for flag in ("KernelBootKnown", "SourceAuthenticated", "LoadedBytesKnown", "PolicyComplete", "PredictiveEligible", "DurableAck", "ClockDomainsVerified"):
    assert f"set{flag}(false)" in attach


def test_init_reference_roundtrip_preserves_original_bootlog_native_fields_and_unknowns(isolated_schema):
  msg = isolated_schema.Event.new_message(logMonoTime=100, valid=False)
  native = msg.init("initData")
  native.wallTimeNanos = 999999
  ref = native.init("loggerReference")
  ref.present, ref.kind, ref.ownBootlogLabelKnown = True, "bootInit", True
  ref.ownBootlogLabel = "00000001--0123456789"
  ref.initConstructionEventNs = msg.logMonoTime
  ref.kernelBootKnown = ref.sourceAuthenticated = ref.loadedBytesKnown = ref.policyComplete = ref.predictiveEligible = False
  with isolated_schema.Event.from_bytes(msg.to_bytes()) as restored:
    assert not restored.valid and restored.initData.bootlogId == ""
    assert restored.initData.wallTimeNanos == 999999
    assert restored.initData.loggerReference.ownBootlogLabel != restored.initData.bootlogId
    assert not restored.initData.loggerReference.kernelBootKnown


def test_cached_init_construction_clock_and_fresh_sentinel_are_independent(isolated_schema):
  clocks = []
  for part, timestamp in ((0, 200), (1, 300)):
    msg = isolated_schema.Event.new_message(logMonoTime=timestamp)
    native = msg.init("sentinel")
    native.type = "startOfRoute" if part == 0 else "startOfSegment"
    ref = native.init("loggerReference")
    ref.initConstructionEventNs, ref.boundaryEventNs, ref.segment = 100, msg.logMonoTime, part
    ref.present, ref.kind = True, "segmentBoundary"
    with isolated_schema.Event.from_bytes(msg.to_bytes()) as restored:
      clocks.append((restored.sentinel.loggerReference.initConstructionEventNs,
                     restored.sentinel.loggerReference.boundaryEventNs, restored.sentinel.loggerReference.segment))
  assert clocks == [(100, 200, 0), (100, 300, 1)]


def test_overlay_registers_all_owned_cpp_sources():
  tree = ast.parse(source(".elantra/sync.py"))
  modified = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "OVERLAY_MODIFIED")
  assert {"openpilot/system/loggerd/logger.cc", "openpilot/system/loggerd/logger.h", "openpilot/system/loggerd/bootlog.cc"} <= set(modified)


def test_zero_or_tied_initial_clock_cannot_claim_complete_read_bracket():
  attach = function(source(), "static void logger_attach_reference(")
  predicate = re.search(r"const bool bracket_known = ([^;]+);", attach).group(1)
  assert "reference->init_event_known" in predicate
  assert "reference->init_event_ns < read_begin" in predicate and "read_begin < read_end" in predicate
  assert "if (!bracket_known) reference->incomplete = true;" in attach


def test_boundary_clock_must_follow_consumed_known_read_end_not_only_init():
  attach = function(source(), "static void logger_attach_reference(")
  predicate = re.search(r"const bool clock_known = ([^;]+);", attach).group(1)
  assert "reference->read_end_known" in predicate and "now > reference->read_end_ns" in predicate
  assert "reference->read_end_ns = read_end;" in attach and "reference->read_end_known = bracket_known;" in attach


def test_git_commit_newline_is_source_field_only_not_original_git_claim():
  text = source()
  validator = function(text, "static bool logger_git_claim_valid(")
  assert "bool allow_newline = false" in validator and "allow_newline && value.size() == 41" in validator
  attach = function(text, "static void logger_attach_reference(")
  assert "logger_git_claim_valid(git)" in attach and "logger_git_claim_valid(source_git, true)" in attach


def brace_end(text, begin):
  depth = 0
  for index in range(begin, len(text)):
    depth += (text[index] == "{") - (text[index] == "}")
    if depth == 0:
      return index
  raise AssertionError("unterminated scoped C++ source")


def remove_block(text, marker, *, keep_body=False, remove_else=False):
  while marker in text:
    start = text.index(marker)
    begin = text.index("{", start)
    end = brace_end(text, begin)
    body = text[begin+1:end] if keep_body else ""
    if remove_else:
      following = end+1
      while text[following].isspace():
        following += 1
      assert text[following:following+4] == "else"
      end = brace_end(text, text.index("{", following))
    text = text[:start] + body + text[end+1:]
  return text


def disabled_projection(text):
  """Remove only new observer paths; fixed original token hashes are portable."""
  start, end = text.find("// Logger-reference supplement."), text.find("// ***** log metadata *****")
  if start >= 0:
    text = text[:start] + text[end:]
  text = text.replace("#include <limits>", "")
  for marker in ("if (!reference_context.enabled)", "if (!boot_reference.enabled)"):
    text = remove_block(text, marker, keep_body=True, remove_else=True)
  text = re.sub(r"^\s*if \(reference != nullptr && reference->enabled\) reference_read_(?:begin|end) = nanos_since_boot\(\);", "", text, flags=re.M)
  for marker in ("if (reference != nullptr && reference->enabled)", "if (auto* reference = log->referenceContext())"):
    text = remove_block(text, marker)
  if "struct LoggerReferenceContext" in text:
    start = text.index("struct LoggerReferenceContext")
    end = brace_end(text, text.index("{", start))
    text = text[:start] + text[end+2:]
  added_lines = [r"bool logger_reference_observer_enabled\(\);", r"inline LoggerReferenceContext\* referenceContext\(\).*",
                 r"LoggerReferenceContext (?:reference_context|boot_reference);",
                 r"(?:reference_context|boot_reference)\.enabled = logger_reference_observer_enabled\(\);",
                 r"uint64_t reference_read_begin = 0, reference_read_end = 0;"]
  text = re.sub(r"^\s*(?:" + "|".join(added_lines) + ")", "", text, flags=re.M)
  text = re.sub(r", LoggerReferenceContext\* reference(?: = nullptr)?,\s*const std::string\* owned_label(?: = nullptr)?", "", text)
  return text


def native_token_hash(text):
  tokens = re.findall(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_]\w*|\d+|::|->|<=|>=|==|!=|&&|\|\||\S', text)
  retained = [token for token in tokens if not token.startswith(("//", "/*"))]
  return hashlib.sha256(" ".join(retained).encode()).hexdigest()


def test_disabled_native_sources_match_portable_original_tokens():
  expected = {"openpilot/system/loggerd/logger.cc": "9255b4c7002cdb561a67cb163aca2bb15a0ffea31d564f4a7536a01195255731",
              "openpilot/system/loggerd/logger.h": "ae022f0f6b09d064fcfff33ba74c902b0547464411365542df3dedfec8fcb3ba",
              "openpilot/system/loggerd/bootlog.cc": "eef1a46389ae6ed8611bd293cc118f1bef932211ef3f8e0640b4d131dff62d41"}
  for relative, digest in expected.items():
    assert native_token_hash(disabled_projection(source(relative))) == digest, relative


def test_size_privacy_bounds_precede_every_dynamic_copy_and_no_full_payload_walk():
  text = source()
  label = function(text, "static bool logger_label_valid(")
  assert label.index("label.size() != 20") < label.index("for (")
  git = function(text, "static bool logger_git_claim_valid(")
  assert git.index("value.size() != 40") < git.index("for (")
  version = function(text, "static bool logger_version_claim_valid(")
  assert version.index("value.size() > 64") < version.index("for (")
  attach = function(text, "static void logger_attach_reference(")
  assert not any(name in attach for name in ("getParams(", "getCommands(", "getDongleId(", "getGitRemote(", "getGitBranch(", "getKernelArgs("))
  assert attach.index("logger_version_claim_valid(version)") < attach.index("setCompiledVersionClaim(version)")
  assert attach.index("logger_label_valid(cached->second)") < attach.index("setCachedBootlogLabel(cached->second)")


def test_original_cached_init_and_sentinel_writes_remain_one_each_in_native_order():
  next_segment = function(source(), "bool LoggerState::next()")
  assert next_segment.count("write(init_data.asBytes(), true)") == 1
  assert next_segment.index("log_sentinel(this, SentinelType::END_OF_SEGMENT)") < next_segment.index("++part")
  assert next_segment.index("++part") < next_segment.index("write(init_data.asBytes(), true)") < next_segment.index("log_sentinel(this, part > 0")
  sentinel = function(source(), "static void log_sentinel(")
  assert sentinel.count("msg.initEvent()") == sentinel.count("log->write(") == 1
  assert sentinel.index("sen.setSignal(exit_signal)") < sentinel.index("logger_attach_reference(") < sentinel.index("log->write(")
  loggerd = source("openpilot/system/loggerd/loggerd.cc")
  assert loggerd.index("logger_rotate(&s)") < loggerd.index('Params().put("CurrentRoute", s.logger.routeName())')
  boot = function(source("openpilot/system/loggerd/bootlog.cc"), "int main(")
  assert boot.index('logger_get_identifier("BootCount")') < boot.index('logger_reference_observer_enabled()')
  assert boot.index("file.write(build_boot_log().asBytes())") < boot.index('Params().put("CurrentBootlog", id.c_str())')


def test_observer_only_failure_has_sticky_loss_and_no_retry_write_or_native_valid_change():
  text = source()
  error = function(text, "static void logger_reference_error(")
  assert "reference->loss = true" in error and "reference->errors < std::numeric_limits<uint64_t>::max()" in error
  attach = function(text, "static void logger_attach_reference(")
  assert "catch (...)" in attach and "logger_reference_error(reference)" in attach
  assert "setValid" not in attach and "write(" not in attach and "retry" not in attach
  assert attach.index("ref.setPresent(false)") < attach.index("ref.setPresent(true)")


def test_maximum_added_envelope_is_bounded_without_raw_maps_or_identity_flags(isolated_schema):
  ref = isolated_schema.LoggerReferenceV1.new_message(present=True, version=1, kind="routeInit")
  ref.routeLabel = ref.ownBootlogLabel = ref.cachedBootlogLabel = "ffffffff--ffffffffff"
  ref.gitCommitClaim, ref.gitSrcCommitClaim, ref.compiledVersionClaim = "f"*40, "f"*40+"\n", "v"*64
  for name in ("initConstructionEventNs", "wallTimeNanos", "readAllBeginNs", "readAllEndNs", "boundaryEventNs", "boundarySequence", "observerErrors"):
    setattr(ref, name, 2**64-1)
  raw = ref.to_bytes()
  assert len(raw) <= 1024
  with isolated_schema.LoggerReferenceV1.from_bytes(raw) as restored:
    for name in ("kernelBootKnown", "sourceAuthenticated", "loadedBytesKnown", "policyComplete", "predictiveEligible", "durableAck", "clockDomainsVerified"):
      assert getattr(restored, name) is False
    assert restored.gitSrcCommitClaim.endswith("\n") and len(restored.compiledVersionClaim) == 64
  assert b"VIN_PRIVATE" not in raw and b"PRIVATE_PATH" not in raw
