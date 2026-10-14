#include "system/loggerd/logger.h"

#include <fstream>
#include <map>
#include <vector>
#include <iostream>
#include <sstream>
#include <random>
#include <limits>

#include "common/params.h"
#include "common/swaglog.h"
#include "common/version.h"

#include "sunnypilot/common/version.h"

// Logger-reference supplement. No reads/hashes/publishers; only bounded built values.
bool logger_reference_observer_enabled() {
  const char* flag = getenv("ELANTRA_LOGGER_REFERENCE_OBSERVER");
  return flag != nullptr && flag[0] == '1' && flag[1] == '\0';
}

static bool logger_hex(char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

static bool logger_label_valid(const std::string& label) {
  if (label.size() != 20) return false;
  for (size_t i = 0; i < 20; ++i) {
    if ((i == 8 || i == 9) ? label[i] != '-' : !logger_hex(label[i])) return false;
  }
  return true;
}

static bool logger_git_claim_valid(capnp::Text::Reader value, bool allow_newline = false) {
  if (value.size() != 40 && !(allow_newline && value.size() == 41 && value[40] == '\n')) return false;
  for (size_t i = 0; i < 40; ++i) if (!logger_hex(value[i])) return false;
  return true;
}

static bool logger_version_claim_valid(capnp::Text::Reader value) {
  if (value.size() == 0 || value.size() > 64) return false;
  for (char c : value) {
    if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
          c == '.' || c == '_' || c == '-' || c == '+' || c == ' ' || c == '(' || c == ')')) return false;
  }
  return true;
}

static void logger_reference_error(LoggerReferenceContext* reference) {
  reference->loss = true;
  if (reference->errors < std::numeric_limits<uint64_t>::max()) ++reference->errors;
}

static void logger_attach_reference(MessageBuilder& msg, LoggerReferenceContext* reference,
                                    const std::string* owned_label, const std::map<std::string, std::string>* params_map,
                                    bool route_log, int part, uint64_t read_begin, uint64_t read_end) {
  if (reference == nullptr || !reference->enabled) return;
  try {
    auto event = msg.getRoot<cereal::Event>();  // original Event; never initEvent twice
    const bool init_record = params_map != nullptr;
    auto ref = init_record ? event.getInitData().initLoggerReference() : event.getSentinel().initLoggerReference();
    ref.setPresent(false);  // final commit marker; partial observer encoding remains unavailable
    ref.setVersion(1);
    const bool label_known = owned_label != nullptr && logger_label_valid(*owned_label);
    if (label_known) {
      if (init_record && !route_log) {
        ref.setOwnBootlogLabel(*owned_label);
        ref.setOwnBootlogLabelKnown(true);
      } else {
        ref.setRouteLabel(*owned_label);
        ref.setRouteLabelKnown(true);
      }
    } else {
      reference->incomplete = true;
    }
    ref.setInitConstructionEventNs(reference->init_event_ns);
    ref.setInitConstructionKnown(reference->init_event_known);
    if (init_record) {
      ref.setKind(route_log ? cereal::LoggerReferenceV1::Kind::ROUTE_INIT : cereal::LoggerReferenceV1::Kind::BOOT_INIT);
      auto native = event.getInitData().asReader();
      ref.setWallTimeNanos(native.getWallTimeNanos());  // REALTIME; never compare to BOOTTIME
      const bool bracket_known = reference->init_event_known && reference->init_event_ns < read_begin && read_begin < read_end;
      reference->read_end_ns = read_end;
      reference->read_end_known = bracket_known;
      ref.setReadAllBeginNs(read_begin);
      ref.setReadAllEndNs(read_end);
      ref.setReadAllBracketKnown(bracket_known);
      if (!bracket_known) reference->incomplete = true;
      if (route_log) {
        const auto cached = params_map->find("CurrentBootlog");  // returned map only; never insert or reread
        ref.setCachedBootlogMapEntryPresent(cached != params_map->end());
        const bool cached_known = cached != params_map->end() && logger_label_valid(cached->second);
        ref.setCachedBootlogLabelKnown(cached_known);
        if (cached_known) ref.setCachedBootlogLabel(cached->second);
        else reference->incomplete = true;  // cached label may be missing/stale; never actual boot identity
      }
      auto git = native.getGitCommit();
      const bool git_known = logger_git_claim_valid(git);
      if (git_known) ref.setGitCommitClaim(git);
      ref.setGitCommitClaimKnown(git_known);
      auto source_git = native.getGitSrcCommit();
      const bool source_known = logger_git_claim_valid(source_git, true);
      if (source_known) ref.setGitSrcCommitClaim(source_git);
      ref.setGitSrcCommitClaimKnown(source_known);
      auto version = native.getVersion();
      const bool version_known = logger_version_claim_valid(version);
      if (version_known) ref.setCompiledVersionClaim(version);
      ref.setCompiledVersionClaimKnown(version_known);
      ref.setDirtyEnvironmentClaim(native.getDirty());
      ref.setDirtyEnvironmentClaimKnown(true);
      if (!git_known || !source_known || !version_known) reference->incomplete = true;
    } else {
      ref.setKind(cereal::LoggerReferenceV1::Kind::SEGMENT_BOUNDARY);
      const uint64_t now = event.getLogMonoTime();
      const bool clock_known = reference->init_event_known && reference->read_end_known &&
                               now > reference->read_end_ns && now > reference->last_boundary_ns;
      ref.setBoundaryEventNs(now);
      ref.setBoundaryClockKnown(clock_known);
      ref.setSegment(part >= 0 ? part : 0);
      ref.setSegmentKnown(part >= 0);
      ref.setSentinelType(event.getSentinel().getType());
      const bool sequence_known = reference->boundaries < std::numeric_limits<uint64_t>::max();
      if (sequence_known) ++reference->boundaries;
      else logger_reference_error(reference);
      ref.setBoundarySequence(reference->boundaries);
      ref.setBoundarySequenceKnown(sequence_known);
      reference->last_boundary_ns = now;
      if (!clock_known || part < 0) reference->incomplete = true;
    }
    ref.setInitReferencePresent(init_record || reference->init_reference_present);
    ref.setObserverLoss(reference->loss);
    ref.setObserverErrors(reference->errors);
    ref.setReferenceCaptureComplete(!reference->loss && !reference->incomplete);
    ref.setKernelBootKnown(false);
    ref.setSourceAuthenticated(false);
    ref.setLoadedBytesKnown(false);
    ref.setPolicyComplete(false);
    ref.setPredictiveEligible(false);
    ref.setDurableAck(false);
    ref.setClockDomainsVerified(false);
    ref.setParamsAtomic(false);
    ref.setRawFilePresenceKnown(false);
    ref.setIndividualReadClockKnown(false);
    ref.setPresent(true);
    if (init_record) reference->init_reference_present = true;
  } catch (...) {
    logger_reference_error(reference);  // observer-only failure; original native exceptions/calls stay outside this catch
  }
}

// ***** log metadata *****
kj::Array<capnp::word> logger_build_init_data(bool route_log, LoggerReferenceContext* reference, const std::string* owned_label) {
  uint64_t wall_time = nanos_since_epoch();

  MessageBuilder msg;
  auto init = msg.initEvent().initInitData();

  init.setWallTimeNanos(wall_time);
  init.setVersion(SUNNYPILOT_VERSION);
  init.setDirty(!getenv("CLEAN"));
  init.setDeviceType(Hardware::get_device_type());

  // log kernel args
  std::ifstream cmdline_stream("/proc/cmdline");
  std::vector<std::string> kernel_args;
  std::string buf;
  while (cmdline_stream >> buf) {
    kernel_args.push_back(buf);
  }

  auto lkernel_args = init.initKernelArgs(kernel_args.size());
  for (int i=0; i<kernel_args.size(); i++) {
    lkernel_args.set(i, kernel_args[i]);
  }

  init.setKernelVersion(util::read_file("/proc/version"));
  init.setOsVersion(util::read_file("/VERSION"));

  // log params
  Params params(util::getenv("PARAMS_COPY_PATH", ""));
  uint64_t reference_read_begin = 0, reference_read_end = 0;
  if (reference != nullptr && reference->enabled) reference_read_begin = nanos_since_boot();
  std::map<std::string, std::string> params_map = params.readAll();
  if (reference != nullptr && reference->enabled) reference_read_end = nanos_since_boot();

  init.setGitCommit(params_map["GitCommit"]);
  init.setGitCommitDate(params_map["GitCommitDate"]);
  init.setGitBranch(params_map["GitBranch"]);
  init.setGitRemote(params_map["GitRemote"]);
  init.setPassive(false);
  init.setDongleId(params_map["DongleId"]);

  // for prebuilt branches
  init.setGitSrcCommit(util::read_file("../../git_src_commit"));
  init.setGitSrcCommitDate(util::read_file("../../git_src_commit_date"));

  auto lparams = init.initParams().initEntries(params_map.size());
  int j = 0;
  for (auto& [key, value] : params_map) {
    auto lentry = lparams[j];
    lentry.setKey(key);
    if ( !(params.getKeyFlag(key) & DONT_LOG) ) {
      lentry.setValue(capnp::Data::Reader((const kj::byte*)value.data(), value.size()));
    }
    j++;
  }

  // log commands
  std::vector<std::string> log_commands = {
    "df -h",  // usage for all filesystems
  };

  auto hw_logs = Hardware::get_init_logs(route_log);

  auto commands = init.initCommands().initEntries(log_commands.size() + hw_logs.size());
  for (int i = 0; i < log_commands.size(); i++) {
    auto lentry = commands[i];

    lentry.setKey(log_commands[i]);

    const std::string result = util::check_output(log_commands[i]);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)result.data(), result.size()));
  }

  int i = log_commands.size();
  for (auto &[key, value] : hw_logs) {
    auto lentry = commands[i];
    lentry.setKey(key);
    lentry.setValue(capnp::Data::Reader((const kj::byte*)value.data(), value.size()));
    i++;
  }

  if (reference != nullptr && reference->enabled) {
    reference->init_event_ns = msg.getRoot<cereal::Event>().getLogMonoTime();
    reference->init_event_known = reference->init_event_ns > 0;
    logger_attach_reference(msg, reference, owned_label, &params_map, route_log, -1, reference_read_begin, reference_read_end);
  }
  return capnp::messageToFlatArray(msg);
}

std::string logger_get_identifier(std::string key) {
  // a log identifier is a 32 bit counter, plus a 10 character unique ID.
  // e.g. 000001a3--c20ba54385

  Params params;
  uint32_t cnt;
  try {
    cnt = std::stoul(params.get(key));
  } catch (std::exception &e) {
    cnt = 0;
  }
  params.put(key, std::to_string(cnt + 1));

  std::stringstream ss;
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist(0, 15);
  for (int i = 0; i < 10; ++i) {
    ss << std::hex << dist(mt);
  }

  return util::string_format("%08x--%s", cnt, ss.str().c_str());
}

std::string zstd_decompress(const std::string &in) {
  ZSTD_DCtx *dctx = ZSTD_createDCtx();
  assert(dctx != nullptr);

  // Initialize input and output buffers
  ZSTD_inBuffer input = {in.data(), in.size(), 0};

  // Estimate and reserve memory for decompressed data
  size_t estimatedDecompressedSize = ZSTD_getFrameContentSize(in.data(), in.size());
  if (estimatedDecompressedSize == ZSTD_CONTENTSIZE_ERROR || estimatedDecompressedSize == ZSTD_CONTENTSIZE_UNKNOWN) {
    estimatedDecompressedSize = in.size() * 2;  // Use a fallback size
  }

  std::string decompressedData;
  decompressedData.reserve(estimatedDecompressedSize);

  const size_t bufferSize = ZSTD_DStreamOutSize();  // Recommended output buffer size
  std::string outputBuffer(bufferSize, '\0');

  while (input.pos < input.size) {
    ZSTD_outBuffer output = {outputBuffer.data(), bufferSize, 0};

    size_t result = ZSTD_decompressStream(dctx, &output, &input);
    if (ZSTD_isError(result)) {
      break;
    }

    decompressedData.append(outputBuffer.data(), output.pos);
  }

  ZSTD_freeDCtx(dctx);
  decompressedData.shrink_to_fit();
  return decompressedData;
}


static void log_sentinel(LoggerState *log, SentinelType type, int exit_signal = 0) {
  MessageBuilder msg;
  auto sen = msg.initEvent().initSentinel();
  sen.setType(type);
  sen.setSignal(exit_signal);
  if (auto* reference = log->referenceContext()) {
    logger_attach_reference(msg, reference, &log->routeName(), nullptr, true, log->segment(), 0, 0);
  }
  log->write(msg.toBytes(), true);
}

LoggerState::LoggerState(const std::string &log_root) {
  route_name = logger_get_identifier("RouteCount");
  route_path = log_root + "/" + route_name;
  reference_context.enabled = logger_reference_observer_enabled();
  if (!reference_context.enabled) {
    init_data = logger_build_init_data(true);
  } else {
    init_data = logger_build_init_data(true, &reference_context, &route_name);
  }
}

LoggerState::~LoggerState() {
  if (rlog) {
    log_sentinel(this, SentinelType::END_OF_ROUTE, exit_signal);
    std::remove(lock_file.c_str());
  }
}

bool LoggerState::next() {
  if (rlog) {
    log_sentinel(this, SentinelType::END_OF_SEGMENT);
    std::remove(lock_file.c_str());
  }

  segment_path = route_path + "--" + std::to_string(++part);
  bool ret = util::create_directories(segment_path, 0775);
  assert(ret == true);

  lock_file = segment_path + "/rlog.lock";
  std::ofstream{lock_file};

  rlog.reset(new ZstdFileWriter(segment_path + "/rlog.zst", LOG_COMPRESSION_LEVEL));
  qlog.reset(new ZstdFileWriter(segment_path + "/qlog.zst", LOG_COMPRESSION_LEVEL));

  // log init data & sentinel type.
  write(init_data.asBytes(), true);
  log_sentinel(this, part > 0 ? SentinelType::START_OF_SEGMENT : SentinelType::START_OF_ROUTE);
  return true;
}

void LoggerState::write(uint8_t* data, size_t size, bool in_qlog) {
  rlog->write(data, size);
  if (in_qlog) qlog->write(data, size);
}
