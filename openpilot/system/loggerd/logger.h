#pragma once

#include <cassert>
#include <memory>
#include <string>

#include "openpilot/cereal/messaging/messaging.h"
#include "common/util.h"
#include "common/hardware/hw.h"
#include "system/loggerd/zstd_writer.h"

constexpr int LOG_COMPRESSION_LEVEL = 10;

typedef cereal::Sentinel::SentinelType SentinelType;

// Logger-owned reference state only; never a source/kernel-boot identity.
struct LoggerReferenceContext {
  bool enabled = false, loss = false, incomplete = false;
  bool init_event_known = false, init_reference_present = false, read_end_known = false;
  uint64_t init_event_ns = 0, read_end_ns = 0, last_boundary_ns = 0, boundaries = 0, errors = 0;
};
bool logger_reference_observer_enabled();

class LoggerState {
public:
  LoggerState(const std::string& log_root = Path::log_root());
  ~LoggerState();
  bool next();
  void write(uint8_t* data, size_t size, bool in_qlog);
  inline int segment() const { return part; }
  inline const std::string& segmentPath() const { return segment_path; }
  inline const std::string& routeName() const { return route_name; }
  inline void write(kj::ArrayPtr<kj::byte> bytes, bool in_qlog) { write(bytes.begin(), bytes.size(), in_qlog); }
  inline void setExitSignal(int signal) { exit_signal = signal; }
  inline LoggerReferenceContext* referenceContext() { return reference_context.enabled ? &reference_context : nullptr; }

protected:
  int part = -1, exit_signal = 0;
  std::string route_path, route_name, segment_path, lock_file;
  kj::Array<capnp::word> init_data;
  std::unique_ptr<ZstdFileWriter> rlog, qlog;
  LoggerReferenceContext reference_context;
};

kj::Array<capnp::word> logger_build_init_data(bool route_log = false, LoggerReferenceContext* reference = nullptr,
                                           const std::string* owned_label = nullptr);
std::string logger_get_identifier(std::string key);
std::string zstd_decompress(const std::string &in);
