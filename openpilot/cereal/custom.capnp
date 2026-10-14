using Cxx = import "/include/c++.capnp";
$Cxx.namespace("cereal");

@0xb526ba661d550a59;

# custom.capnp: a home for empty structs reserved for custom forks
# These structs are guaranteed to remain reserved and empty in mainline
# cereal, so use these if you want custom events in your fork.

# DO rename the structs
# DON'T change the identifier (e.g. @0x81c2f05a394cf4af)

struct ModularAssistiveDrivingSystem {
  state @0 :ModularAssistiveDrivingSystemState;
  enabled @1 :Bool;
  active @2 :Bool;
  available @3 :Bool;

  enum ModularAssistiveDrivingSystemState {
    disabled @0;
    paused @1;
    enabled @2;
    softDisabling @3;
    overriding @4;
  }
}

struct IntelligentCruiseButtonManagement {
  state @0 :IntelligentCruiseButtonManagementState;
  sendButton @1 :SendButtonState;
  vTarget @2 :Float32;

  enum IntelligentCruiseButtonManagementState {
    inactive @0;      # No button press or default state
    preActive @1;     # Pre-active state before transitioning to increasing or decreasing
    increasing @2;    # Increasing speed
    decreasing @3;    # Decreasing speed
    holding @4;       # Holding steady speed
  }

  enum SendButtonState {
    none @0;
    increase @1;
    decrease @2;
  }
}

# Same struct as Log.RadarState.LeadData
struct LeadData {
  dRel @0 :Float32;
  yRel @1 :Float32;
  vRel @2 :Float32;
  aRel @3 :Float32;
  vLead @4 :Float32;
  dPath @6 :Float32;
  vLat @7 :Float32;
  vLeadK @8 :Float32;
  aLeadK @9 :Float32;
  fcw @10 :Bool;
  status @11 :Bool;
  aLeadTau @12 :Float32;
  modelProb @13 :Float32;
  radar @14 :Bool;
  radarTrackId @15 :Int32 = -1;

  aLeadDEPRECATED @5 :Float32;
}

struct SelfdriveStateSP @0x81c2f05a394cf4af {
  mads @0 :ModularAssistiveDrivingSystem;
  intelligentCruiseButtonManagement @1 :IntelligentCruiseButtonManagement;
  buttonsPressed @2 :UInt16;
  buttonsReleaseToggle @3 :UInt16;

  enum AudibleAlert {
    none @0;

    engage @1;
    disengage @2;
    refuse @3;

    warningSoft @4;
    warningImmediate @5;

    prompt @6;
    promptRepeat @7;
    promptDistracted @8;

    # unused, these are reserved for upstream events so we don't collide
    reserved9 @9;
    reserved10 @10;
    reserved11 @11;
    reserved12 @12;
    reserved13 @13;
    reserved14 @14;
    reserved15 @15;
    reserved16 @16;
    reserved17 @17;
    reserved18 @18;
    reserved19 @19;
    reserved20 @20;
    reserved21 @21;
    reserved22 @22;
    reserved23 @23;
    reserved24 @24;
    reserved25 @25;
    reserved26 @26;
    reserved27 @27;
    reserved28 @28;
    reserved29 @29;
    reserved30 @30;

    promptSingleLow @31;
    promptSingleHigh @32;
  }
}

struct ModelManagerSP @0xaedffd8f31e7b55d {
  activeBundle @0 :ModelBundle;
  selectedBundle @1 :ModelBundle;
  availableBundles @2 :List(ModelBundle);

  struct DownloadUri {
    uri @0 :Text;
    sha256 @1 :Text;
  }

  enum DownloadStatus {
    notDownloading @0;
    downloading @1;
    downloaded @2;
    cached @3;
    failed @4;
    verifying @5;
  }

  struct DownloadProgress {
    status @0 :DownloadStatus;
    progress @1 :Float32;
    eta @2 :UInt32;
  }

  struct Chunk {
    fileName @0 :Text;
    sha256 @1 :Text;
  }

  struct Artifact {
    fileName @0 :Text;
    downloadUri @1 :DownloadUri;
    downloadProgress @2 :DownloadProgress;
    chunks @3 :List(Chunk);
  }

  struct Model {
    type @0 :Type;
    artifact @1 :Artifact;  # Main artifact
    metadata @2 :Artifact;  # Metadata artifact

    enum Type {
      supercombo @0;
      navigation @1;
      vision @2;
      policy @3;
      offPolicy @4;
      onPolicy @5;
      chunked @6;
    }
  }

  enum Runner {
    snpe @0;
    tinygrad @1;
    stock @2;
  }

  struct Override {
    key @0 :Text;
    value @1 :Text;
  }

  struct ModelBundle {
    index @0 :UInt32;
    internalName @1 :Text;
    displayName @2 :Text;
    models @3 :List(Model);
    status @4 :DownloadStatus;
    generation @5 :UInt32;
    environment @6 :Text;
    runner @7 :Runner;
    is20hz @8 :Bool;
    ref @9 :Text;
    minimumSelectorVersion @10 :UInt32;
    overrides @11 :List(Override);
  }
}

struct LongitudinalPlanSP @0xf35cc4560bbf6ec2 {
  dec @0 :DynamicExperimentalControl;
  longitudinalPlanSource @1 :LongitudinalPlanSource;
  smartCruiseControl @2 :SmartCruiseControl;
  speedLimit @3 :SpeedLimit;
  vTarget @4 :Float32;
  aTarget @5 :Float32;
  events @6 :List(OnroadEventSP.Event);
  e2eAlerts @7 :E2eAlerts;

  struct DynamicExperimentalControl {
    state @0 :DynamicExperimentalControlState;
    enabled @1 :Bool;
    active @2 :Bool;

    enum DynamicExperimentalControlState {
      acc @0;
      blended @1;
    }
  }

  struct SmartCruiseControl {
    vision @0 :Vision;
    map @1 :Map;

    struct Vision {
      state @0 :VisionState;
      vTarget @1 :Float32;
      aTarget @2 :Float32;
      currentLateralAccel @3 :Float32;
      maxPredictedLateralAccel @4 :Float32;
      enabled @5 :Bool;
      active @6 :Bool;
    }

    struct Map {
      state @0 :MapState;
      vTarget @1 :Float32;
      aTarget @2 :Float32;
      enabled @3 :Bool;
      active @4 :Bool;
    }

    enum VisionState {
      disabled @0; # System disabled or inactive.
      enabled @1; # No predicted substantial turn on vision range.
      entering @2; # A substantial turn is predicted ahead, adapting speed to turn comfort levels.
      turning @3; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      leaving @4; # Road ahead straightens. Start to allow positive acceleration.
      overriding @5; # System overriding with manual control.
    }

    enum MapState {
      disabled @0; # System disabled or inactive.
      enabled @1; # No predicted substantial turn on map range.
      turning @2; # Actively turning. Managing acceleration to provide a roll on turn feeling.
      overriding @3; # System overriding with manual control.
    }
  }

  struct SpeedLimit {
    resolver @0 :Resolver;
    assist @1 :Assist;

    struct Resolver {
      speedLimit @0 :Float32;
      distToSpeedLimit @1 :Float32;
      source @2 :Source;
      speedLimitOffset @3 :Float32;
      speedLimitLast @4 :Float32;
      speedLimitFinal @5 :Float32;
      speedLimitFinalLast @6 :Float32;
      speedLimitValid @7 :Bool;
      speedLimitLastValid @8 :Bool;
    }

    struct Assist {
      state @0 :AssistState;
      enabled @1 :Bool;
      active @2 :Bool;
      vTarget @3 :Float32;
      aTarget @4 :Float32;
    }

    enum Source {
      none @0;
      car @1;
      map @2;
    }

    enum AssistState {
      disabled @0;
      inactive @1; # No speed limit set or not enabled by parameter.
      preActive @2;
      pending @3; # Awaiting new speed limit.
      adapting @4; # Reducing speed to match new speed limit.
      active @5; # Cruising at speed limit.
    }
  }

  enum LongitudinalPlanSource {
    cruise @0;
    sccVision @1;
    sccMap @2;
    speedLimitAssist @3;
  }

  struct E2eAlerts {
    greenLightAlert @0 :Bool;
    leadDepartAlert @1 :Bool;
  }
}

struct OnroadEventSP @0xda96579883444c35 {
  events @0 :List(Event);

  struct Event {
    name @0 :EventName;

    # event types
    enable @1 :Bool;
    noEntry @2 :Bool;
    warning @3 :Bool;   # alerts presented only when  enabled or soft disabling
    userDisable @4 :Bool;
    softDisable @5 :Bool;
    immediateDisable @6 :Bool;
    preEnable @7 :Bool;
    permanent @8 :Bool; # alerts presented regardless of openpilot state
    overrideLateral @10 :Bool;
    overrideLongitudinal @9 :Bool;
  }

  enum EventName {
    lkasEnable @0;
    lkasDisable @1;
    manualSteeringRequired @2;
    manualLongitudinalRequired @3;
    silentLkasEnable @4;
    silentLkasDisable @5;
    silentBrakeHold @6;
    silentWrongGear @7;
    silentReverseGear @8;
    silentDoorOpen @9;
    silentSeatbeltNotLatched @10;
    silentParkBrake @11;
    controlsMismatchLateral @12;
    hyundaiRadarTracksConfirmed @13;
    experimentalModeSwitched @14;
    wrongCarModeAlertOnly @15;
    pedalPressedAlertOnly @16;
    laneTurnLeft @17;
    laneTurnRight @18;
    speedLimitPreActive @19;
    speedLimitActive @20;
    speedLimitChanged @21;
    speedLimitPending @22;
    e2eChime @23;
    laneChangeRoadEdge @24;
    bigModelReady @25;
  }
}

struct CarParamsSP @0x80ae746ee2596b11 {
  flags @0 :UInt32;        # flags for car specific quirks in sunnypilot
  safetyParam @1 : Int16;  # flags for sunnypilot's custom safety flags
  pcmCruiseSpeed @3 :Bool;
  intelligentCruiseButtonManagementAvailable @4 :Bool;
  enableGasInterceptor @5 :Bool;

  neuralNetworkLateralControl @2 :NeuralNetworkLateralControl;

  struct NeuralNetworkLateralControl {
    model @0 :Model;
    fuzzyFingerprint @1 :Bool;

    struct Model {
      path @0 :Text;
      name @1 :Text;
    }
  }
}

struct CarControlSP @0xa5cd762cd951a455 {
  mads @0 :ModularAssistiveDrivingSystem;
  params @1 :List(Param);
  leadOne @2 :LeadData;
  leadTwo @3 :LeadData;
  intelligentCruiseButtonManagement @4 :IntelligentCruiseButtonManagement;

  struct Param {
    key @0 :Text;
    type @2 :ParamType;
    value @3 :Data;

    valueDEPRECATED @1 :Text; # The data type change may cause issues with backwards compatibility.
  }

  enum ParamType {
    string @0;
    bool @1;
    int @2;
    float @3;
    time @4;
    json @5;
    bytes @6;
  }
}

struct BackupManagerSP @0xf98d843bfd7004a3 {
  backupStatus @0 :Status;
  restoreStatus @1 :Status;
  backupProgress @2 :Float32;
  restoreProgress @3 :Float32;
  lastError @4 :Text;
  currentBackup @5 :BackupInfo;
  backupHistory @6 :List(BackupInfo);

  enum Status {
    idle @0;
    inProgress @1;
    completed @2;
    failed @3;
  }

  struct Version {
    major @0 :UInt16;
    minor @1 :UInt16;
    patch @2 :UInt16;
    build @3 :UInt16;
    branch @4 :Text;
  }

  struct MetadataEntry {
    key @0 :Text;
    value @1 :Text;
    tags @2 :List(Text);
  }

  struct BackupInfo {
    deviceId @0 :Text;
    version @1 :UInt32;
    config @2 :Text;
    isEncrypted @3 :Bool;
    createdAt @4 :Text;  # ISO timestamp
    updatedAt @5 :Text;  # ISO timestamp
    sunnypilotVersion @6 :Version;
    backupMetadata @7 :List(MetadataEntry);
  }
}

struct CarStateSP @0xb86e6369214c01c8 {
  speedLimit @0 :Float32;
}

struct LiveMapDataSP @0xf416ec09499d9d19 {
  speedLimitValid @0 :Bool;
  speedLimit @1 :Float32;
  speedLimitAheadValid @2 :Bool;
  speedLimitAhead @3 :Float32;
  speedLimitAheadDistance @4 :Float32;
  roadName @5 :Text;
}

struct ModelDataV2SP @0xa1680744031fdb2d {
  laneTurnDirection @0 :TurnDirection;
  leftLaneChangeEdgeBlock @1 :Bool;
  rightLaneChangeEdgeBlock @2 :Bool;
  # Default-off instance references only; no compiled-byte/source authentication.
  modelAssetReference @3 :ModelAssetReferenceV1;
  # Separate opt-in version; legacy @3 is never reinterpreted or dual-published.
  modelHeldDelayBinaryReference @4 :ModelHeldDelayBinaryReferenceV1;

  struct ModelHeldDelayBinaryReferenceV1 {
    # Fixed976B ELMW payload; full version decoder required. No per-field addressability.
    # Matching metadata fences do not admit ownership/source/held-lineage proof.
    codecVersion @0 :UInt16;
    payload @1 :Data;
  }

  struct ModelAssetReferenceV1 {
    present @0 :Bool; processEpoch @1 :Data; runner @2 :Runner;
    activeInstanceId @3 :UInt16; attemptedInstanceId @4 :UInt16;
    attemptSequence @5 :UInt64; publicationAttemptId @6 :UInt64;
    appendNs @7 :UInt64; appendKnown @8 :Bool;
    modelEventNs @9 :UInt64; drivingEventNs @10 :UInt64; spEventNs @11 :UInt64;
    modelEventValid @12 :Bool; drivingEventValid @13 :Bool; spEventValid @14 :Bool;
    modelFrameId @15 :UInt32; modelFrameIdExtra @16 :UInt32; modelTimestampEof @17 :UInt64;
    referenceAttempts @18 :UInt64; dropped @19 :UInt64; errors @20 :UInt64; epochLoss @21 :Bool;
    streamLayout @22 :Unknown; consumedDigest @23 :Unknown; fullAssetDigest @24 :Unknown;
    fullAssetKnown @25 :Bool; sourceAuthenticated @26 :Bool; policyComplete @27 :Bool; predictiveEligible @28 :Bool;
    bootIdentityKnown @29 :Bool; clockDomainsVerified @30 :Bool; exposureOriginVerified @31 :Bool;
    references @32 :List(Reference); referenceCaptureComplete @33 :Bool;
    # Copied from drivingModelData itself; no driving EOF or borrowed model-frame identity.
    drivingFrameId @34 :UInt32; drivingFrameIdExtra @35 :UInt32;
    # Default-off prepared-input references, never graph/runtime proof.
    modelDelayReference @36 :ModelDelayReferenceV1;

    struct ModelDelayReferenceV1 {
      present @0 :Bool;
      preparedStatus @1 :PreparedStatus;  # Prepared for run; NOT run-enter or passed-to-graph.
      preparedAttemptId @2 :UInt64; preparedInstanceId @3 :UInt16;
      preparedNs @4 :UInt64; preparedClockStatus @5 :ClockStatus; preparationComplete @6 :Bool;
      # Original received FrameMeta only, distinct from model/driving output fields.
      mainFrameId @7 :UInt32; extraFrameId @8 :UInt32;
      mainTimestampSof @9 :UInt64; mainTimestampEof @10 :UInt64;
      extraTimestampSof @11 :UInt64; extraTimestampEof @12 :UInt64;
      frameMetadataKnown @13 :Bool; singleCameraAlias @14 :Bool; singleCameraAliasKnown @15 :Bool;
      smFrame @16 :UInt64;
      # Raw publication use is not capture-asof and does not prove CP-vs-learner origin.
      publishedDelay @17 :Float64; publishedDelayKnown @18 :Bool;
      publishedEventNs @19 :UInt64; publishedRecvFrame @20 :UInt64;
      publishedSeen @21 :Bool; publishedUpdated @22 :Bool; publishedValid @23 :Bool;
      publishedAlive @24 :Bool; publishedReferenceKnown @25 :Bool;
      cachedValue @26 :Float64; cachedValueKind @27 :ValueKind;
      cachedReadId @28 :UInt64; cachedReadIdKnown @29 :Bool;
      cachedWriteNs @30 :UInt64; cachedWriteClockStatus @31 :ClockStatus; cachedOrigin @32 :CacheOrigin;
      effectiveLatOffset @33 :Float64; effectiveLatOffsetKnown @34 :Bool;
      publishedLatOffset @35 :Float64; publishedLatOffsetKnown @36 :Bool;
      # Private immutable constructor snapshot survives ordinary pending-read retirement.
      constructorCacheValue @37 :Float64; constructorValueKind @38 :ValueKind;
      constructorReadId @39 :UInt64; constructorReadIdKnown @40 :Bool;
      constructorWriteNs @41 :UInt64; constructorWriteClockStatus @42 :ClockStatus;
      cpLongitudinalActuatorDelay @43 :Float64; cpLongitudinalActuatorDelayKnown @44 :Bool; cpOrigin @45 :CpOrigin;
      longDelay @46 :Float64; longDelayKnown @47 :Bool;
      initialLongSmoothing @48 :Float64; initialLongSmoothingKnown @49 :Bool;
      initialLongOwnerId @50 :UInt16; initialLongOwnerKnown @51 :Bool;
      currentLatSmoothing @52 :Float64; currentLongSmoothing @53 :Float64;
      currentLatSmoothingKnown @54 :Bool; currentLongSmoothingKnown @55 :Bool;
      currentSmoothingOrigin @56 :SmoothingOrigin; postprocessSmoothingBranch @57 :UseBranch;
      # Actual Python offsets separately retain their Float64 representation/use.
      pythonLatActionT @58 :Float64; pythonLongActionT @59 :Float64; pythonActionOffsetsKnown @60 :Bool;
      frameDelay @61 :Float64; actionDelay @62 :Float64; heuristicOffsetsKnown @63 :Bool;
      # EXACT original ndarray(float32, shape=(2,)); absent != zero-vector.
      preparedActionLat @64 :Float32; preparedActionLong @65 :Float32; preparedActionStatus @66 :VectorStatus;
      preparedLateralVEgo @67 :Float32; preparedLateralDelay @68 :Float32; preparedLateralStatus @69 :VectorStatus;
      preparationVEgo @70 :Float64; preparationVEgoKnown @71 :Bool;
      postprocessLatActionT @72 :Float64; postprocessLongActionT @73 :Float64;
      postprocessActionOffsetsKnown @74 :Bool;
      readCaptureComplete @75 :Bool; reads @76 :List(PackedReadV1);
      preparedAttemptCount @77 :UInt64; preparationUnavailableCount @78 :UInt64;
      outputNoneObservedCount @79 :UInt64; failedFinalSpSendCount @80 :UInt64;
      discardedBeforeFinalSpReturnCount @81 :UInt64; lastDiscardedAttemptId @82 :UInt64;
      lastDiscardedInstanceId @83 :UInt16; lastDiscardReason @84 :DiscardReason;
      counterOverflowKnown @85 :Bool; epochLoss @86 :Bool;
      runCallKnown @87 :Bool; graphCopyKnown @88 :Bool; graphConsumptionKnown @89 :Bool;
      rawPresenceKnown @90 :Bool; defaultOriginKnown @91 :Bool;
      publishedCpVersusLearnerOriginKnown @92 :Bool;
      sourceAuthenticated @93 :Bool; fullAssetKnown @94 :Bool; policyComplete @95 :Bool; predictiveEligible @96 :Bool;
      clockDomainsVerified @97 :Bool; captureOriginVerified @98 :Bool; durableAckKnown @99 :Bool;

      enum PreparedStatus { unavailable @0; preparedForRun @1; }
      enum ClockStatus { unavailable @0; known @1; tie @2; regression @3; invalid @4; }
      enum ValueKind { unknown @0; knownBool @1; knownFloat @2; knownNone @3; }
      enum CacheOrigin { unknown @0; constructorCache @1; publishedHelper @2; cachedHelper @3; }
      enum CpOrigin { unknown @0; parsed @1; demo @2; }
      enum SmoothingOrigin { unknown @0; stockLiteral @1; declaredProcessing @2; }
      enum UseBranch { unknown @0; used @1; bypassed @2; }
      enum VectorStatus { unavailable @0; prepared @1; absentOriginalBranch @2; }
      enum DiscardReason { unknown @0; outputNone @1; finalSendRaised @2; replacedUnreturned @3; }

      # packed_read_status_v1: exact19-field semantic recovery.
      # statusBits0..24 used;25..31 reserved/rejected. No deployed compatibility.
      struct PackedReadV1 {
        readId @0 :UInt64; instanceId @1 :UInt16;
        beginNs @2 :UInt64; endNs @3 :UInt64; floatValue @4 :Float64; statusBits @5 :UInt32;
      }

      # Frozen original semantic schema retained as a reference, never the packed list target.
      struct Read {
        readId @0 :UInt64; instanceId @1 :UInt16; site @2 :Site; key @3 :Key;
        method @4 :Method; registryType @5 :RegistryType; defaultRequested @6 :Bool;
        rawPresence @7 :Unknown; defaultOrigin @8 :Unknown;
        beginNs @9 :UInt64; endNs @10 :UInt64;
        beginStatus @11 :ClockStatus; endStatus @12 :ClockStatus; outcome @13 :Outcome;
        valueKind @14 :ValueKind; boolValue @15 :Bool; floatValue @16 :Float64;
        # Later conversion is not re-labelled as a getter exception or successful write.
        castOutcome @17 :CastOutcome; valid @18 :Bool;
        enum Site { unknown @0; delayInitial @1; lagdToggle @2; delayCache @3; }
        enum Key { unknown @0; lagdValueCache @1; lagdToggle @2; }
        enum Method { unknown @0; get @1; getBool @2; }
        enum RegistryType { unknown @0; floating @1; boolean @2; }
        enum Outcome { unavailable @0; returned @1; raised @2; }
        enum CastOutcome { notObserved @0; succeeded @1; raised @2; notApplied @3; }
      }
    }
    enum Runner { stock @0; tinygrad @1; }
    enum Unknown { unknown @0; }
    struct Reference {
      sequence @0 :UInt64; kind @1 :Kind; fromInstanceId @2 :UInt16; toInstanceId @3 :UInt16;
      failedAttemptId @4 :UInt64; observationNs @5 :UInt64; clockKnown @6 :Bool; valid @7 :Bool;
      # Begin is after ModelStateBase init; end is constructor completion, NOT warmup/handoff.
      selectionBeginNs @8 :UInt64; constructorEndNs @9 :UInt64; constructorClockKnown @10 :Bool;
      origin @11 :Origin; chestnut @12 :Bool; roleKnown @13 :Bool;
      # Actual processing metadata from selected bundle; not compiled-asset generation.
      declaredGeneration @14 :UInt32; declaredGenerationKnown @15 :Bool;
      camWidth @16 :UInt16; camHeight @17 :UInt16; geometryKnown @18 :Bool;
      mode @19 :Mode; device @20 :Device;
      latSmoothSeconds @21 :Float64; latSmoothKnown @22 :Bool; longSmoothSeconds @23 :Float64; longSmoothKnown @24 :Bool;
      enum Kind { load @0; initialSelection @1; runtimeFallback @2; bigUnavailable @3; }
      enum Origin { unknown @0; envOverride @1; bundleLookup @2; stockBuiltin @3; }
      enum Mode { unknown @0; stock @1; supercombo @2; split @3; multiPolicy @4; }
      enum Device { unknown @0; cpu @1; qcom @2; amd @3; }
    }
  }

  enum TurnDirection {
    none @0;
    turnLeft @1;
    turnRight @2;
  }
}

struct ControlsObservationV1 @0xcb9fd56c7057593a {
  observation @0 :ControllerObservationRecordV1;
}

struct CardObservationV1 @0xc2243c65e0340384 {
  observation @0 :ControllerObservationRecordV1;
}

# First-stage source clocks only. No policy/source/plant admission or TX delivery claim.
struct ControllerObservationRecordV1 {
  version @0 :UInt16;
  producer @1 :Producer;
  processEpoch @2 :Data;  # random 16-byte process identifier, not boot/route identity
  sequence @3 :UInt64;  # attempted capture sequence, including dropped captures
  loop @4 :UInt64;
  stage @5 :Stage;
  service @6 :Service;
  captureNs @7 :UInt64;  # actual producer observation clock; Event time is worker construction
  beginNs @8 :UInt64;
  applyClockNs @9 :UInt64;  # original CI.apply argument, not inferred from publication time
  applyId @10 :UInt64;
  eventNs @11 :UInt64;
  eventPresent @12 :Bool;
  eventValid @13 :Bool;
  reason @14 :Reason;
  captureComplete @15 :Bool;
  policyComplete @16 :Bool;
  sourceAuthenticated @17 :Bool;
  predictiveEligible @18 :Bool;
  rawBoolPresence @19 :Availability;
  txEchoCorrelation @20 :Correlation;
  calibratedRollUncertainty @21 :Availability;
  bootIdentityKnown @22 :Bool;
  semanticDigestAvailable @23 :Bool;
  callbackTxObserved @24 :Bool;
  dropped @25 :UInt64;
  errors @26 :UInt64;
  references @27 :List(MessageReference);
  tx @28 :List(TxFrame);  # ordered CI.apply output, not proven panda transmission/delivery
  faults @29 :List(Fault);
  startupReference @30 :StartupReference;
  policyObservation @31 :PolicyObservation;
  cardAsyncPolicyObservation @32 :CardAsyncPolicyObservation;

  enum Producer { controls @0; card @1; }
  enum Availability { unknown @0; }
  enum Correlation { unsupported @0; }
  enum Stage {
    loopStart @0; updateBegin @1; updateEnd @2; consumeEnd @3;
    computeBegin @4; computeEnd @5; paramsRefreshBegin @6; paramsRefreshEnd @7;
    publishBegin @8; publishEnd @9; publishFailed @10; applyBegin @11; applyEnd @12;
    applySkipped @13; initializeBegin @14; initializeEnd @15; startupBegin @16; startupEnd @17; policy @18; cardAsyncPolicy @19;
  }
  enum Service {
    none @0; lateralDelay @1; vehicleParameters @2; lateralTorqueParameters @3;
    modelV2 @4; selfdriveState @5; extrinsicsCalibration @6; deviceMotion @7;
    longitudinalPlan @8; lateralManeuverPlan @9; carState @10; carOutput @11;
    driverMonitoringState @12; onroadEvents @13; driverAssistance @14; radarState @15;
    selfdriveStateSP @16; pandaStates @17; carControl @18; carControlSP @19;
    longitudinalPlanSP @20; carParams @21; carParamsSP @22; carStateSP @23;
    controlsState @24; sendcan @25; radarTracks @26; liveLocationKalman @27;
  }
  enum Reason { none @0; passive @1; notInitialized @2; ccNotAlive @3; }
  enum Status { notSubscribed @0; unseen @1; observed @2; unknown @3; }
  enum IdentityKind { unavailable @0; timestampReference @1; }
  enum Fault {
    unknownField @0; nonfinite @1; sourceInvalid @2; futureReference @3;
    timestampTie @4; timestampRegression @5; clockTie @6; clockRegression @7;
    txOversize @8; eventOversize @9; publisherFailed @10; unseenSource @11;
    priorLoss @12; identityUnchecked @13;
  }
  struct MessageReference {
    service @0 :Service; status @1 :Status; identityKind @2 :IdentityKind;
    eventNs @3 :UInt64; recvFrame @4 :UInt64; recvTime @5 :Float64;
    seen @6 :Bool; updated @7 :Bool; valid @8 :Bool; alive @9 :Bool; freqOk @10 :Bool;
  }
  struct TxFrame { address @0 :UInt32; src @1 :UInt8; payloadLength @2 :UInt8; payloadCaptured @3 :Bool; }
  # Offered-use values do NOT identify their async read, an atomic config or current CP/SP.
  struct CardAsyncPolicyObservation {
    present @0 :Bool; kind @1 :Kind; site @2 :Site; key @3 :Key; method @4 :Method; registryType @5 :Scalar;
    consumerId @6 :UInt64; readSequence @7 :UInt64; cycleId @8 :UInt64;
    beginNs @9 :UInt64; endNs @10 :UInt64; assignmentNs @11 :UInt64;
    beginStatus @12 :Clock; endStatus @13 :Clock; assignmentStatus @14 :Clock;
    getterOutcome @15 :Outcome; operationRaised @16 :Bool;
    returnedKnown @17 :Bool; returnedValue @18 :Bool; effectiveKnown @19 :Bool; effectiveValue @20 :Bool;
    cpConditional @21 :Bool; cpKnown @22 :Bool; cpValue @23 :Bool; offeredKnown @24 :Bool; offeredValue @25 :Bool;
    originReadKnown @26 :Bool; originReadId @27 :UInt64;
    attempted @28 :UInt64; returned @29 :UInt64; raised @30 :UInt64; queued @31 :UInt64; dropped @32 :UInt64; errors @33 :UInt64;
    loss @34 :Bool; valid @35 :Bool; complete @36 :Bool;
    rawPresence @37 :Availability; defaultUsed @38 :Availability;
    policyComplete @39 :Bool; sourceAuthenticated @40 :Bool; predictiveEligible @41 :Bool;
    enum Kind { read @0; directSubsetComplete @1; use @2; }
    enum Site { none @0; dynamicInitial @1; metricInitial @2; experimentalInitial @3;
                metricRefresh @4; experimentalRefresh @5; dynamicRefresh @6;
                metricSpeedLimitUse @7; metricCruiseUse @8; experimentalInitializeUse @9; dynamicInitializeUse @10; }
    enum Key { none @0; isMetric @1; experimentalMode @2; dynamicExperimentalControl @3; }
    enum Method { none @0; getBool @1; }
    enum Scalar { bool @0; }
    enum Clock { unavailable @0; known @1; tie @2; regression @3; }
    enum Outcome { unknown @0; returned @1; raised @2; }
    enum Availability { unknown @0; }
  }
  # Returned typed values and separate scalar writes only. Raw/default origin remains unknown.
  struct PolicyObservation {
    present @0 :Bool; kind @1 :Kind; group @2 :Group; groupId @3 :UInt32;
    consumerId @4 :UInt32; parentId @5 :UInt32; consumerKind @6 :Consumer;
    site @7 :Site; key @8 :Key; method @9 :Method; registryType @10 :Scalar;
    defaultRequested @11 :Bool; rawPresence @12 :Availability; defaultUsed @13 :Availability;
    readSequence @14 :UInt32; scalarKind @15 :Scalar; boolValue @16 :Bool; intValue @17 :Int64; floatValue @18 :Float64;
    beginNs @19 :UInt64; endNs @20 :UInt64; beginKnown @21 :Bool; valid @22 :Bool;
    writeSource @23 :WriteSource; factor @24 :Float64; offset @25 :Float64; friction @26 :Float64;
    sourceNs @27 :UInt64; sourceFrame @28 :UInt64;
    selectedId @29 :UInt32; selectedKnown @30 :Bool; selectedClass @31 :SelectedClass; selectionReason @32 :SelectionReason;
    attempts @33 :UInt32; returned @34 :UInt32; raised @35 :UInt32; accepted @36 :UInt32; dropped @37 :UInt32; complete @38 :Bool;
    enum Kind { consumerCreated @0; readReturned @1; readRaised @2; controllerSelected @3; torqueParametersWritten @4;
                delayCacheWritten @5; groupComplete @6; }
    enum Group { controlsInit @0; stockConstructor @1; v0Constructor @2; selection @3; overrideRefresh @4;
                 controlsRefresh @5; learnerWrite @6; }
    enum Consumer { controls @0; blinker @1; stockTorque @2; stockExtension @3; v0Torque @4; v0Extension @5; }
    enum Method { none @0; get @1; getBool @2; }
    enum Scalar { unknown @0; none @1; bool @2; int @3; float @4; }
    enum Availability { unknown @0; }
    enum WriteSource { none @0; constructor @1; learner @2; override @3; cachedDelay @4; publishedDelay @5; }
    enum SelectedClass { unknown @0; angle @1; curvature @2; pid @3; stockTorque @4; v0Torque @5; }
    enum SelectionReason { observed @0; nonTorqueNotInstrumented @1; notInstrumented @2; }
    enum Key {
      none @0; enforceTorqueControl @1; torqueControlTune @2; lateralJerkTorqueController @3; neuralNetworkLateralControl @4;
      torqueParamsOverrideEnabled @5; torqueParamsOverrideLatAccelFactor @6; torqueParamsOverrideFriction @7;
      lagdToggle @8; lagdValueCache @9; blinkerPauseLateralControl @10; isMetric @11; blinkerMinLateralControlSpeed @12;
      blinkerLateralReengageDelay @13; customTorqueParams @14; liveTorqueParamsToggle @15; liveTorqueParamsRelaxedToggle @16;
    }
    enum Site {
      none @0; selectionEnforce @1; selectionTune @2; jerkEnable @3; nnEnable @4; overrideEnforce @5; overrideInitial @6;
      overrideRefresh @7; overrideFactor @8; overrideFriction @9; delayInitial @10; lagdToggle @11; delayCache @12;
      blinkerInitial @13; metricInitial @14; blinkerRefresh @15; metricRefresh @16; blinkerMinSpeed @17; blinkerReengage @18;
    }
  }
  # Exact startup buffer only: NOT current CP, source authentication, semantic or paired-static identity.
  struct StartupReference {
    present @0 :Bool;
    referenceId @1 :UInt64;
    source @2 :Source;
    decoder @3 :Decoder;
    wire @4 :Wire;
    beginNs @5 :UInt64;
    endNs @6 :UInt64;
    beginStatus @7 :Availability;
    endStatus @8 :Availability;
    outcome @9 :Outcome;
    status @10 :Status;
    byteCount @11 :UInt64;
    byteCountKnown @12 :Bool;
    sha256 @13 :Data;  # empty or32-byte digest; NEVER original CP/SP bytes
    digestReadyNs @14 :UInt64;
    enum Source { paramsCarParams @0; paramsCarParamsSP @1; serializedCarParams @2; serializedCarParamsSP @3; }
    enum Decoder { carParams @0; carParamsSP @1; }
    enum Wire { capnpUnpackedMessage @0; }
    enum Availability { unknown @0; available @1; invalid @2; }
    enum Outcome { pending @0; complete @1; readFailed @2; parseFailed @3; serializationFailed @4; }
    enum Status {
      pending @0; ready @1; nonbytes @2; oversize @3; readFailed @4; parseFailed @5; serializationFailed @6;
      clockInvalid @7; digestFailed @8; publisherFailed @9; priorLoss @10;
    }
  }
}

struct CustomReserved12 @0x9ccdc8676701b412 {
}

struct CustomReserved13 @0xcd96dafb67a082d0 {
}

struct CustomReserved14 @0xb057204d7deadf3f {
}

struct CustomReserved15 @0xbd443b539493bc68 {
}

struct CustomReserved16 @0xfc6241ed8877b611 {
}

struct CustomReserved17 @0xa30662f84033036c {
}

struct CustomReserved18 @0xc86a3d38d13eb3ef {
}

struct CustomReserved19 @0xa4f1eb3323f5f582 {
}
