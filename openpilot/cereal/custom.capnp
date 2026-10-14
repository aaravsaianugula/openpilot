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

  enum Producer { controls @0; card @1; }
  enum Availability { unknown @0; }
  enum Correlation { unsupported @0; }
  enum Stage {
    loopStart @0; updateBegin @1; updateEnd @2; consumeEnd @3;
    computeBegin @4; computeEnd @5; paramsRefreshBegin @6; paramsRefreshEnd @7;
    publishBegin @8; publishEnd @9; publishFailed @10; applyBegin @11; applyEnd @12;
    applySkipped @13; initializeBegin @14; initializeEnd @15; startupBegin @16; startupEnd @17; policy @18;
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
