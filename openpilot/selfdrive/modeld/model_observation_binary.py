"""Default-off fixed model observation representation and observed SM fences.

Matching metadata NEVER proves completed-update ownership. Supplied typed bit
images do not authenticate earlier native/NumPy provenance. No files/maps/hashes,
Params/model arrays/publishers/workers are read. Enabled overhead is unmeasured.
"""
import math
import struct
from types import MappingProxyType, SimpleNamespace

VERSION, PAYLOAD_BYTES, SCALAR_BITS = 1, 976, 7744
MAGIC = b"ELMW"
U64 = 2**64-1

ENUMS = { 'Runner': {'stock': 0, 'tinygrad': 1},
  'Unknown': {'unknown': 0},
  'PreparedStatus': {'unavailable': 0, 'preparedForRun': 1},
  'ClockStatus': {'unavailable': 0, 'known': 1, 'tie': 2, 'regression': 3, 'invalid': 4},
  'ValueKind': {'unknown': 0, 'knownBool': 1, 'knownFloat': 2, 'knownNone': 3},
  'CacheOrigin': {'unknown': 0, 'constructorCache': 1, 'publishedHelper': 2, 'cachedHelper': 3},
  'CpOrigin': {'unknown': 0, 'parsed': 1, 'demo': 2},
  'SmoothingOrigin': {'unknown': 0, 'stockLiteral': 1, 'declaredProcessing': 2},
  'UseBranch': {'unknown': 0, 'used': 1, 'bypassed': 2},
  'VectorStatus': {'unavailable': 0, 'prepared': 1, 'absentOriginalBranch': 2},
  'DiscardReason': {'unknown': 0, 'outputNone': 1, 'finalSendRaised': 2, 'replacedUnreturned': 3},
  'Kind': {'load': 0, 'initialSelection': 1, 'runtimeFallback': 2, 'bigUnavailable': 3},
  'Origin': {'unknown': 0, 'envOverride': 1, 'bundleLookup': 2, 'stockBuiltin': 3},
  'Mode': {'unknown': 0, 'stock': 1, 'supercombo': 2, 'split': 3, 'multiPolicy': 4},
  'Device': {'unknown': 0, 'cpu': 1, 'qcom': 2, 'amd': 3},
  'Site': {'unknown': 0, 'delayInitial': 1, 'lagdToggle': 2, 'delayCache': 3},
  'Key': {'unknown': 0, 'lagdValueCache': 1, 'lagdToggle': 2},
  'Method': {'unknown': 0, 'get': 1, 'getBool': 2},
  'RegistryType': {'unknown': 0, 'floating': 1, 'boolean': 2},
  'Outcome': {'unavailable': 0, 'returned': 1, 'raised': 2},
  'CastOutcome': {'notObserved': 0, 'succeeded': 1, 'raised': 2, 'notApplied': 3},
  'AnchorAssociation': {'unknown': 0, 'observed': 1, 'unsupported': 2, 'lost': 3, 'invalid': 4}}

PARENT_SPEC = (
  ('present', 'Bool', 1),
  ('processEpoch', 'Data', 128),
  ('runner', 'Runner', 1),
  ('activeInstanceId', 'UInt16', 16),
  ('attemptedInstanceId', 'UInt16', 16),
  ('attemptSequence', 'UInt64', 64),
  ('publicationAttemptId', 'UInt64', 64),
  ('appendNs', 'UInt64', 64),
  ('appendKnown', 'Bool', 1),
  ('modelEventNs', 'UInt64', 64),
  ('drivingEventNs', 'UInt64', 64),
  ('spEventNs', 'UInt64', 64),
  ('modelEventValid', 'Bool', 1),
  ('drivingEventValid', 'Bool', 1),
  ('spEventValid', 'Bool', 1),
  ('modelFrameId', 'UInt32', 32),
  ('modelFrameIdExtra', 'UInt32', 32),
  ('modelTimestampEof', 'UInt64', 64),
  ('referenceAttempts', 'UInt64', 64),
  ('dropped', 'UInt64', 64),
  ('errors', 'UInt64', 64),
  ('epochLoss', 'Bool', 1),
  ('streamLayout', 'Unknown', 1),
  ('consumedDigest', 'Unknown', 1),
  ('fullAssetDigest', 'Unknown', 1),
  ('fullAssetKnown', 'Bool', 1),
  ('sourceAuthenticated', 'Bool', 1),
  ('policyComplete', 'Bool', 1),
  ('predictiveEligible', 'Bool', 1),
  ('bootIdentityKnown', 'Bool', 1),
  ('clockDomainsVerified', 'Bool', 1),
  ('exposureOriginVerified', 'Bool', 1),
  ('referenceCaptureComplete', 'Bool', 1),
  ('drivingFrameId', 'UInt32', 32),
  ('drivingFrameIdExtra', 'UInt32', 32),
)

SNAPSHOT_SPEC = (
  ('present', 'Bool', 1),
  ('preparedStatus', 'PreparedStatus', 1),
  ('preparedAttemptId', 'UInt64', 64),
  ('preparedInstanceId', 'UInt16', 16),
  ('preparedNs', 'UInt64', 64),
  ('preparedClockStatus', 'ClockStatus', 3),
  ('preparationComplete', 'Bool', 1),
  ('mainFrameId', 'UInt32', 32),
  ('extraFrameId', 'UInt32', 32),
  ('mainTimestampSof', 'UInt64', 64),
  ('mainTimestampEof', 'UInt64', 64),
  ('extraTimestampSof', 'UInt64', 64),
  ('extraTimestampEof', 'UInt64', 64),
  ('frameMetadataKnown', 'Bool', 1),
  ('singleCameraAlias', 'Bool', 1),
  ('singleCameraAliasKnown', 'Bool', 1),
  ('smFrame', 'UInt64', 64),
  ('publishedDelay', 'Float64', 64),
  ('publishedDelayKnown', 'Bool', 1),
  ('publishedEventNs', 'UInt64', 64),
  ('publishedRecvFrame', 'UInt64', 64),
  ('publishedSeen', 'Bool', 1),
  ('publishedUpdated', 'Bool', 1),
  ('publishedValid', 'Bool', 1),
  ('publishedAlive', 'Bool', 1),
  ('publishedReferenceKnown', 'Bool', 1),
  ('cachedValue', 'Float64', 64),
  ('cachedValueKind', 'ValueKind', 2),
  ('cachedReadId', 'UInt64', 64),
  ('cachedReadIdKnown', 'Bool', 1),
  ('cachedWriteNs', 'UInt64', 64),
  ('cachedWriteClockStatus', 'ClockStatus', 3),
  ('cachedOrigin', 'CacheOrigin', 2),
  ('effectiveLatOffset', 'Float64', 64),
  ('effectiveLatOffsetKnown', 'Bool', 1),
  ('publishedLatOffset', 'Float64', 64),
  ('publishedLatOffsetKnown', 'Bool', 1),
  ('constructorCacheValue', 'Float64', 64),
  ('constructorValueKind', 'ValueKind', 2),
  ('constructorReadId', 'UInt64', 64),
  ('constructorReadIdKnown', 'Bool', 1),
  ('constructorWriteNs', 'UInt64', 64),
  ('constructorWriteClockStatus', 'ClockStatus', 3),
  ('cpLongitudinalActuatorDelay', 'Float64', 64),
  ('cpLongitudinalActuatorDelayKnown', 'Bool', 1),
  ('cpOrigin', 'CpOrigin', 2),
  ('longDelay', 'Float64', 64),
  ('longDelayKnown', 'Bool', 1),
  ('initialLongSmoothing', 'Float64', 64),
  ('initialLongSmoothingKnown', 'Bool', 1),
  ('initialLongOwnerId', 'UInt16', 16),
  ('initialLongOwnerKnown', 'Bool', 1),
  ('currentLatSmoothing', 'Float64', 64),
  ('currentLongSmoothing', 'Float64', 64),
  ('currentLatSmoothingKnown', 'Bool', 1),
  ('currentLongSmoothingKnown', 'Bool', 1),
  ('currentSmoothingOrigin', 'SmoothingOrigin', 2),
  ('postprocessSmoothingBranch', 'UseBranch', 2),
  ('pythonLatActionT', 'Float64', 64),
  ('pythonLongActionT', 'Float64', 64),
  ('pythonActionOffsetsKnown', 'Bool', 1),
  ('frameDelay', 'Float64', 64),
  ('actionDelay', 'Float64', 64),
  ('heuristicOffsetsKnown', 'Bool', 1),
  ('preparedActionLat', 'Float32', 32),
  ('preparedActionLong', 'Float32', 32),
  ('preparedActionStatus', 'VectorStatus', 2),
  ('preparedLateralVEgo', 'Float32', 32),
  ('preparedLateralDelay', 'Float32', 32),
  ('preparedLateralStatus', 'VectorStatus', 2),
  ('preparationVEgo', 'Float64', 64),
  ('preparationVEgoKnown', 'Bool', 1),
  ('postprocessLatActionT', 'Float64', 64),
  ('postprocessLongActionT', 'Float64', 64),
  ('postprocessActionOffsetsKnown', 'Bool', 1),
  ('readCaptureComplete', 'Bool', 1),
  ('preparedAttemptCount', 'UInt64', 64),
  ('preparationUnavailableCount', 'UInt64', 64),
  ('outputNoneObservedCount', 'UInt64', 64),
  ('failedFinalSpSendCount', 'UInt64', 64),
  ('discardedBeforeFinalSpReturnCount', 'UInt64', 64),
  ('lastDiscardedAttemptId', 'UInt64', 64),
  ('lastDiscardedInstanceId', 'UInt16', 16),
  ('lastDiscardReason', 'DiscardReason', 2),
  ('counterOverflowKnown', 'Bool', 1),
  ('epochLoss', 'Bool', 1),
  ('runCallKnown', 'Bool', 1),
  ('graphCopyKnown', 'Bool', 1),
  ('graphConsumptionKnown', 'Bool', 1),
  ('rawPresenceKnown', 'Bool', 1),
  ('defaultOriginKnown', 'Bool', 1),
  ('publishedCpVersusLearnerOriginKnown', 'Bool', 1),
  ('sourceAuthenticated', 'Bool', 1),
  ('fullAssetKnown', 'Bool', 1),
  ('policyComplete', 'Bool', 1),
  ('predictiveEligible', 'Bool', 1),
  ('clockDomainsVerified', 'Bool', 1),
  ('captureOriginVerified', 'Bool', 1),
  ('durableAckKnown', 'Bool', 1),
)

ASSET_SPEC = (
  ('sequence', 'UInt64', 64),
  ('kind', 'Kind', 2),
  ('fromInstanceId', 'UInt16', 16),
  ('toInstanceId', 'UInt16', 16),
  ('failedAttemptId', 'UInt64', 64),
  ('observationNs', 'UInt64', 64),
  ('clockKnown', 'Bool', 1),
  ('valid', 'Bool', 1),
  ('selectionBeginNs', 'UInt64', 64),
  ('constructorEndNs', 'UInt64', 64),
  ('constructorClockKnown', 'Bool', 1),
  ('origin', 'Origin', 2),
  ('chestnut', 'Bool', 1),
  ('roleKnown', 'Bool', 1),
  ('declaredGeneration', 'UInt32', 32),
  ('declaredGenerationKnown', 'Bool', 1),
  ('camWidth', 'UInt16', 16),
  ('camHeight', 'UInt16', 16),
  ('geometryKnown', 'Bool', 1),
  ('mode', 'Mode', 3),
  ('device', 'Device', 2),
  ('latSmoothSeconds', 'Float64', 64),
  ('latSmoothKnown', 'Bool', 1),
  ('longSmoothSeconds', 'Float64', 64),
  ('longSmoothKnown', 'Bool', 1),
)

READ_SPEC = (
  ('readId', 'UInt64', 64),
  ('instanceId', 'UInt16', 16),
  ('site', 'Site', 2),
  ('key', 'Key', 2),
  ('method', 'Method', 2),
  ('registryType', 'RegistryType', 2),
  ('defaultRequested', 'Bool', 1),
  ('rawPresence', 'Unknown', 1),
  ('defaultOrigin', 'Unknown', 1),
  ('beginNs', 'UInt64', 64),
  ('endNs', 'UInt64', 64),
  ('beginStatus', 'ClockStatus', 3),
  ('endStatus', 'ClockStatus', 3),
  ('outcome', 'Outcome', 2),
  ('valueKind', 'ValueKind', 2),
  ('boolValue', 'Bool', 1),
  ('floatValue', 'Float64', 64),
  ('castOutcome', 'CastOutcome', 2),
  ('valid', 'Bool', 1),
)

ANCHOR_SPEC = (
  ('present', 'Bool', 1),
  ('processEpoch', 'Data', 128),
  ('instanceId', 'UInt16', 16),
  ('cacheReadId', 'UInt64', 64),
  ('associatedReadKnown', 'Bool', 1),
  ('cacheWriteNs', 'UInt64', 64),
  ('cacheWriteClockStatus', 'ClockStatus', 3),
  ('cacheWriteKnown', 'Bool', 1),
  ('sourceEventNs', 'UInt64', 64),
  ('sourceRecvFrame', 'UInt64', 64),
  ('sourceSmFrame', 'UInt64', 64),
  ('sourceSeen', 'Bool', 1),
  ('sourceUpdated', 'Bool', 1),
  ('sourceValid', 'Bool', 1),
  ('sourceAlive', 'Bool', 1),
  ('sourceReferenceKnown', 'Bool', 1),
  ('argumentValue', 'Float64', 64),
  ('argumentKind', 'ValueKind', 2),
  ('argumentCaptured', 'Bool', 1),
  ('refreshTokenId', 'UInt64', 64),
  ('refreshBeginNs', 'UInt64', 64),
  ('refreshEndNs', 'UInt64', 64),
  ('refreshBeginStatus', 'ClockStatus', 3),
  ('refreshEndStatus', 'ClockStatus', 3),
  ('branch', 'CacheOrigin', 2),
  ('associationStatus', 'AnchorAssociation', 3),
  ('valid', 'Bool', 1),
  ('sourceAuthenticated', 'Bool', 1),
  ('predictiveEligible', 'Bool', 1),
  ('policyComplete', 'Bool', 1),
)

SHAPE_FIELDS = ( {'name': 'assetCount', 'lsb': 0, 'width_bits': 3, 'allowed_codes': [0, 1, 2, 3, 4]},
  {'name': 'nativeReadCount', 'lsb': 3, 'width_bits': 3, 'allowed_codes': [0, 1, 2, 3, 4]},
  {'name': 'delayAllocated', 'lsb': 6, 'width_bits': 1, 'allowed_codes': [0, 1]},
  {'name': 'referencesAllocated', 'lsb': 7, 'width_bits': 1, 'allowed_codes': [0, 1]},
  {'name': 'readsAllocated', 'lsb': 8, 'width_bits': 1, 'allowed_codes': [0, 1]},
  { 'name': 'parentEpochByteCount',
    'lsb': 9,
    'width_bits': 5,
    'allowed_codes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]},
  { 'name': 'anchorEpochByteCount',
    'lsb': 14,
    'width_bits': 5,
    'allowed_codes': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]})



def _fields():
  result, offset = [], 64
  for prefix, group, spec in (("parent", "parent", PARENT_SPEC),
      ("parent.modelDelayReference", "snapshot", SNAPSHOT_SPEC),
      *((f"parent.references[{i}]", "asset", ASSET_SPEC) for i in range(4)),
      *((f"parent.modelDelayReference.reads[{i}]", "read", READ_SPEC) for i in range(4)),
      ("heldCacheSourceAnchor", "held_anchor", ANCHOR_SPEC)):
    for name, typ, width in spec:
      row = {
        'source_path': prefix+"."+name,
        'group': group,
        'source_type': typ,
        'compact_width_bits': width,
        'fixed_binary_payload_bit_offset': offset,
      }
      if typ in ENUMS:
        row["allowed_codes"] = ENUMS[typ]
      result.append(row)
      offset += width
  return tuple(result)


FIELDS = _fields()
ENUM_MAX_CHARS = {typ: max(map(len, codes)) for typ, codes in ENUMS.items()}
PARENT = tuple(r for r in FIELDS if r["group"] == "parent")
SNAPSHOT = tuple(r for r in FIELDS if r["group"] == "snapshot")
ANCHOR = tuple(r for r in FIELDS if r["group"] == "held_anchor")
ASSETS = tuple(tuple(r for r in FIELDS if r["source_path"].startswith(f"parent.references[{i}].")) for i in range(4))
READS = tuple(tuple(r for r in FIELDS if r["source_path"].startswith(f"parent.modelDelayReference.reads[{i}].")) for i in range(4))



class LayoutError(ValueError):
    """Fixed public reasons only; never format untrusted/private values."""



def _keys(value, expected):
    if type(value) is not dict:
        raise LayoutError("mapping_type")
    if len(value) != len(expected):
        raise LayoutError("mapping_shape")
    if any(type(key) is not str or len(key) > 128 for key in value):
        raise LayoutError("key_type")
    if frozenset(value) != frozenset(expected):
        raise LayoutError("mapping_shape")



def _uint(value, bits):
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise LayoutError("unsigned_type_or_range")
    return value



def _shape(shape):
    _keys(shape, tuple(r["name"] for r in SHAPE_FIELDS))
    word = 0
    for row in SHAPE_FIELDS:
        value = shape[row["name"]]
        if row["width_bits"] == 1:
            if type(value) is not bool:
                raise LayoutError("shape_bool")
            number = int(value)
        else:
            number = _uint(value, row["width_bits"])
        if number not in row["allowed_codes"]:
            raise LayoutError("shape_domain")
        word |= number << row["lsb"]
    if (
        shape["assetCount"]
        and not shape["referencesAllocated"]
        or shape["nativeReadCount"]
        and not shape["readsAllocated"]
        or shape["readsAllocated"]
        and not shape["delayAllocated"]
    ):
        raise LayoutError("shape_combination")
    return word



def _unshape(word):
    _uint(word, 32)
    if word >> 19:
        raise LayoutError("reserved_shape_bits")
    result = {}
    for row in SHAPE_FIELDS:
        number = word >> row["lsb"] & ((1 << row["width_bits"]) - 1)
        result[row["name"]] = bool(number) if row["width_bits"] == 1 else number
    _shape(result)
    return result



def _active(row, shape):
    group = row["group"]
    if group == "snapshot":
        return shape["delayAllocated"]
    if group in ("asset", "read"):
        index = int(row["source_path"].split("[")[1].split("]")[0])
        return index < shape["assetCount" if group == "asset" else "nativeReadCount"]
    return True



def _data_size(row, shape):
    return shape[
        "parentEpochByteCount" if row["group"] == "parent" else "anchorEpochByteCount"
    ]



def _code(row, value, shape):
    typ = row["source_type"]
    if typ == "Bool":
        if type(value) is not bool:
            raise LayoutError("bool_type")
        return int(value)
    if typ.startswith("UInt"):
        return _uint(value, int(typ[4:]))
    if typ.startswith("Float"):
        _keys(value, ("bits",))
        return _uint(value["bits"], int(typ[5:]))
    if typ == "Data":
        if (
            type(value) is not bytes
            or len(value) != _data_size(row, shape)
            or len(value) > 16
        ):
            raise LayoutError("epoch_type_or_length")
        return int.from_bytes(value.ljust(16, b"\0"), "little")
    if type(value) is not str:
        raise LayoutError("enum_state")
    if len(value) > ENUM_MAX_CHARS[typ]:
        raise LayoutError("enum_label_size")
    if value not in row["allowed_codes"]:
        raise LayoutError("enum_state")
    return row["allowed_codes"][value]



def _value(row, number, shape):
    typ = row["source_type"]
    if typ == "Bool":
        return bool(number)
    if typ.startswith("UInt"):
        return number
    if typ.startswith("Float"):
        return {"bits": number}
    if typ == "Data":
        data = number.to_bytes(16, "little")
        length = _data_size(row, shape)
        if any(data[length:]):
            raise LayoutError("epoch_tail")
        return data[:length]
    inverse = {code: label for label, code in row["allowed_codes"].items()}
    if number not in inverse:
        raise LayoutError("invalid_enum_code")
    return inverse[number]



def _image(packet):
    _keys(packet, ("shape", "values"))
    shape = packet["shape"]
    _shape(shape)
    values = packet["values"]
    _keys(values, tuple(r["source_path"] for r in FIELDS if _active(r, shape)))
    image = 0
    for row in FIELDS:
        if _active(row, shape):
            image |= _code(row, values[row["source_path"]], shape) << (
                row["fixed_binary_payload_bit_offset"] - 64
            )
    return image



def _recover(image, shape):
    if image >> SCALAR_BITS:
        raise LayoutError("reserved_image_bits")
    values = {}
    for row in FIELDS:
        number = image >> (row["fixed_binary_payload_bit_offset"] - 64) & (
            (1 << row["compact_width_bits"]) - 1
        )
        if _active(row, shape):
            values[row["source_path"]] = _value(row, number, shape)
        elif number:
            raise LayoutError("nonzero_absent_slot")
    return {"shape": dict(shape), "values": values}



def _nested(packet):
    shape = packet["shape"]
    values = packet["values"]

    def node(rows):
        return {
            r["source_path"].rsplit(".", 1)[1]: _value(
                r, _code(r, values[r["source_path"]], shape), shape
            )
            for r in rows
        }

    parent = node(PARENT)
    parent["references"] = (
        [node(ASSETS[i]) for i in range(shape["assetCount"])]
        if shape["referencesAllocated"]
        else None
    )
    delay = node(SNAPSHOT) if shape["delayAllocated"] else None
    if delay is not None:
        delay["reads"] = (
            [node(READS[i]) for i in range(shape["nativeReadCount"])]
            if shape["readsAllocated"]
            else None
        )
    parent["modelDelayReference"] = delay
    return {"parent": parent, "heldCacheSourceAnchor": node(ANCHOR)}



def _from_nested(wire):
    _keys(wire, ("parent", "heldCacheSourceAnchor"))
    parent = wire["parent"]
    anchor = wire["heldCacheSourceAnchor"]
    _keys(
        parent,
        tuple(r["source_path"].rsplit(".", 1)[1] for r in PARENT)
        + ("references", "modelDelayReference"),
    )
    _keys(anchor, tuple(r["source_path"].rsplit(".", 1)[1] for r in ANCHOR))
    refs = parent["references"]
    delay = parent["modelDelayReference"]
    if refs is not None and (type(refs) is not list or len(refs) > 4):
        raise LayoutError("reference_list")
    if delay is not None:
        _keys(
            delay,
            tuple(r["source_path"].rsplit(".", 1)[1] for r in SNAPSHOT) + ("reads",),
        )
    reads = delay["reads"] if delay is not None else None
    if reads is not None and (type(reads) is not list or len(reads) > 4):
        raise LayoutError("read_list")
    for data in (parent["processEpoch"], anchor["processEpoch"]):
        if type(data) is not bytes or len(data) > 16:
            raise LayoutError("epoch_type_or_length")
    shape = {
      'assetCount': len(refs) if refs is not None else 0,
      'nativeReadCount': len(reads) if reads is not None else 0,
      'delayAllocated': delay is not None,
      'referencesAllocated': refs is not None,
      'readsAllocated': reads is not None,
      'parentEpochByteCount': len(parent["processEpoch"]),
      'anchorEpochByteCount': len(anchor["processEpoch"]),
    }
    values = {}

    def take(rows, node, extra=()):
        _keys(node, tuple(r["source_path"].rsplit(".", 1)[1] for r in rows) + extra)
        for row in rows:
            values[row["source_path"]] = node[row["source_path"].rsplit(".", 1)[1]]

    take(PARENT, parent, ("references", "modelDelayReference"))
    take(ANCHOR, anchor)
    if delay is not None:
        take(SNAPSHOT, delay, ("reads",))
    if refs is not None:
        for i, node in enumerate(refs):
            take(ASSETS[i], node)
    if reads is not None:
        for i, node in enumerate(reads):
            take(READS[i], node)
    return _recover(_image({"shape": shape, "values": values}), shape)


def encode(packet):
  image = _image(packet)
  return {"codecVersion": VERSION, "payload": MAGIC+_shape(packet["shape"]).to_bytes(4, "little")+image.to_bytes(968, "little")}


def decode(wire):
  _keys(wire, ("codecVersion", "payload"))
  if _uint(wire["codecVersion"], 16) != VERSION:
    raise LayoutError("version")
  payload = wire["payload"]
  if type(payload) is not bytes or len(payload) != PAYLOAD_BYTES:
    raise LayoutError("payload_type_or_length")
  if payload[:4] != MAGIC:
    raise LayoutError("magic")
  return _recover(int.from_bytes(payload[8:], "little"), _unshape(int.from_bytes(payload[4:8], "little")))


def decode_body(body):
  """Actual SDK _has contract; outer versions must never be merged or borrowed."""
  legacy = body._has("modelAssetReference")
  binary = body._has("modelHeldDelayBinaryReference")
  if type(legacy) is not bool or type(binary) is not bool:
    raise LayoutError("outer_presence_type")
  if legacy and binary:
    raise LayoutError("outer_version_conflict")
  if not binary:
    raise LayoutError("binary_absent")
  header = body.modelHeldDelayBinaryReference
  return decode({"codecVersion": header.codecVersion, "payload": header.payload})


def _empty(row):
  typ = row["source_type"]
  if typ == "Bool":
    return False
  if typ == "Data":
    return b""
  if typ.startswith("Float"):
    return {"bits": 0}
  if typ.startswith("UInt"):
    return 0
  return next(label for label, code in row["allowed_codes"].items() if code == 0)


def empty_packet():
  # Unobserved representation only: scalar zeros are never known native defaults.
  shape = {
    'assetCount': 0,
    'nativeReadCount': 0,
    'delayAllocated': False,
    'referencesAllocated': False,
    'readsAllocated': False,
    'parentEpochByteCount': 0,
    'anchorEpochByteCount': 0,
  }
  return {"shape": shape, "values": {r["source_path"]: _empty(r) for r in FIELDS if _active(r, shape)}}


def empty_anchor():
  return {r["source_path"].rsplit(".", 1)[1]: (0. if r["source_type"].startswith("Float") else _empty(r)) for r in ANCHOR}


def _semantic_node(data, rows, extra=()):
  _keys(data, tuple(r["source_path"].rsplit(".", 1)[1] for r in rows)+extra)
  result = {}
  for row in rows:
    key, typ = row["source_path"].rsplit(".", 1)[1], row["source_type"]
    value = data[key]
    if typ.startswith("Float"):
      if type(value) is not float:
        raise LayoutError("native_float_type")
      value = {"bits": int.from_bytes(struct.pack("<f" if typ == "Float32" else "<d", value), "little")}
    size = len(value) if typ == "Data" and type(value) is bytes else 0
    _code(row, value, {"parentEpochByteCount": size, "anchorEpochByteCount": size})
    result[key] = value  # Every scalar's type/domain/privacy validated before copy.
  return result


def encode_semantic(parent, anchor):
  """Bridge only the captured Python scalar; no raw-array bit provenance claim."""
  _keys(parent, tuple(r["source_path"].rsplit(".", 1)[1] for r in PARENT)+("references", "modelDelayReference"))
  refs, delay = parent["references"], parent["modelDelayReference"]
  if refs is not None and (type(refs) is not list or len(refs) > 4):
    raise LayoutError("reference_list")
  result = _semantic_node(parent, PARENT, ("references", "modelDelayReference"))
  result["references"] = None if refs is None else [_semantic_node(row, ASSETS[i]) for i, row in enumerate(refs)]
  result["modelDelayReference"] = None
  if delay is not None:
    item = _semantic_node(delay, SNAPSHOT, ("reads",))
    reads = delay["reads"]
    if reads is not None and (type(reads) is not list or len(reads) > 4):
      raise LayoutError("read_list")
    item["reads"] = None if reads is None else [_semantic_node(row, READS[i]) for i, row in enumerate(reads)]
    result["modelDelayReference"] = item
  return encode(_from_nested({"parent": result, "heldCacheSourceAnchor": _semantic_node(anchor, ANCHOR)}))


class AssetPacket(SimpleNamespace):
  """Detached scalar builder written only by the existing bounded asset stage."""
  def init(self, name, count):
    if name != "references" or type(count) is not int or not 0 <= count <= 4:
      raise LayoutError("reference_list")
    self.references = [SimpleNamespace() for _ in range(count)]
    return self.references

  def packet(self):
    raw = vars(self)
    _semantic_node(raw, PARENT, ("references", "modelDelayReference"))
    refs, delay = raw["references"], raw["modelDelayReference"]
    if type(refs) is not list or len(refs) > 4:
      raise LayoutError("reference_list")
    for i, row in enumerate(refs):
      if type(row) is not SimpleNamespace:
        raise LayoutError("reference_type")
      _semantic_node(vars(row), ASSETS[i])
    _semantic_node(delay, SNAPSHOT, ("reads",))
    reads = delay["reads"]
    if type(reads) is not list or len(reads) > 4:
      raise LayoutError("read_list")
    for i, row in enumerate(reads):
      _semantic_node(row, READS[i])
    result = dict(raw)
    result["references"] = [dict(vars(row)) for row in refs]
    result["modelDelayReference"] = dict(delay, reads=[dict(row) for row in reads])
    return result


def _metadata(sm):
  # Eight scalar reads / fourteen Python lookups per side; never service data.
  first = sm.frame
  event = sm.logMonoTime["lateralDelay"]
  received = sm.recv_frame["lateralDelay"]
  seen = sm.seen["lateralDelay"]
  updated = sm.updated["lateralDelay"]
  valid = sm.valid["lateralDelay"]
  alive = sm.alive["lateralDelay"]
  last = sm.frame
  values = (first, event, received, seen, updated, valid, alive, last)
  if (any(type(v) is not int or not 0 <= v <= U64 for v in values[:3]+values[7:])
      or any(type(v) is not bool for v in values[3:7]) or not seen or not event or first != last):
    raise LayoutError("metadata_unavailable")
  return values


class HeldCacheTracker:
  """Per-consumer observed fence; no runtime ownership admission exists here.

  Matching tuples cannot exclude concurrent partial/untracked updates. Source
  reference, valid and association NEVER become known/valid/observed from them.
  """
  def __init__(self, owner):
    self.owner, self.sequence, self.inflight = owner, 0, None
    self.snapshot = MappingProxyType(empty_anchor())
    self.pending = False

  def begin(self, sm):
    try:
      if self.inflight is not None or self.sequence == U64:
        raise LayoutError("refresh_identity_unavailable")
      self.sequence += 1
      stamp, status = self.owner.now()
      token = {
        'sequence': self.sequence,
        'before': _metadata(sm),
        'begin': stamp,
        'begin_status': status,
        'old_read': self.owner.last_read,
        'offered': False,
        'value': 0.,
        'kind': "unknown",
      }
      self.inflight = token
      return token
    except Exception:
      self.owner.error()
      return None

  def offer(self, token, value):
    try:
      if token is None or token is not self.inflight or token["offered"]:
        raise LayoutError("refresh_token")
      token["offered"] = True
      if type(value) is float and math.isfinite(value):
        token["kind"], token["value"] = "knownFloat", value
      elif value is None:
        token["kind"] = "knownNone"
      else:
        self.owner.error()
    except Exception:
      self.owner.error()
    return value  # Exact original native argument, never coerced or replaced.

  def finish(self, token, sm):
    if token is None or token is not self.inflight:
      self.owner.error()
      return
    self.inflight = None
    row = empty_anchor()
    row.update(present=True, processEpoch=self.owner.epoch, instanceId=self.owner.identity,
               refreshTokenId=token["sequence"], refreshBeginNs=token["begin"], refreshBeginStatus=token["begin_status"],
               argumentValue=token["value"], argumentKind=token["kind"], argumentCaptured=token["offered"] and token["kind"] != "unknown")
    try:
      after = _metadata(sm)
      stamp, status = self.owner.now()
      row.update(refreshEndNs=stamp, refreshEndStatus=status)
      before = token["before"]
      row.update(sourceSmFrame=before[0], sourceEventNs=before[1], sourceRecvFrame=before[2],
                 sourceSeen=before[3], sourceUpdated=before[4], sourceValid=before[5], sourceAlive=before[6])
      cache, read = self.owner.cached, self.owner.last_read
      branch = cache["origin"]
      right_site = (read is not None and (
        (branch == "publishedHelper" and read.row["site"] == "lagdToggle" and read.row["valueKind"] == "knownBool"
         and read.row["boolValue"] is True)
        or (branch == "cachedHelper" and read.row["site"] == "delayCache" and read.row["castOutcome"] == "succeeded")))
      linked = (read is not None and read is not token["old_read"] and read.owner is self.owner
                and type(read.row["instanceId"]) is int and read.row["instanceId"] == self.owner.identity
                and read.completed and read.retained and read.write_seen and read.row["valid"] is True and right_site
                and read.row["outcome"] == "returned" and cache["read_known"] and cache["read_id"] == read.row["readId"])
      row.update(cacheReadId=cache["read_id"], associatedReadKnown=bool(linked), cacheWriteNs=cache["write_ns"],
                 cacheWriteClockStatus=cache["clock_status"], cacheWriteKnown=cache["clock_status"] == "known", branch=cache["origin"])
      if (before != after or not linked or not row["argumentCaptured"] or token["begin_status"] != "known"
          or status != "known" or not token["begin"] < cache["write_ns"] < stamp):
        raise LayoutError("refresh_witness_unavailable")
      row["associationStatus"] = "unsupported"  # No independently admitted completed-update ownership.
    except Exception:
      row["associationStatus"] = "invalid"
      self.owner.error()
    if self.pending:
      self.owner.error()  # Lost unretired history, never coalesced as complete.
    self.pending = True
    self.snapshot = MappingProxyType(row)

  def failed(self):
    self.inflight = None
    self.owner.error()

  def retire(self, revision):
    if revision is self.snapshot:
      self.pending = False
