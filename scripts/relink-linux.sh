#!/bin/bash
# Re-link the NativeAOT object file with a proper ELF version script so the
# Sts2_* / Sts2Run_* API symbols land in .dynsym. Works around ILC 9.x on
# Linux generating a `local: *` version script that hides every export.
#
# Usage (inside the dotnet SDK build container, after `dotnet publish`):
#   bash scripts/relink-linux.sh <project-dir> <output-dir>
#
# project-dir contains obj/Release/net9.0/linux-x64/native/Sts2Emulator.o and
# Sts2Emulator.linux.exports; NuGet packages are under /root/.nuget.
set -euo pipefail

PROJECT_DIR="${1:?project dir}"
OUT_DIR="${2:?output dir}"
NATIVE_DIR="$PROJECT_DIR/obj/Release/net9.0/linux-x64/native"
ILC_ROOT="$(echo /root/.nuget/packages/runtime.linux-x64.microsoft.dotnet.ilcompiler/*/ | head -1)"

if [ ! -f "$NATIVE_DIR/Sts2Emulator.o" ]; then
    echo "object file not found: $NATIVE_DIR/Sts2Emulator.o" >&2
    exit 1
fi
if [ -z "$ILC_ROOT" ] || [ ! -d "$ILC_ROOT" ]; then
    echo "ILCompiler package not found under /root/.nuget" >&2
    exit 1
fi

DEFSYMS=()
while IFS='=' read -r short long; do
    [ -n "$short" ] && DEFSYMS+=("-Wl,--defsym,${short}=${long}")
done <<'EOF'
Sts2_ObsSize=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_ObsSize
Sts2_MaxEnemies=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_MaxEnemies
Sts2_NativeApiVersion=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_NativeApiVersion
Sts2_Create=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_Create
Sts2_Reset=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_Reset
Sts2_ResetEncounter=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_ResetEncounter
Sts2_ResetWithDeck=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_ResetWithDeck
Sts2_ResetWithDeckAndEncounter=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_ResetWithDeckAndEncounter
Sts2_ResetWithDeckEncounterAndRelics=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_ResetWithDeckEncounterAndRelics
Sts2_Step=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_Step
Sts2_StepTargeted=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_StepTargeted
Sts2_PlayerWon=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_PlayerWon
Sts2_EncounterId=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_EncounterId
Sts2_ActionCount=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_ActionCount
Sts2_ValidActions=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_ValidActions
Sts2_Destroy=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2_Destroy
Sts2Run_NativeApiVersion=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_NativeApiVersion
Sts2Run_ObsSize=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_ObsSize
Sts2Run_MaxActions=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_MaxActions
Sts2Run_InfoSize=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_InfoSize
Sts2Run_Create=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_Create
Sts2Run_Reset=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_Reset
Sts2Run_Step=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_Step
Sts2Run_StartCombat=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_StartCombat
Sts2Run_ActionMask=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_ActionMask
Sts2Run_GetInfo=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_GetInfo
Sts2Run_GetStateList=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_GetStateList
Sts2Run_GetPhase=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_GetPhase
Sts2Run_PlayerWon=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_PlayerWon
Sts2Run_EncounterId=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_EncounterId
Sts2Run_GetShuffleRngCallCount=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_GetShuffleRngCallCount
Sts2Run_GetNicheRngCallCount=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_GetNicheRngCallCount
Sts2Run_Destroy=Sts2Emulator_Sts2Emulator_Interop_NativeExports__Sts2Run_Destroy
EOF

mkdir -p "$OUT_DIR"
cd "$PROJECT_DIR"

# shellcheck disable=SC2046
clang "$NATIVE_DIR/Sts2Emulator.o" -o "$OUT_DIR/Sts2Emulator.so" \
    -Wl,--version-script=Sts2Emulator.linux.exports \
    -Wl,--export-dynamic \
    "${DEFSYMS[@]}" \
    -gz=zlib -fuse-ld=bfd \
    "$ILC_ROOT"sdk/libbootstrapperdll.o \
    "$ILC_ROOT"sdk/libRuntime.WorkstationGC.a \
    "$ILC_ROOT"sdk/libeventpipe-disabled.a \
    "$ILC_ROOT"sdk/libRuntime.VxsortEnabled.a \
    "$ILC_ROOT"sdk/libstandalonegc-disabled.a \
    "$ILC_ROOT"sdk/libstdc++compat.a \
    "$ILC_ROOT"framework/libSystem.Native.a \
    "$ILC_ROOT"framework/libSystem.IO.Compression.Native.a \
    "$ILC_ROOT"framework/libSystem.Net.Security.Native.a \
    "$ILC_ROOT"framework/libSystem.Security.Cryptography.Native.OpenSsl.a \
    "$ILC_ROOT"sdk/libz.a \
    -g -Wl,-rpath,'$ORIGIN' -Wl,--build-id=sha1 -Wl,--as-needed -Wl,-e,0x0 \
    -pthread -ldl -lrt -lm -shared \
    -Wl,-z,relro -Wl,-z,now -Wl,--eh-frame-hdr -Wl,--discard-all -Wl,--gc-sections

echo "relinked: $OUT_DIR/Sts2Emulator.so"
nm -D "$OUT_DIR/Sts2Emulator.so" | grep -cE 'Sts2|Sts2Run' || true
