#!/bin/sh
# Measures peak memory of reading a 3 GiB file with one pread, and with 2 GiB chunks joined once (what safetensors falls back to).
# Run with: ERL=path/to/erl ./memory.sh
# On macOS with current OTP versions, the single read fails with einval.
set -e
ERL=${ERL:-erl}
FILE=$(mktemp -t big3)
mkfile -n 3g "$FILE" 2>/dev/null || truncate -s 3G "$FILE"
"$ERL" -noshell -eval 'io:format("OTP ~s, erts ~s~n", [erlang:system_info(otp_release), erlang:system_info(version)]), halt().'
for mode in single chunks; do
  /usr/bin/time -l "$ERL" -noshell -eval '
    [Mode, File] = init:get_plain_arguments(), S = 3 bsl 30, M = (1 bsl 31) - 1,
    {ok, F} = file:open(File, [raw, read, binary]),
    R = case Mode of
      "single" -> file:pread(F, 0, S);
      "chunks" -> {ok, A} = file:pread(F, 0, M), {ok, C} = file:pread(F, M, S - M), {ok, iolist_to_binary([A, C])}
    end,
    case R of
      {ok, B} -> io:format("~s: read ~p bytes~n", [Mode, byte_size(B)]);
      Other -> io:format("~s: ~p~n", [Mode, Other])
    end,
    halt().' -extra $mode "$FILE" 2>&1 | grep -E "read|einval|error|peak memory"
done
rm -f "$FILE"
