usage.jsonl is created here at runtime.

usage-aggregate-cache.json is a restart accelerator: it stores the dashboard's
incremental aggregates (usage snapshot, per-account totals, analytics all-time
fold) together with the log offset they cover, so the first refresh after a
restart only reads the rows appended since. It is validated against the log
(offset, device/inode, tail signature) and against the price tables and realm
assignments when it is loaded; anything that does not match is ignored and the
full scan happens instead, so deleting the file is always safe. Set
WB_USAGE_CACHE=0 to turn it off entirely.

usage-summary.json is a leftover from older builds, which rewrote it on every
request even though nothing ever read it back. New versions append to
usage.jsonl only; you can delete the old summary file safely.
