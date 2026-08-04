# Demo audio

The project page reads its comparison data from
`assets/site/demo-data.js`. The current curated set was copied from
`/export/home2/n2505683d/VoxCPM/output/cloudtest` and uses this layout:

```text
assets/audio/
├── sample-01/
│   ├── gt.flac
│   ├── soulx-singer.flac
│   ├── tcsinger.flac
│   ├── techsinger.flac
│   ├── vevo2.flac
│   ├── vocalrender.flac
│   └── vocalrender-pro.flac
├── sample-02/
│   └── ...
└── sample-08/
    └── ...
```

Matching score images live at `assets/scores/sample-01.png` through
`sample-08.png`. Missing files are handled gracefully: the corresponding
player shows **Pending audio** and remains disabled.

Edit the sample metadata and source list in `demo-data.js`. Adding another
sample object automatically creates a new comparison card—no HTML changes are
needed.

The checked-in FLAC files retain each system's native experiment sample rate
(24 or 48 kHz). If repository size or page-load time becomes a concern,
publish compressed browser copies and update the `src` extension in the
manifest while retaining lossless originals with the experiment artifacts.
