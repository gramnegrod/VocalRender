/* Human-reviewed, anonymized comparisons from cloudtest_supplementary. */
(function () {
  const sources = [
    { id: "gt", name: "Ground Truth", shortName: "GT", type: "reference" },
    { id: "soulx-singer", name: "SoulX-Singer", type: "baseline" },
    { id: "tcsinger", name: "TCSinger", type: "baseline" },
    { id: "techsinger", name: "TechSinger", type: "baseline" },
    { id: "vevo2", name: "Vevo2", type: "baseline" },
    { id: "vocalrender", name: "VocalRender", type: "proposed", tag: "Ours" },
    { id: "vocalrender-pro", name: "VocalRender-Pro", type: "proposed", tag: "Ours · Pro" }
  ];

  const samples = Array.from({ length: 10 }, (_, index) => ({
    id: `sample-${String(index + 1).padStart(2, "0")}`,
    number: String(index + 1).padStart(2, "0")
  }));

  window.VOCALRENDER_DEMOS = samples.map((sample) => ({
    ...sample,
    score: `assets/scores/${sample.id}.png`,
    methods: sources.map((source) => ({
      ...source,
      src: `assets/audio/${sample.id}/${source.id}.flac`
    }))
  }));
})();
