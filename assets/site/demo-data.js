/* Curated audio comparisons copied from output/cloudtest. */
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

  const samples = [
    { id: "sample-01", number: "01", artist: "黄玮昕 (Haezee)", title: "不完美心跳", clip: "Excerpt 0006" },
    { id: "sample-02", number: "02", artist: "駝背人", title: "笨手笨脚地爱你", clip: "Excerpt 0000" },
    { id: "sample-03", number: "03", artist: "罗森涛", title: "太多余", clip: "Excerpt 0007" },
    { id: "sample-04", number: "04", artist: "罗森涛", title: "零点", clip: "Excerpt 0008" },
    { id: "sample-05", number: "05", artist: "罗森涛", title: "不眠", clip: "Excerpt 0011" },
    { id: "sample-06", number: "06", artist: "郭家玮", title: "恋罪诀", clip: "Excerpt 0007" },
    { id: "sample-07", number: "07", artist: "王琪玮", title: "丁达尔现象", clip: "Excerpt 0016" },
    { id: "sample-08", number: "08", artist: "大凉山妞妞合唱团", title: "花季", clip: "Excerpt 0009" }
  ];

  window.VOCALRENDER_DEMOS = samples.map((sample) => ({
    ...sample,
    language: "zh",
    languageLabel: "Mandarin",
    score: `assets/scores/${sample.id}.png`,
    methods: sources.map((source) => ({
      ...source,
      src: `assets/audio/${sample.id}/${source.id}.flac`
    }))
  }));
})();
