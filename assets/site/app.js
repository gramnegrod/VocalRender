(function () {
  "use strict";

  const demos = window.VOCALRENDER_DEMOS || [];
  const demoList = document.getElementById("demo-list");
  let activeAudio = null;
  let activeButton = null;

  const escapeHtml = (value) =>
    String(value).replace(/[&<>"]/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"
    })[character]);

  const formatTime = (seconds) => {
    if (!Number.isFinite(seconds)) return "0:00";
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
  };

  const methodMarkup = (sample, method) => {
    const audioId = `audio-${sample.id}-${method.id}`;
    return `
      <div class="method-tile" data-type="${escapeHtml(method.type)}" data-method="${escapeHtml(method.id)}">
        <div class="method-tile-head">
          <div class="method-name"><i aria-hidden="true"></i>${escapeHtml(method.name)}</div>
          ${method.tag ? `<span class="method-tag">${escapeHtml(method.tag)}</span>` : ""}
          <div class="audio-status pending">Loading</div>
        </div>
        <div class="audio-player" data-src="${escapeHtml(method.src)}">
          <audio id="${audioId}" preload="metadata"></audio>
          <button class="play-button" type="button" aria-label="Play ${escapeHtml(method.name)}" disabled>▶</button>
          <input class="progress" type="range" min="0" max="100" value="0" step="0.1" aria-label="Audio progress" disabled />
          <span class="time">0:00</span>
        </div>
      </div>`;
  };

  const cardMarkup = (sample) => `
    <article class="demo-card">
      <header class="demo-card-header">
        <div class="sample-number"><span>Sample</span>${escapeHtml(sample.number)}</div>
      </header>
      <div class="demo-card-body">
        <figure class="score-panel">
          <div class="score-label"><span>Input score</span><small>Click to enlarge</small></div>
          <button class="score-open" type="button" data-score="${escapeHtml(sample.score)}" data-title="Sample ${escapeHtml(sample.number)}" aria-label="Enlarge score for sample ${escapeHtml(sample.number)}">
            <img src="${escapeHtml(sample.score)}" alt="Music score for sample ${escapeHtml(sample.number)}" loading="lazy" />
          </button>
        </figure>
        <div class="methods-panel">
          <div class="methods-label"><span>Audio renders</span><small>One score · seven sources</small></div>
          <div class="methods-grid">${sample.methods.map((method) => methodMarkup(sample, method)).join("")}</div>
        </div>
      </div>
    </article>`;

  demoList.innerHTML = demos.map(cardMarkup).join("");

  const stopActiveAudio = (except) => {
    if (activeAudio && activeAudio !== except) {
      activeAudio.pause();
      if (activeButton) {
        activeButton.textContent = "▶";
        activeButton.classList.remove("playing");
        activeButton.setAttribute("aria-label", activeButton.getAttribute("aria-label").replace("Pause", "Play"));
      }
    }
  };

  document.querySelectorAll(".audio-player").forEach((player) => {
    const audio = player.querySelector("audio");
    const button = player.querySelector(".play-button");
    const progress = player.querySelector(".progress");
    const time = player.querySelector(".time");
    const status = player.parentElement.querySelector(".audio-status");
    const source = player.dataset.src;

    audio.src = source;
    audio.addEventListener("loadedmetadata", () => {
      button.disabled = false;
      progress.disabled = false;
      time.textContent = formatTime(audio.duration);
      status.textContent = "Ready";
      status.className = "audio-status ready";
    });
    audio.addEventListener("error", () => {
      button.disabled = true;
      progress.disabled = true;
      status.textContent = "Unavailable";
      status.className = "audio-status pending";
    });
    audio.addEventListener("timeupdate", () => {
      progress.value = audio.duration ? (audio.currentTime / audio.duration) * 100 : 0;
      time.textContent = `${formatTime(audio.currentTime)} / ${formatTime(audio.duration)}`;
    });
    audio.addEventListener("ended", () => {
      button.textContent = "▶";
      button.classList.remove("playing");
      button.setAttribute("aria-label", button.getAttribute("aria-label").replace("Pause", "Play"));
      progress.value = 0;
    });

    button.addEventListener("click", () => {
      if (audio.paused) {
        stopActiveAudio(audio);
        audio.play().then(() => {
          activeAudio = audio;
          activeButton = button;
          button.textContent = "Ⅱ";
          button.classList.add("playing");
          button.setAttribute("aria-label", button.getAttribute("aria-label").replace("Play", "Pause"));
        }).catch(() => {});
      } else {
        audio.pause();
        button.textContent = "▶";
        button.classList.remove("playing");
        button.setAttribute("aria-label", button.getAttribute("aria-label").replace("Pause", "Play"));
      }
    });
    progress.addEventListener("input", () => {
      if (audio.duration) audio.currentTime = (Number(progress.value) / 100) * audio.duration;
    });
  });

  const scoreDialog = document.getElementById("score-dialog");
  const dialogImage = scoreDialog.querySelector("img");
  document.querySelectorAll(".score-open").forEach((button) => {
    button.addEventListener("click", () => {
      dialogImage.src = button.dataset.score;
      dialogImage.alt = `Music score for ${button.dataset.title}`;
      scoreDialog.showModal();
    });
  });
  scoreDialog.querySelector(".score-dialog-close").addEventListener("click", () => scoreDialog.close());
  scoreDialog.addEventListener("click", (event) => {
    if (event.target === scoreDialog) scoreDialog.close();
  });

})();
