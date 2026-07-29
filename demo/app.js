/**
 * demo/app.js
 * -----------
 * IRA · Image Studio — interactive playground to test the Flux image model.
 *
 * Talks to the local FastAPI demo endpoints:
 *   GET  /api/image/status  → is the HF token configured?
 *   POST /api/image         → generate one image + metadata
 *
 * No Telegram / initData needed — this is a standalone model test harness.
 */

(function () {
  "use strict";

  // ----------------------------------------------------------------------- //
  //  Example prompts (also power the "Surprise me" button)
  // ----------------------------------------------------------------------- //
  const SUGGESTIONS = [
    "a neon-lit cyberpunk street market in the rain, cinematic, ultra detailed",
    "a cozy treehouse library at golden hour, warm light, fantasy art",
    "macro shot of a hummingbird drinking from a glowing flower, 8k",
    "an astronaut floating over a desert planet with two moons, retro poster",
    "a cute corgi wearing a tiny wizard hat, studio lighting, pixar style",
    "misty mountain village at dawn, traditional ink wash painting",
    "futuristic electric sports car in a glass tunnel, reflections, photoreal",
  ];

  // ----------------------------------------------------------------------- //
  //  DOM refs
  // ----------------------------------------------------------------------- //
  const $ = (id) => document.getElementById(id);
  const form        = $("promptForm");
  const promptEl    = $("prompt");
  const generateBtn = $("generateBtn");
  const surpriseBtn = $("surpriseBtn");
  const suggestions = $("suggestions");
  const gallery     = $("gallery");
  const emptyState  = $("emptyState");
  const clearBtn    = $("clearBtn");
  const statusDot   = $("statusDot");
  const statusText  = $("statusText");
  const toast       = $("toast");

  let cardSeq = 0;

  // ----------------------------------------------------------------------- //
  //  Boot
  // ----------------------------------------------------------------------- //
  renderSuggestions();
  checkStatus();

  // Cmd/Ctrl + Enter submits from the textarea
  promptEl.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  form.addEventListener("submit", onSubmit);
  surpriseBtn.addEventListener("click", onSurprise);
  clearBtn.addEventListener("click", onClear);

  // ----------------------------------------------------------------------- //
  //  Status check
  // ----------------------------------------------------------------------- //
  async function checkStatus() {
    setStatus("checking…", "");
    try {
      const res = await fetch("/api/image/status");
      const data = await res.json();
      if (data.token_configured) {
        setStatus("Model ready", "ok");
      } else {
        setStatus("HF token not configured", "warn");
        showToast(
          "No Hugging Face token found. Set HF_TOKEN_1, HF_TOKEN_2 or HF_TOKEN, " +
          "then restart the server to test the model.",
          false
        );
      }
    } catch (err) {
      console.error(err);
      setStatus("Server offline", "err");
    }
  }

  function setStatus(text, state) {
    statusText.textContent = text;
    statusDot.className = "dot" + (state ? " " + state : "");
  }

  // ----------------------------------------------------------------------- //
  //  Suggestions + surprise
  // ----------------------------------------------------------------------- //
  function renderSuggestions() {
    suggestions.innerHTML = "";
    SUGGESTIONS.slice(0, 4).forEach((text) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = text.split(",")[0];
      chip.title = text;
      chip.addEventListener("click", () => {
        promptEl.value = text;
        promptEl.focus();
      });
      suggestions.appendChild(chip);
    });
  }

  function onSurprise() {
    const pick = SUGGESTIONS[Math.floor(Math.random() * SUGGESTIONS.length)];
    promptEl.value = pick;
    promptEl.focus();
  }

  // ----------------------------------------------------------------------- //
  //  Submit → generate
  // ----------------------------------------------------------------------- //
  async function onSubmit(e) {
    e.preventDefault();
    const prompt = promptEl.value.trim();
    if (!prompt) {
      showToast("Please enter a prompt first.", false);
      promptEl.focus();
      return;
    }
    if (generateBtn.disabled) return; // prevent double-submit

    setBusy(true);
    const card = addCard({ prompt, loading: true });

    try {
      const res = await fetch("/api/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok || !data.ok) {
        throw new Error(data.detail || `Request failed (HTTP ${res.status})`);
      }

      renderImageCard(card, data);
      showToast("Image generated ✓", true);
    } catch (err) {
      console.error(err);
      renderErrorCard(card, prompt, err.message);
      showToast(err.message, false);
    } finally {
      setBusy(false);
    }
  }

  // ----------------------------------------------------------------------- //
  //  Cards
  // ----------------------------------------------------------------------- //
  function addCard({ prompt, loading }) {
    // Hide the empty state the first time we render anything.
    if (emptyState) emptyState.hidden = true;
    clearBtn.hidden = false;

    const card = document.createElement("article");
    card.className = "card" + (loading ? " loading" : "");
    card.dataset.id = ++cardSeq;

    card.innerHTML = `
      <div class="card-media">
        ${loading ? `<span class="card-loading-label"><span class="mini-spinner"></span> generating…</span>` : ""}
      </div>
      <div class="card-body">
        <p class="card-prompt">${escapeHtml(prompt)}</p>
      </div>`;

    // Newest first
    gallery.prepend(card);
    return card;
  }

  function renderImageCard(card, data) {
    card.classList.remove("loading");
    const media = card.querySelector(".card-media");
    const body = card.querySelector(".card-body");

    const src = `data:${data.content_type};base64,${data.image_b64}`;
    const ext = (data.content_type || "image/png").split("/")[1] || "png";
    const sizeKb = (data.size_bytes / 1024).toFixed(1);
    const shortModel = data.model.split("/").pop();

    media.innerHTML = `<img src="${src}" alt="${escapeHtml(data.prompt)}" loading="lazy" />`;

    const download = document.createElement("div");
    download.className = "card-actions";
    download.innerHTML =
      `<a class="btn-download" href="${src}" download="ira-${Date.now()}.${ext}">⬇ Download</a>`;

    body.insertAdjacentHTML(
      "beforeend",
      `<span class="card-model" title="${escapeHtml(data.endpoint)}">${escapeHtml(shortModel)}</span>
       <div class="card-meta">
         <span>⏱ <b>${data.elapsed_ms} ms</b></span>
         <span>📦 ${sizeKb} KB</span>
         <span>${escapeHtml(ext.toUpperCase())}</span>
       </div>`
    );
    body.appendChild(download);
  }

  function renderErrorCard(card, prompt, message) {
    card.classList.remove("loading");
    card.classList.add("error");
    card.querySelector(".card-media").innerHTML =
      `<span style="color:var(--muted);font-size:28px;">⚠️</span>`;
    card.querySelector(".card-body").insertAdjacentHTML(
      "beforeend",
      `<div class="card-meta"><span>${escapeHtml(message)}</span></div>`
    );
  }

  function onClear() {
    gallery.querySelectorAll(".card").forEach((c) => c.remove());
    if (emptyState) emptyState.hidden = false;
    clearBtn.hidden = true;
  }

  // ----------------------------------------------------------------------- //
  //  Helpers
  // ----------------------------------------------------------------------- //
  function setBusy(busy) {
    generateBtn.disabled = busy;
    const label = generateBtn.querySelector(".btn-label");
    const spinner = generateBtn.querySelector(".btn-spinner");
    if (label && spinner) {
      spinner.hidden = !busy;
      label.textContent = busy ? "Generating…" : "✨ Generate";
    }
  }

  let toastTimer = null;
  function showToast(message, ok) {
    toast.textContent = message;
    toast.className = "toast show" + (ok ? " ok" : "");
    toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toast.classList.remove("show");
      // Hide after the fade-out transition
      setTimeout(() => { toast.hidden = true; }, 260);
    }, 4200);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
})();
