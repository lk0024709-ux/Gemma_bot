/* ==========================================================================
   Gemma 3 Neuro-System — dashboard client
   Same-origin: every request uses a relative path, so no API base URL needed.
   ========================================================================== */

(() => {
  "use strict";

  // ---------------------------------------------------------------- config
  const API = {
    chat: "/api/chat",
    health: "/health",
    providers: "/api/providers",
  };
  const HISTORY_LIMIT = 10;      // messages sent back as context
  const HEALTH_POLL_MS = 60_000; // refresh sidebar every minute
  const REQUEST_TIMEOUT_MS = 120_000;

  // ------------------------------------------------------------------- dom
  const $ = (id) => document.getElementById(id);
  const els = {
    messages: $("messages"),
    welcome: $("welcome"),
    input: $("input"),
    sendBtn: $("sendBtn"),
    clearBtn: $("clearBtn"),
    menuBtn: $("menuBtn"),
    sidebar: $("sidebar"),
    statusDot: $("statusDot"),
    statusText: $("statusText"),
    uptime: $("uptime"),
    botStatus: $("botStatus"),
    lastProvider: $("lastProvider"),
    providerList: $("providerList"),
    providerBadge: $("providerBadge"),
    modelLine: $("modelLine"),
    temperature: $("temperature"),
    tempValue: $("tempValue"),
    maxTokens: $("maxTokens"),
    useHistory: $("useHistory"),
  };

  /** @type {{role: "user"|"assistant", content: string}[]} */
  let history = [];
  let busy = false;

  // ----------------------------------------------------------- utilities
  const escapeHtml = (s) =>
    String(s).replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );

  /** Minimal, safe markdown: escape first, then re-introduce code/bold/italic. */
  function render(text) {
    let html = escapeHtml(text);
    html = html.replace(/```(\w+)?\n?([\s\S]*?)```/g, (_, lang, code) => {
      const cls = lang ? ` class="language-${escapeHtml(lang)}"` : "";
      return `<pre><code${cls}>${code.replace(/\n$/, "")}</code></pre>`;
    });
    html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    return html;
  }

  function fmtUptime(seconds) {
    const s = Number(seconds) || 0;
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (d) return `${d}d ${h}h`;
    if (h) return `${h}h ${m}m`;
    if (m) return `${m}m`;
    return `${s}s`;
  }

  const scrollDown = () => {
    els.messages.scrollTop = els.messages.scrollHeight;
  };

  function hideWelcome() {
    if (els.welcome && els.welcome.parentNode) els.welcome.remove();
  }

  // ------------------------------------------------------------ messages
  function addMessage(role, text, meta) {
    hideWelcome();

    const row = document.createElement("div");
    row.className = `msg ${role === "user" ? "user" : "bot"}${meta?.error ? " error" : ""}`;

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = role === "user" ? "🧑" : "🧠";

    const body = document.createElement("div");
    body.className = "body";

    const who = document.createElement("div");
    who.className = "who";
    who.textContent = role === "user" ? "You" : "Gemma";

    const content = document.createElement("div");
    content.className = "content";
    if (role === "user") content.textContent = text;
    else content.innerHTML = render(text);

    body.append(who, content);

    if (meta && !meta.error && (meta.provider || meta.latency_ms != null)) {
      const bar = document.createElement("div");
      bar.className = "meta";
      const bits = [];
      if (meta.provider) bits.push(`⚡ ${meta.provider}`);
      if (meta.model) bits.push(`🧩 ${meta.model}`);
      if (meta.latency_ms != null) bits.push(`⏱ ${meta.latency_ms} ms`);
      if (meta.memory_id) bits.push(`💾 #${meta.memory_id}`);
      bits.forEach((b) => {
        const span = document.createElement("span");
        span.textContent = b;
        bar.appendChild(span);
      });
      body.appendChild(bar);
    }

    row.append(avatar, body);
    els.messages.appendChild(row);
    scrollDown();
    return row;
  }

  function addTyping() {
    hideWelcome();
    const row = document.createElement("div");
    row.className = "msg bot";
    row.id = "typingRow";
    row.innerHTML =
      '<div class="avatar">🧠</div>' +
      '<div class="body"><div class="who">Gemma</div>' +
      '<div class="typing"><i></i><i></i><i></i></div></div>';
    els.messages.appendChild(row);
    scrollDown();
    return row;
  }

  // ------------------------------------------------------------- sending
  async function send() {
    const text = els.input.value.trim();
    if (!text || busy) return;

    busy = true;
    els.sendBtn.disabled = true;
    els.input.value = "";
    els.input.style.height = "auto";

    addMessage("user", text);
    const typing = addTyping();
    els.modelLine.textContent = "Thinking…";

    const payload = {
      message: text,
      temperature: parseFloat(els.temperature.value),
      max_tokens: parseInt(els.maxTokens.value, 10) || 1024,
    };
    if (els.useHistory.checked && history.length) {
      payload.history = history.slice(-HISTORY_LIMIT);
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      const res = await fetch(API.chat, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      let data = null;
      try {
        data = await res.json();
      } catch {
        /* non-JSON error body */
      }

      typing.remove();

      if (!res.ok) {
        const detail =
          (data && (data.detail || data.error)) || `Request failed (HTTP ${res.status})`;
        addMessage("assistant", `⚠️ ${typeof detail === "string" ? detail : JSON.stringify(detail)}`, {
          error: true,
        });
        els.modelLine.textContent = "Error";
        return;
      }

      const reply = (data && data.response) || "(empty response)";
      addMessage("assistant", reply, data);

      history.push({ role: "user", content: text });
      history.push({ role: "assistant", content: reply });
      if (history.length > HISTORY_LIMIT * 2) history = history.slice(-HISTORY_LIMIT * 2);

      if (data && data.provider) {
        els.providerBadge.hidden = false;
        els.providerBadge.textContent = `${data.provider} · ${data.model || ""}`.trim();
        els.lastProvider.textContent = data.provider;
        markActiveProvider(data.provider);
      }
      els.modelLine.textContent = "Ready";
    } catch (err) {
      typing.remove();
      const msg =
        err.name === "AbortError"
          ? "Request timed out. The model took too long to respond."
          : `Network error: ${err.message}. Is the server running?`;
      addMessage("assistant", `⚠️ ${msg}`, { error: true });
      els.modelLine.textContent = "Error";
    } finally {
      clearTimeout(timer);
      busy = false;
      els.sendBtn.disabled = false;
      els.input.focus();
    }
  }

  // -------------------------------------------------------- health/status
  function markActiveProvider(name) {
    els.providerList.querySelectorAll(".provider").forEach((li) => {
      li.classList.toggle("active", li.dataset.name === name);
    });
  }

  function paintProviders(list) {
    els.providerList.innerHTML = "";
    list.forEach((p) => {
      const li = document.createElement("li");
      li.className = "provider";
      li.dataset.name = p.name;
      li.innerHTML =
        `<span class="dot ${p.configured ? "ok" : "err"}"></span>` +
        `<span class="name">${escapeHtml(p.name)}</span>` +
        `<span class="model">${escapeHtml(p.model || "")}</span>`;
      els.providerList.appendChild(li);
    });
  }

  async function refreshHealth() {
    try {
      const res = await fetch(API.health, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const h = await res.json();

      const healthy = h.status === "healthy";
      els.statusDot.className = `dot ${healthy ? "ok" : "warn"}`;
      els.statusText.textContent = h.status || "unknown";
      els.uptime.textContent = fmtUptime(h.uptime_seconds);

      if (Array.isArray(h.providers)) paintProviders(h.providers);

      const bot = h.telegram_bot || {};
      els.botStatus.textContent = !bot.enabled
        ? "disabled"
        : bot.running
        ? `online (${bot.messages_handled ?? 0} msgs)`
        : "offline";
      if (bot.last_provider) els.lastProvider.textContent = bot.last_provider;
    } catch (err) {
      els.statusDot.className = "dot err";
      els.statusText.textContent = "offline";
      console.warn("Health check failed:", err);
    }
  }

  // --------------------------------------------------------------- events
  function autoGrow() {
    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(els.input.scrollHeight, 190)}px`;
  }

  els.input.addEventListener("input", autoGrow);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  els.sendBtn.addEventListener("click", send);

  els.temperature.addEventListener("input", () => {
    els.tempValue.textContent = parseFloat(els.temperature.value).toFixed(1);
  });

  els.clearBtn.addEventListener("click", () => {
    history = [];
    els.messages.innerHTML = "";
    els.providerBadge.hidden = true;
    els.modelLine.textContent = "Ready";
    if (els.welcome) els.messages.appendChild(els.welcome);
    els.sidebar.classList.remove("open");
  });

  els.menuBtn.addEventListener("click", () => els.sidebar.classList.toggle("open"));

  document.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (chip) {
      els.input.value = chip.textContent.trim();
      autoGrow();
      send();
      return;
    }
    // Close the mobile drawer when tapping outside it
    if (
      window.innerWidth <= 860 &&
      els.sidebar.classList.contains("open") &&
      !els.sidebar.contains(e.target) &&
      e.target !== els.menuBtn
    ) {
      els.sidebar.classList.remove("open");
    }
  });

  // ----------------------------------------------------------------- init
  refreshHealth();
  setInterval(refreshHealth, HEALTH_POLL_MS);
  els.input.focus();
})();
