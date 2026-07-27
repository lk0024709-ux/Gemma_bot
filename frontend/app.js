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
    config: "/api/config",
    membership: (id) => `/api/membership/${id}`,
  };
  const HISTORY_LIMIT = 10;      // messages sent back as context
  const HEALTH_POLL_MS = 60_000; // refresh sidebar every minute
  const REQUEST_TIMEOUT_MS = 120_000;

  // ------------------------------------------------- Telegram Mini App init
  // window.Telegram only exists when the page is opened inside Telegram (or if
  // the CDN script loaded). Everything below degrades gracefully in a browser.
  const tg = window.Telegram?.WebApp || null;
  const inTelegram = Boolean(tg && tg.initData !== undefined && tg.platform !== "unknown");
  let tgUserId = null;
  let inviteLink = "";

  if (tg) {
    try {
      tg.ready();
      tg.expand?.();
      tgUserId = tg.initDataUnsafe?.user?.id ?? null;
      if (inTelegram) document.body.classList.add("tma");
    } catch (err) {
      console.warn("Telegram WebApp init failed:", err);
    }
  }

  /** Open a t.me link via Telegram when available, else a normal new tab. */
  function openChannel(link) {
    if (!link) return;
    try {
      if (tg && typeof tg.openTelegramLink === "function" && /^https?:\/\/t\.me\//i.test(link)) {
        tg.openTelegramLink(link);
        return;
      }
      if (tg && typeof tg.openLink === "function") {
        tg.openLink(link);
        return;
      }
    } catch (err) {
      console.warn("Telegram link open failed, falling back:", err);
    }
    window.open(link, "_blank", "noopener");
  }

  const haptic = (type) => {
    try {
      tg?.HapticFeedback?.notificationOccurred?.(type);
    } catch {
      /* not available outside Telegram */
    }
  };

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
    gate: $("gate"),
    gateText: $("gateText"),
    gateJoin: $("gateJoin"),
    gateRecheck: $("gateRecheck"),
    gateHint: $("gateHint"),
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
    if (tgUserId != null) payload.telegram_user_id = tgUserId;
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

      // --- Force-subscribe: user has not joined the channel --------------- //
      if (res.status === 403 && data && data.error === "FORBIDDEN_NOT_MEMBER") {
        inviteLink = data.invite_link || inviteLink;
        haptic("error");
        showJoinPrompt(data.message);
        els.modelLine.textContent = "Access denied";
        return;
      }

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

  // ------------------------------------------------------ force-subscribe
  /** Render the "join the channel" message inside the chat transcript. */
  function showJoinPrompt(message) {
    hideWelcome();

    const row = document.createElement("div");
    row.className = "msg bot error";

    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "🔒";

    const body = document.createElement("div");
    body.className = "body";

    const who = document.createElement("div");
    who.className = "who";
    who.textContent = "Access";

    const content = document.createElement("div");
    content.className = "content";
    content.textContent =
      message || "🔒 Access Denied! Please join our channel to use this AI.";

    body.append(who, content);

    if (!tgUserId && !inTelegram) {
      const note = document.createElement("div");
      note.className = "meta";
      note.textContent =
        "Open this app from inside Telegram so we can verify your membership.";
      body.appendChild(note);
    }

    if (inviteLink) {
      const link = document.createElement("a");
      link.className = "join-inline";
      link.href = inviteLink;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = "📢 Join Channel";
      link.addEventListener("click", (e) => {
        // Inside Telegram, open natively instead of spawning a browser tab.
        if (tg && typeof tg.openTelegramLink === "function") {
          e.preventDefault();
          openChannel(inviteLink);
        }
      });
      body.appendChild(link);

      const recheck = document.createElement("button");
      recheck.className = "join-inline";
      recheck.type = "button";
      recheck.style.marginLeft = "8px";
      recheck.textContent = "✅ I've joined";
      recheck.addEventListener("click", () => recheckMembership(recheck));
      body.appendChild(recheck);
    }

    row.append(avatar, body);
    els.messages.appendChild(row);
    scrollDown();
  }

  /** Ask the backend to re-verify membership (after the user joins). */
  async function recheckMembership(btn) {
    if (tgUserId == null) {
      showGate("Open this Mini App from Telegram so we can verify your membership.");
      return;
    }
    const original = btn ? btn.textContent : null;
    if (btn) {
      btn.disabled = true;
      btn.textContent = "Checking…";
    }
    try {
      const res = await fetch(API.membership(tgUserId), { cache: "no-store" });
      const data = await res.json();
      if (data.is_member) {
        haptic("success");
        hideGate();
        addMessage("assistant", "✅ Membership verified — you're all set. Ask me anything!");
      } else {
        haptic("error");
        setGateHint("Still not a member. Join the channel, then try again.", "err");
        if (btn) btn.textContent = "❌ Not yet — retry";
      }
    } catch (err) {
      setGateHint(`Could not verify: ${err.message}`, "err");
      if (btn) btn.textContent = original || "✅ I've joined";
    } finally {
      if (btn) {
        btn.disabled = false;
        setTimeout(() => {
          if (original) btn.textContent = original;
        }, 2500);
      }
    }
  }

  function setGateHint(text, kind) {
    els.gateHint.textContent = text || "";
    els.gateHint.className = `gate-hint${kind ? " " + kind : ""}`;
  }

  function showGate(reason) {
    if (reason) els.gateText.textContent = reason;
    if (inviteLink) {
      els.gateJoin.href = inviteLink;
      els.gateJoin.hidden = false;
    } else {
      els.gateJoin.hidden = true;
    }
    els.gate.hidden = false;
  }

  const hideGate = () => {
    els.gate.hidden = true;
    setGateHint("");
  };

  /** Load public config and, when force-subscribe is on, pre-verify the user. */
  async function initGate() {
    let cfg;
    try {
      const res = await fetch(API.config, { cache: "no-store" });
      cfg = await res.json();
    } catch (err) {
      console.warn("Could not load config:", err);
      return;
    }

    inviteLink = cfg.invite_link || "";
    if (!cfg.force_subscribe) return; // gate disabled server-side

    if (tgUserId == null) {
      // Opened outside Telegram (or initData unavailable): explain, don't block
      // silently. The server still enforces the rule on every request.
      showGate(
        inTelegram
          ? "We couldn't read your Telegram account. Please reopen the app from the bot."
          : "This app must be opened from inside Telegram so we can verify your channel membership."
      );
      els.gateRecheck.hidden = !inTelegram;
      return;
    }

    try {
      const res = await fetch(API.membership(tgUserId), { cache: "no-store" });
      const data = await res.json();
      if (!data.is_member) {
        showGate("Please join our channel to use this AI.");
      }
    } catch (err) {
      console.warn("Membership pre-check failed:", err);
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

  els.gateJoin.addEventListener("click", (e) => {
    if (tg && typeof tg.openTelegramLink === "function" && inviteLink) {
      e.preventDefault();
      openChannel(inviteLink);
    }
  });

  els.gateRecheck.addEventListener("click", () => recheckMembership(els.gateRecheck));

  // ----------------------------------------------------------------- init
  initGate();
  refreshHealth();
  setInterval(refreshHealth, HEALTH_POLL_MS);
  if (!inTelegram) els.input.focus(); // avoid forcing the keyboard open on mobile
})();
