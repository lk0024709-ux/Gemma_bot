/**
 * frontend/app.js
 * ---------------
 * Telegram Mini App (TMA) frontend — secure initData handling.
 *
 * Replaces the insecure `tg.initDataUnsafe` pattern with the cryptographically
 * signed `window.Telegram.WebApp.initData` raw string, which is validated by
 * the backend before any processing.
 */

(function () {
  "use strict";

  // -----------------------------------------------------------------------
  //  Telegram WebApp SDK — must be loaded before this script runs
  //  <script src="https://telegram.org/js/telegram-web-app.js"></script>
  // -----------------------------------------------------------------------
  const tg = window.Telegram?.WebApp;

  if (!tg) {
    console.error(
      "Telegram WebApp SDK not found. " +
      "Make sure https://telegram.org/js/telegram-web-app.js is loaded before app.js."
    );
    return;
  }

  // Expand the Mini App to fill the available viewport.
  tg.expand();

  // -----------------------------------------------------------------------
  //  DOM references (adjust selectors to match your HTML)
  // -----------------------------------------------------------------------
  const chatForm    = document.getElementById("chat-form");     // <form> element
  const messageInput = document.getElementById("message-input"); // <input> or <textarea>
  const chatWindow  = document.getElementById("chat-window");   // container for messages

  // -----------------------------------------------------------------------
  //  Secure initData — the raw, signed string from Telegram
  // -----------------------------------------------------------------------
  const rawInitData = tg.initData;   // ← SECURE: cryptographically signed by Telegram

  if (!rawInitData) {
    showError(
      "Unable to retrieve Telegram session data. " +
      "Please open this page inside the Telegram app."
    );
    return;
  }

  // -----------------------------------------------------------------------
  //  Chat form submission
  // -----------------------------------------------------------------------
  if (chatForm) {
    chatForm.addEventListener("submit", async function (e) {
      e.preventDefault();

      const message = messageInput?.value?.trim();
      if (!message) return;

      appendMessage("user", message);
      if (messageInput) messageInput.value = "";

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tg_init_data: rawInitData,   // ← send the signed initData string
          }),
        });

        // ----------------------------------------------------------------
        //  Handle 401 Unauthorized — spoofing detected or opened outside TG
        // ----------------------------------------------------------------
        if (response.status === 401) {
          const errBody = await response.json().catch(() => ({}));
          if (errBody.detail === "UNAUTHORIZED_SPOOFING_DETECTED") {
            showError(
              "Access denied: unable to verify your Telegram session. " +
              "Please make sure you are using this app inside Telegram."
            );
          } else {
            showError("Unauthorized. Please reopen this app from Telegram.");
          }
          return;
        }

        if (!response.ok) {
          showError("Server error. Please try again later.");
          return;
        }

        const data = await response.json();
        appendMessage("bot", data.reply);
      } catch (err) {
        console.error("Chat request failed:", err);
        showError("Network error. Please check your connection and try again.");
      }
    });
  }

  // -----------------------------------------------------------------------
  //  DOM helpers
  // -----------------------------------------------------------------------
  function appendMessage(role, text) {
    if (!chatWindow) return;

    const bubble = document.createElement("div");
    bubble.className = `message message--${role}`;
    bubble.textContent = text;
    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  function showError(msg) {
    // Display an inline error banner.
    let banner = document.getElementById("tma-error-banner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "tma-error-banner";
      banner.style.cssText =
        "background:#ff3b3b;color:#fff;padding:12px;text-align:center;font-size:14px;";
      document.body.prepend(banner);
    }
    banner.textContent = msg;
    banner.style.display = "block";
  }
})();
