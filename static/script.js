const chatBox = document.getElementById("chatBox");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const rebuildBtn = document.getElementById("rebuildBtn");
const rebuildStatus = document.getElementById("rebuildStatus");

function appendMessage(text, cls) {
  const el = document.createElement("div");
  el.className = "msg " + cls;
  el.innerHTML = text; // allow HTML (for images)
  chatBox.appendChild(el);
  chatBox.scrollTop = chatBox.scrollHeight;
}

sendBtn.onclick = sendMessage;
userInput.addEventListener("keypress", (e) => { if (e.key === "Enter") sendMessage(); });

function sendMessage() {
  const q = userInput.value.trim();
  if (!q) return;
  appendMessage(`<b>You:</b> ${escapeHtml(q)}`, "user");
  userInput.value = "";
  appendMessage("<i>Thinking...</i>", "bot");
  const thinking = chatBox.lastChild;

  fetch("/ask", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query: q})
  }).then(r => r.json())
    .then(d => {
      thinking.remove();
      appendMessage(`<b>Bot:</b> ${d.answer}`, "bot");
    })
    .catch(err => {
      thinking.remove();
      appendMessage(`<b>Bot:</b> Error: ${err}`, "bot");
    });
}

rebuildBtn.onclick = () => {
  rebuildStatus.textContent = "Rebuilding index…";
  fetch("/rebuild_index", { method: "POST" })
    .then(r => r.json())
    .then(d => {
      if (d.status === "ok") rebuildStatus.textContent = d.message;
      else rebuildStatus.textContent = "Error rebuilding: " + d.message;
    })
    .catch(e => rebuildStatus.textContent = "Error: " + e);
};

// escape only for user input (not bot's HTML)
function escapeHtml(s){ return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }
