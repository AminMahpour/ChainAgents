function asText(value) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

const STATUS_CLASSES = {
  idle: "border-border bg-muted text-muted-foreground",
  running: "border-primary/40 bg-primary/10 text-primary",
  stopping: "border-amber-500/40 bg-amber-500/10 text-amber-600",
  stopped: "border-border bg-muted text-muted-foreground",
  failed: "border-destructive/40 bg-destructive/10 text-destructive",
};

function isTerminal(status) {
  return status === "stopped" || status === "failed";
}

function formatTime(value) {
  const text = asText(value);
  if (!text) {
    return "";
  }
  const date = new Date(text);
  if (Number.isNaN(date.getTime())) {
    return text;
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function AgentCard({ agent }) {
  const [draft, setDraft] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const agentRef = asText(agent.id || agent.name);
  const name = asText(agent.name) || agentRef;
  const status = asText(agent.status) || "idle";
  const thread = Array.isArray(agent.thread) ? agent.thread : [];
  const terminal = isTerminal(status);

  async function send() {
    const body = draft.trim();
    if (!body || busy || terminal) {
      return;
    }
    setBusy(true);
    try {
      await callAction({
        name: "live_agent_message",
        payload: { agent: agentRef, body },
      });
      setDraft("");
    } finally {
      setBusy(false);
    }
  }

  async function shutDown() {
    if (terminal || !window.confirm(`Shut down live agent "${name}"?`)) {
      return;
    }
    await callAction({ name: "live_agent_kill", payload: { agent: agentRef } });
  }

  return (
    <section className="w-full rounded-md border border-border bg-card text-card-foreground shadow-sm">
      <div className="flex items-start justify-between gap-2 border-b border-border px-3 py-2">
        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold leading-6">{name}</h3>
          <p className="truncate text-xs text-muted-foreground">
            {asText(agent.template) || "custom"} · {asText(agent.model) || "default"} ·{" "}
            {asText(agent.turns) || "0"} turns
          </p>
        </div>
        <span
          className={`flex-none rounded-full border px-2 py-0.5 text-xs font-medium ${
            STATUS_CLASSES[status] || STATUS_CLASSES.idle
          }`}
        >
          {status}
        </span>
      </div>
      <div className="max-h-64 space-y-2 overflow-y-auto px-3 py-2 text-sm">
        {thread.length === 0 ? (
          <p className="text-xs text-muted-foreground">No messages yet.</p>
        ) : (
          thread.map((entry, index) => (
            <div key={index} className="rounded-md bg-muted/60 px-2 py-1">
              <div className="flex justify-between gap-2 text-xs text-muted-foreground">
                <span className="truncate">
                  {asText(entry.from)} → {asText(entry.to)}
                </span>
                <span className="flex-none">{formatTime(entry.at)}</span>
              </div>
              <p className="whitespace-pre-wrap break-words leading-5">
                {asText(entry.body)}
              </p>
            </div>
          ))
        )}
      </div>
      <div className="flex gap-2 border-t border-border px-3 py-2">
        <input
          className="min-w-0 flex-1 rounded-md border border-border bg-background px-2 py-1 text-sm text-foreground"
          placeholder={terminal ? "Agent stopped" : `Message ${name}`}
          value={draft}
          disabled={terminal || busy}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send();
            }
          }}
        />
        <button
          type="button"
          className="rounded-md border border-border bg-background px-3 py-1 text-sm font-medium text-foreground transition-colors hover:bg-muted disabled:opacity-50"
          disabled={terminal || busy || !draft.trim()}
          onClick={send}
        >
          Send
        </button>
        <button
          type="button"
          className="rounded-md border border-destructive/40 px-3 py-1 text-sm font-medium text-destructive transition-colors hover:bg-destructive/10 disabled:opacity-50"
          disabled={terminal}
          onClick={shutDown}
        >
          Shut down
        </button>
      </div>
    </section>
  );
}

export default function LiveAgentsPanel() {
  const agents = Array.isArray(props.agents) ? props.agents : [];
  if (agents.length === 0) {
    return <p className="p-3 text-sm text-muted-foreground">No live agents.</p>;
  }
  return (
    <div className="flex w-full flex-col gap-3">
      {agents.map((agent) => (
        <AgentCard key={asText(agent.id || agent.name)} agent={agent} />
      ))}
    </div>
  );
}
