/**
 * ACP compatibility bridge for the Antigravity CLI (`agy`).
 *
 * Antigravity exposes no native agent protocol. This module speaks the slice
 * of ACP that `AcpSessionRuntime` uses over stdio and executes each turn
 * through Antigravity's documented non-interactive `--print` mode, while
 * reconstructing a live event stream from hooks and the trajectory transcript
 * (see `agyEvents.ts` and `agyTranscript.ts`).
 *
 * Runs as a subcommand of the server binary (`t3 agy-acp`) so it ships inside
 * the same bundle rather than as a loose script.
 *
 * @module provider/acp/antigravity/agyBridge
 */
// @effect-diagnostics nodeBuiltinImport:off - Standalone stdio bridge process, not an Effect runtime.
// @effect-diagnostics globalTimers:off - Polls Antigravity hook output outside any Effect runtime.
import * as NodeChildProcess from "node:child_process";
import * as NodeCrypto from "node:crypto";
import * as NodeFS from "node:fs";
import * as NodeOS from "node:os";
import * as NodePath from "node:path";

import packageJson from "../../../../package.json" with { type: "json" };
import {
  agyHookResponse,
  agyTargetPath,
  agyToolKind,
  hookSessionUpdate,
  makeAgyTurnState,
  type AgyHookEvent,
  type AgyHookPayload,
  type AgySessionUpdate,
  type AgyTurnState,
} from "./agyEvents.ts";
import {
  AgyTranscriptCursor,
  parseTranscriptLine,
  transcriptRecordUpdates,
} from "./agyTranscript.ts";

const HOOK_DIR_ENV = "T3_AGY_HOOK_DIR";
const HOOK_POLL_INTERVAL_MS = 50;
const DEFAULT_PRINT_TIMEOUT = "2h";
const HOOKS_KEY = "t3code-antigravity-observer";

// Cache the resolved agy binary path so per-turn filesystem searches
// (120+ statSync calls across PATH directories × 4 extensions) only happen once.
let cachedAgyCommand: { command: string; shell: boolean } | null = null;

interface BridgeSession {
  readonly cwd: string;
  systemPrompt: string | undefined;
  conversationId: string | undefined;
  /** Persistent hook workspace for this session — created once, reused across turns. */
  hookWorkspace: string | undefined;
  mode?: string | undefined;
  model?: string | undefined;
  effort?: string | undefined;
}

const sessions = new Map<string, BridgeSession>();

// ── Session id ⇄ Antigravity conversation id ──────────────────────────
//
// `session/new` must return an id before the first turn runs, which is before
// Antigravity has created a trajectory. The mapping is persisted so a later
// `session/load` — potentially in a fresh bridge process — can still resume
// the right conversation.

function getAgyAppDataDir(): string {
  return (
    process.env["T3_AGY_APP_DATA_DIR"]?.trim() ||
    NodePath.join(NodeOS.homedir(), ".gemini", "antigravity-cli")
  );
}

function stateFilePath(): string {
  return NodePath.join(getAgyAppDataDir(), "t3code-acp-sessions.json");
}

function findExistingTranscriptPath(conversationId: string): string | undefined {
  const trimmed = conversationId.trim();
  if (!trimmed) return undefined;
  const baseDir = NodePath.join(getAgyAppDataDir(), "brain", trimmed, ".system_generated", "logs");
  const condensed = NodePath.join(baseDir, "transcript.jsonl");
  if (NodeFS.existsSync(condensed)) {
    return condensed;
  }
  const full = NodePath.join(baseDir, "transcript_full.jsonl");
  if (NodeFS.existsSync(full)) {
    return full;
  }
  return undefined;
}

function readSessionMap(): Record<string, string> {
  try {
    const parsed: unknown = JSON.parse(NodeFS.readFileSync(stateFilePath(), "utf8"));
    return typeof parsed === "object" && parsed !== null ? (parsed as Record<string, string>) : {};
  } catch {
    return {};
  }
}

function persistConversationId(sessionId: string, conversationId: string): void {
  try {
    const target = stateFilePath();
    NodeFS.mkdirSync(NodePath.dirname(target), { recursive: true });
    const map = readSessionMap();
    if (map[sessionId] === conversationId) {
      return;
    }
    map[sessionId] = conversationId;
    NodeFS.writeFileSync(target, JSON.stringify(map, null, 2));
  } catch {
    // Losing the mapping costs conversation continuity on the next resume,
    // which is not worth failing a turn over.
  }
}

/**
 * Map a bridge session id to the Antigravity conversation it should resume.
 *
 * The persisted map is the only authority. Bridge session ids are themselves
 * random UUIDs, so an id that merely looks like a conversation id is
 * indistinguishable from one the bridge minted — falling back to the shape of
 * the string would make `session/load` resume a conversation that never
 * existed whenever the map is missing or unreadable. Returning `undefined`
 * starts a fresh conversation, which is the recoverable outcome.
 */
function lookupConversationId(sessionId: string): string | undefined {
  return readSessionMap()[sessionId]?.trim() || undefined;
}

// ── JSON-RPC plumbing ─────────────────────────────────────────────────

function writeMessage(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

function sendResult(id: unknown, result: unknown): void {
  writeMessage({ jsonrpc: "2.0", id, result });
}

function sendError(id: unknown, code: number, message: string): void {
  writeMessage({ jsonrpc: "2.0", id, error: { code, message } });
}

function sendSessionUpdate(sessionId: string, update: AgySessionUpdate): void {
  writeMessage({
    jsonrpc: "2.0",
    method: "session/update",
    params: { sessionId, update },
  });
}

// ── Hook observer ─────────────────────────────────────────────────────

/**
 * Hook command the bridge registers with Antigravity. Re-invokes this same
 * binary so the observer always matches the running bridge.
 */
function hookCommandFor(event: string): string {
  const entry = process.argv[1];
  const base = entry
    ? `${quoteArg(process.execPath)} --no-warnings ${quoteArg(entry)}`
    : quoteArg(process.execPath);
  return `${base} agy-hook --event ${event}`;
}

function quoteArg(value: string): string {
  return /[\s"']/.test(value) ? `"${value.replace(/(["\\$`])/g, "\\$1")}"` : value;
}

/**
 * Build a throwaway workspace directory whose only purpose is carrying
 * `.agents/hooks.json`. Antigravity loads `.agents` from every `--add-dir`
 * path, so the observer attaches without writing anything into the user's
 * repository.
 */
function createHookWorkspace(): string {
  const dir = NodeFS.mkdtempSync(NodePath.join(NodeOS.tmpdir(), "t3-agy-hooks-"));
  const agentsDir = NodePath.join(dir, ".agents");
  NodeFS.mkdirSync(agentsDir, { recursive: true });
  const toolHook = (event: string) => [
    { matcher: "*", hooks: [{ type: "command", command: hookCommandFor(event), timeout: 10 }] },
  ];
  NodeFS.writeFileSync(
    NodePath.join(agentsDir, "hooks.json"),
    JSON.stringify(
      {
        [HOOKS_KEY]: {
          PreToolUse: toolHook("pre-tool-use"),
          PostToolUse: toolHook("post-tool-use"),
          Stop: [{ type: "command", command: hookCommandFor("stop"), timeout: 10 }],
        },
      },
      null,
      2,
    ),
  );
  return dir;
}

function readHookEvents(hookDir: string, seen: Set<string>): ReadonlyArray<AgyHookEvent> {
  let entries: Array<string>;
  try {
    entries = NodeFS.readdirSync(hookDir);
  } catch {
    return [];
  }
  const events: Array<AgyHookEvent> = [];
  for (const name of entries.filter((n) => n.endsWith(".json")).sort()) {
    if (seen.has(name)) {
      continue;
    }
    seen.add(name);
    try {
      const raw = NodeFS.readFileSync(NodePath.join(hookDir, name), "utf8");
      const parsed: unknown = JSON.parse(raw);
      if (typeof parsed === "object" && parsed !== null) {
        events.push(parsed as AgyHookEvent);
      }
    } catch {
      // A half-written hook file is picked up on the next poll.
      seen.delete(name);
    }
  }
  return events;
}

/**
 * Largest file the hook will inline into its event record. A diff of anything
 * bigger is not worth the memory it would cost on both sides.
 */
const MAX_CAPTURED_FILE_BYTES = 2 * 1024 * 1024;

function captureFileText(path: string | undefined): string | null {
  if (!path) {
    return null;
  }
  try {
    const stats = NodeFS.statSync(path);
    if (!stats.isFile() || stats.size > MAX_CAPTURED_FILE_BYTES) {
      return null;
    }
    return NodeFS.readFileSync(path, "utf8");
  } catch {
    // A new file has no prior contents; that is a valid diff with no oldText.
    return null;
  }
}

/** Entry point for `t3 agy-hook <event>`. */
export async function runAgyHook(event: string): Promise<void> {
  let raw = "";
  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) {
    raw += chunk;
  }

  const hookDir = process.env[HOOK_DIR_ENV]?.trim();
  if (hookDir) {
    try {
      const payload = JSON.parse(raw) as AgyHookPayload;
      const record: AgyHookEvent = {
        event,
        payload,
        // Snapshot the file here, while the hook still brackets the tool call.
        ...(agyToolKind(payload?.toolCall?.name) === "edit"
          ? { capturedFileText: captureFileText(agyTargetPath(payload?.toolCall)) }
          : {}),
      };
      const name = `${process.hrtime.bigint().toString().padStart(24, "0")}-${event}.json`;
      // Write then rename so the poller never observes a partial file.
      const finalPath = NodePath.join(hookDir, name);
      const tempPath = `${finalPath}.tmp`;
      NodeFS.writeFileSync(tempPath, JSON.stringify(record));
      NodeFS.renameSync(tempPath, finalPath);
    } catch {
      // Observation is best-effort: a hook must never break a tool call.
    }
  }

  process.stdout.write(JSON.stringify(agyHookResponse(event, Boolean(hookDir))));
}

// ── Turn execution ────────────────────────────────────────────────────

function buildAgyArgs(
  session: BridgeSession,
  hookWorkspace: string,
  prompt: string,
): Array<string> {
  const args = [
    "--dangerously-skip-permissions",
    "--print-timeout",
    process.env["T3_AGY_PRINT_TIMEOUT"]?.trim() || DEFAULT_PRINT_TIMEOUT,
  ];
  let model = session.model?.trim() || process.env["T3_AGY_MODEL"]?.trim();
  let effort = session.effort?.trim() || process.env["T3_AGY_EFFORT"]?.trim();

  if (model) {
    const tierMatch = /^(.*)-(high|medium|low)$/.exec(model);
    if (tierMatch) {
      const baseModel = tierMatch[1]!;
      const modelEffort = tierMatch[2]!;
      if (!effort) {
        effort = modelEffort;
      }
      model = baseModel;
    }
  }

  if (model) {
    args.push("--model", model);
  }
  if (effort) {
    args.push("--effort", effort);
  }
  const mode = session.mode?.trim() || process.env["T3_AGY_MODE"]?.trim();
  if (mode) {
    args.push("--mode", mode === "plan" ? "plan" : "accept-edits");
  }
  if (session.conversationId) {
    args.push("--conversation", session.conversationId);
  }
  // Print mode does not infer workspace customizations from cwd alone. The
  // session workspace is registered so its `.agents` skills and rules load;
  // the hook workspace is registered so the observer attaches.
  const resolvedCwd = NodePath.resolve(session.cwd);
  args.push("--add-dir", resolvedCwd, "--add-dir", hookWorkspace);
  args.push("--print", prompt);
  return args;
}

function renderPrompt(session: BridgeSession, promptBlocks: unknown): string | null {
  if (!Array.isArray(promptBlocks)) {
    return null;
  }
  const parts: string[] = [];
  for (const block of promptBlocks) {
    if (typeof block !== "object" || block === null) continue;
    const b = block as Record<string, unknown>;
    if (b["type"] === "text" && typeof b["text"] === "string") {
      parts.push(b["text"]);
    } else if (b["type"] === "image") {
      const path = typeof b["path"] === "string" ? b["path"] : typeof b["uri"] === "string" ? b["uri"] : undefined;
      if (path) {
        parts.push(`[Attached Image: ${path}]`);
      }
    }
  }
  const text = parts.join("\n\n").trim();
  if (text.length === 0) {
    return null;
  }
  const systemPrompt = session.systemPrompt?.trim();
  return systemPrompt ? `System instructions:\n${systemPrompt}\n\nRequest:\n${text}` : text;
}

interface TurnOutcome {
  readonly stopReason: "end_turn" | "cancelled";
  readonly failure?: string;
}

let activeChild: NodeChildProcess.ChildProcess | null = null;
/** Session whose turn is currently running, if any. Gates `session/cancel`. */
let activeTurnSessionId: string | null = null;
const cancelledSessions = new Set<string>();

/**
 * Drain everything Antigravity has produced so far and emit it as ACP updates.
 *
 * Hooks are read first so a tool call is always announced before the
 * transcript record carrying its output is matched against it.
 */
function drain(input: {
  readonly sessionId: string;
  readonly hookDir: string;
  readonly seenHooks: Set<string>;
  readonly state: AgyTurnState;
  readonly cursor: AgyTranscriptCursor;
  readonly transcriptOffset: { value: number };
  readonly assistantText: { emitted: boolean };
  readonly final: boolean;
}): void {
  for (const hook of readHookEvents(input.hookDir, input.seenHooks)) {
    // Diffing the file contents each hook captured, rather than the arguments
    // of the edit, keeps this correct across tools whose argument shapes
    // differ (`replace_file_content` sends a fragment, `write_to_file` sends
    // the whole file).
    const fileText = hook.capturedFileText ?? undefined;
    const update = hookSessionUpdate(hook, input.state, fileText);
    if (update) {
      sendSessionUpdate(input.sessionId, update);
    }
  }

  const transcriptPath = resolveTranscriptPath(input.state);
  if (transcriptPath) {
    let chunk = "";
    try {
      const stats = NodeFS.statSync(transcriptPath);
      if (stats.size > input.transcriptOffset.value) {
        const fd = NodeFS.openSync(transcriptPath, "r");
        try {
          const length = stats.size - input.transcriptOffset.value;
          const buffer = Buffer.alloc(length);
          NodeFS.readSync(fd, buffer, 0, length, input.transcriptOffset.value);
          chunk = buffer.toString("utf8");
          input.transcriptOffset.value = stats.size;
        } finally {
          NodeFS.closeSync(fd);
        }
      }
    } catch {
      chunk = "";
    }

    const lines = chunk.length > 0 ? input.cursor.push(chunk) : [];
    const allLines = input.final ? [...lines, ...input.cursor.flush()] : lines;
    for (const line of allLines) {
      const record = parseTranscriptLine(line);
      if (!record) {
        continue;
      }
      const result = transcriptRecordUpdates(record, input.state);
      for (const update of result.updates) {
        sendSessionUpdate(input.sessionId, update);
      }
      if (result.emittedAssistantText) {
        input.assistantText.emitted = true;
      }
    }
  }
}

/**
 * Hooks report `transcript_full.jsonl`; the sibling `transcript.jsonl` holds
 * the same steps without internal model chatter and is the better stream to
 * render.
 *
 * The choice is pinned for the rest of the turn. `transcriptOffset` and the
 * line cursor are byte positions into whichever file was picked, so switching
 * once the condensed file appears would resume reading at an offset that means
 * nothing in the new file — skipping records, or re-emitting ones already
 * streamed from the other one.
 */
function resolveTranscriptPath(state: AgyTurnState): string | undefined {
  if (state.resolvedTranscriptPath) {
    return state.resolvedTranscriptPath;
  }
  const reported = state.transcriptPath;
  if (!reported) {
    return undefined;
  }
  const condensed = reported.replace(/transcript_full\.jsonl$/, "transcript.jsonl");
  state.resolvedTranscriptPath = NodeFS.existsSync(condensed) ? condensed : reported;
  return state.resolvedTranscriptPath;
}

export function resolveAgyCommand(
  commandInput: string,
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform,
): { command: string; shell: boolean } {
  if (platform !== "win32") {
    return { command: commandInput, shell: false };
  }

  if (NodePath.isAbsolute(commandInput) || commandInput.includes("/") || commandInput.includes("\\")) {
    const ext = NodePath.extname(commandInput).toLowerCase();
    const isCmd = ext === ".cmd" || ext === ".bat";
    return { command: commandInput, shell: isCmd };
  }

  const extensions = [".exe", ".cmd", ".bat", ""];
  const pathEnv = env["PATH"] || env["Path"] || "";
  const pathDirs = pathEnv.split(";").map((p) => p.trim()).filter(Boolean);

  const localAppData =
    env["LOCALAPPDATA"]?.trim() || NodePath.join(NodeOS.homedir(), "AppData", "Local");
  const knownDirs = [
    ...pathDirs,
    NodePath.join(localAppData, "agy", "bin"),
    NodePath.join(NodeOS.homedir(), ".gemini", "antigravity", "bin"),
    NodePath.join(env["APPDATA"] || "", "Antigravity", "bin"),
  ];

  for (const dir of knownDirs) {
    for (const ext of extensions) {
      const candidate = NodePath.join(dir, `${commandInput}${ext}`);
      try {
        if (NodeFS.existsSync(candidate) && NodeFS.statSync(candidate).isFile()) {
          const isCmd =
            candidate.toLowerCase().endsWith(".cmd") || candidate.toLowerCase().endsWith(".bat");
          return { command: candidate, shell: isCmd };
        }
      } catch {
        // continue
      }
    }
  }

  return { command: commandInput, shell: false };
}

async function runTurn(
  sessionId: string,
  session: BridgeSession,
  prompt: string,
): Promise<TurnOutcome> {
  // A cancel that raced the end of an earlier turn must not decide this one.
  cancelledSessions.delete(sessionId);
  const hookDir = NodeFS.mkdtempSync(NodePath.join(NodeOS.tmpdir(), "t3-agy-hookout-"));

  // Reuse the hookWorkspace across turns in the same session — creating and
  // registering a fresh one per turn added ~100ms of per-turn overhead.
  if (!session.hookWorkspace) {
    session.hookWorkspace = createHookWorkspace();
  }
  const hookWorkspace = session.hookWorkspace;

  // Cache the resolved binary path — the path search scans 30+ directories
  // times 4 extensions per turn (120+ statSync calls) without this cache.
  const rawCommand = process.env["T3_AGY_COMMAND"]?.trim() || "agy";
  if (!cachedAgyCommand) {
    cachedAgyCommand = resolveAgyCommand(rawCommand);
  }
  const { command, shell } = cachedAgyCommand;
  const state = makeAgyTurnState(session.conversationId);
  const seenHooks = new Set<string>();
  const cursor = new AgyTranscriptCursor();

  let initialOffset = 0;
  if (session.conversationId) {
    const existingTranscript = findExistingTranscriptPath(session.conversationId);
    if (existingTranscript) {
      state.resolvedTranscriptPath = existingTranscript;
      try {
        initialOffset = NodeFS.statSync(existingTranscript).size;
      } catch {
        initialOffset = 0;
      }
    }
  }
  const transcriptOffset = { value: initialOffset };
  const assistantText = { emitted: false };

  let spawnError: Error | null = null;
  const child = NodeChildProcess.spawn(command, buildAgyArgs(session, hookWorkspace, prompt), {
    cwd: session.cwd,
    env: { ...process.env, [HOOK_DIR_ENV]: hookDir },
    stdio: ["ignore", "pipe", "pipe"],
    shell,
  });
  child.on("error", (err) => {
    spawnError = err;
  });
  activeChild = child;
  activeTurnSessionId = sessionId;

  let stdout = "";
  let stderr = "";
  child.stdout?.setEncoding("utf8");
  child.stdout?.on("data", (chunk: string) => {
    stdout += chunk;
  });
  child.stderr?.setEncoding("utf8");
  child.stderr?.on("data", (chunk: string) => {
    stderr += chunk;
  });

  const triggerDrain = () => {
    drain({
      sessionId,
      hookDir,
      seenHooks,
      state,
      cursor,
      transcriptOffset,
      assistantText,
      final: false,
    });
  };

  let hookWatcher: NodeFS.FSWatcher | null = null;
  try {
    hookWatcher = NodeFS.watch(hookDir, { recursive: true }, () => {
      triggerDrain();
    });
  } catch {
    // Fall back gracefully to interval polling
  }

  // Only run the polling interval as a fallback when fs.watch is unavailable.
  // Running both simultaneously causes redundant drain calls on every hook event.
  const poller = hookWatcher === null ? setInterval(triggerDrain, HOOK_POLL_INTERVAL_MS) : null;

  const exitCode = await new Promise<number | null>((resolve) => {
    child.on("error", () => resolve(null));
    child.on("close", (code) => resolve(code));
  });

  if (hookWatcher) {
    try {
      hookWatcher.close();
    } catch {
      // Ignore cleanup error
    }
  }
  if (poller !== null) clearInterval(poller);
  activeChild = null;
  activeTurnSessionId = null;
  drain({
    sessionId,
    hookDir,
    seenHooks,
    state,
    cursor,
    transcriptOffset,
    assistantText,
    final: true,
  });

  // Any tool still open at exit would otherwise render as spinning forever.
  for (const [, call] of state.toolCalls) {
    if (call.completed) {
      continue;
    }
    sendSessionUpdate(sessionId, {
      sessionUpdate: "tool_call_update",
      toolCallId: call.toolCallId,
      status: "failed",
      rawOutput: { isError: true, error: "Antigravity exited before the tool reported completion" },
    });
  }
  state.toolCalls.clear();

  if (state.conversationId) {
    session.conversationId = state.conversationId;
    persistConversationId(sessionId, state.conversationId);
  }

  // Fire cleanup asynchronously so the hook output dir is removed without
  // blocking the sendResult response back to the ACP client.
  setImmediate(() => cleanupDir(hookDir));
  // hookWorkspace is reused across turns and cleaned up when the session ends.

  if (cancelledSessions.delete(sessionId)) {
    return { stopReason: "cancelled" };
  }
  if (exitCode !== 0 || spawnError !== null) {
    const detail =
      stderr.trim() ||
      stdout.trim() ||
      (spawnError ? (spawnError as Error).message : `agy exited with code ${exitCode}`);
    return { stopReason: "end_turn", failure: detail };
  }

  // The transcript already streamed the assistant text. stdout is only used
  // when transcript observation produced nothing, so the reply is never
  // duplicated.
  if (!assistantText.emitted && stdout.trim().length > 0) {
    sendSessionUpdate(sessionId, {
      sessionUpdate: "agent_message_chunk",
      content: { type: "text", text: stdout.trim() },
    });
  }
  return { stopReason: "end_turn" };
}

function cleanupDir(dir: string): void {
  try {
    NodeFS.rmSync(dir, { recursive: true, force: true });
  } catch {
    // Temp directories are reclaimed by the OS.
  }
}

// ── Request dispatch ──────────────────────────────────────────────────

async function handleRequest(message: Record<string, unknown>): Promise<void> {
  const method = typeof message["method"] === "string" ? message["method"] : undefined;
  const id = message["id"];
  const params = (message["params"] ?? {}) as Record<string, unknown>;

  if (!method) {
    return;
  }

  switch (method) {
    case "initialize": {
      const requested =
        typeof params["protocolVersion"] === "number" ? params["protocolVersion"] : 1;
      sendResult(id, {
        protocolVersion: Math.min(requested, 1),
        agentCapabilities: {
          loadSession: true,
          promptCapabilities: { image: true, audio: false, embeddedContext: false },
          mcpCapabilities: { http: false, sse: false },
        },
        authMethods: [],
        // The bridge has no version of its own; it ships with the server, so
        // that is the version worth reporting. The Antigravity CLI version is
        // reported separately by the provider snapshot (`agy --version`).
        agentInfo: { name: "Antigravity", version: packageJson.version },
      });
      return;
    }
    // Antigravity manages its own Google sign-in; there is nothing for the
    // client to authenticate against, but the handshake still requires a
    // successful reply.
    case "authenticate": {
      sendResult(id, {});
      return;
    }
    case "session/new":
    case "session/load": {
      const cwd = typeof params["cwd"] === "string" ? params["cwd"] : "";
      if (!cwd || !NodePath.isAbsolute(cwd)) {
        sendError(id, -32602, `${method} requires an absolute cwd`);
        return;
      }
      const requestedSessionId =
        typeof params["sessionId"] === "string" ? params["sessionId"] : undefined;
      const sessionId =
        method === "session/load" && requestedSessionId
          ? requestedSessionId
          : NodeCrypto.randomUUID();
      // Create the hook workspace eagerly at session creation time so the first
      // turn can reuse it rather than paying the creation cost mid-turn.
      const hookWorkspace = createHookWorkspace();
      sessions.set(sessionId, {
        cwd,
        systemPrompt:
          typeof params["systemPrompt"] === "string"
            ? params["systemPrompt"]
            : process.env["T3_AGY_SYSTEM_PROMPT"]?.trim(),
        conversationId: requestedSessionId ? lookupConversationId(requestedSessionId) : undefined,
        hookWorkspace,
      });
      const defaultModes = {
        currentModeId: "accept-edits",
        availableModes: [
          { id: "accept-edits", name: "Build Mode", description: "Auto-approve edits and commands" },
          { id: "plan", name: "Plan Mode", description: "Architectural planning mode" },
        ],
      };
      sendResult(
        id,
        method === "session/load"
          ? { modes: defaultModes }
          : { sessionId, modes: defaultModes },
      );
      return;
    }
    case "session/prompt": {
      const sessionId = typeof params["sessionId"] === "string" ? params["sessionId"] : undefined;
      const session = sessionId ? sessions.get(sessionId) : undefined;
      if (!sessionId || !session) {
        sendError(id, -32602, "unknown sessionId");
        return;
      }
      const prompt = renderPrompt(session, params["prompt"]);
      if (prompt === null) {
        sendError(id, -32602, "session/prompt requires at least one text block");
        return;
      }
      const outcome = await runTurn(sessionId, session, prompt);
      if (outcome.failure) {
        sendError(id, -32000, `Antigravity turn failed: ${outcome.failure}`);
        return;
      }
      sendResult(id, { stopReason: outcome.stopReason });
      return;
    }
    case "session/set_config_option": {
      const sessionId = typeof params["sessionId"] === "string" ? params["sessionId"] : undefined;
      const configId = typeof params["configId"] === "string" ? params["configId"] : undefined;
      const value = params["value"];
      const session = sessionId ? sessions.get(sessionId) : undefined;
      if (!sessionId || !session) {
        sendError(id, -32602, "unknown sessionId");
        return;
      }
      if (configId === "mode" && typeof value === "string") {
        session.mode = value;
      } else if (configId === "effort" && typeof value === "string") {
        session.effort = value;
      } else if (configId === "model" && typeof value === "string") {
        session.model = value;
      }
      sendResult(id, { configOptions: [] });
      return;
    }
    case "session/destroy":
    case "session/close":
    case "session/stop": {
      const sessionId = typeof params["sessionId"] === "string" ? params["sessionId"] : undefined;
      if (sessionId) {
        const session = sessions.get(sessionId);
        if (session?.hookWorkspace) {
          cleanupDir(session.hookWorkspace);
        }
        sessions.delete(sessionId);
      }
      if (id !== undefined) {
        sendResult(id, {});
      }
      return;
    }
    case "session/cancel": {
      const sessionId = typeof params["sessionId"] === "string" ? params["sessionId"] : undefined;
      // Only a cancel aimed at the turn actually running can decide its stop
      // reason. Cancels bypass the request queue, so one that arrives after a
      // turn has already finished — or targets an idle session — would
      // otherwise sit in the set and mark the next successful turn cancelled.
      if (sessionId && sessionId === activeTurnSessionId) {
        cancelledSessions.add(sessionId);
        if (activeChild && activeChild.pid) {
          if (process.platform === "win32") {
            try {
              NodeChildProcess.execSync(`taskkill /F /T /PID ${activeChild.pid}`, {
                stdio: "ignore",
              });
            } catch {
              activeChild.kill("SIGTERM");
            }
          } else {
            activeChild.kill("SIGTERM");
          }
        }
      }
      if (id !== undefined) {
        sendResult(id, {});
      }
      return;
    }
    default: {
      if (id !== undefined) {
        sendError(id, -32601, `method not found: ${method}`);
      }
    }
  }
}

/** Entry point for `t3 agy-acp`. */
export async function runAgyBridge(): Promise<void> {
  let buffer = "";
  // Requests are handled strictly in order: a turn holds the agent busy, and
  // ACP clients do not pipeline prompts for one session.
  let queue: Promise<void> = Promise.resolve();

  process.stdin.setEncoding("utf8");
  for await (const chunk of process.stdin) {
    buffer += chunk;
    let newlineIndex = buffer.indexOf("\n");
    while (newlineIndex !== -1) {
      const line = buffer.slice(0, newlineIndex);
      buffer = buffer.slice(newlineIndex + 1);
      newlineIndex = buffer.indexOf("\n");
      if (line.trim().length === 0) {
        continue;
      }

      let message: Record<string, unknown>;
      try {
        const parsed: unknown = JSON.parse(line);
        if (typeof parsed !== "object" || parsed === null) {
          continue;
        }
        message = parsed as Record<string, unknown>;
      } catch {
        sendError(null, -32700, "invalid JSON");
        continue;
      }

      // Cancellation must interrupt an in-flight turn, so it bypasses the
      // queue that would otherwise make it wait for that turn to finish.
      if (message["method"] === "session/cancel") {
        void handleRequest(message);
        continue;
      }
      queue = queue.then(() => handleRequest(message)).catch(() => undefined);
    }
  }

  await queue;
  activeChild?.kill("SIGTERM");
  // Clean up any hook workspaces that were created for sessions that are
  // still open when the bridge process exits (e.g. server restart).
  for (const session of sessions.values()) {
    if (session.hookWorkspace) {
      cleanupDir(session.hookWorkspace);
    }
  }
}
