import { useEffect, useMemo, useRef, useState } from "react";
import { STAGES, STYLES, request } from "./api.js";

const ACTIVE = new Set(STAGES);

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 7h16" />
      <path d="M9 7V5h6v2" />
      <path d="M8 7l1 12h6l1-12" />
    </svg>
  );
}

function runView(detail) {
  const status = detail?.status;
  const job = detail?.job?.state;
  const assemble = status?.stages?.assemble;
  const stopped = job === "stopped" || status?.current_stage === "stopped";
  const finished = !stopped && (job === "done" || assemble === "done" || status?.current_stage === "done");
  const waiting = !finished && !stopped && (job === "awaiting_approval" || status?.meta?.awaiting_approval);
  const failed = job === "failed" || assemble === "failed";
  const working = !finished && !stopped && !waiting && !failed && (
    job === "running" || job === "queued" || job === "stopping"
    || ACTIVE.has(status?.current_stage)
  );
  const stages = status?.stages || {};
  const showShots = stages.storyboard === "done" && (
    waiting || assemble === "failed" || finished || status?.meta?.needs_rebuild
    || stages.stills === "done" || stages.stills === "running"
  );
  const label = stopped ? "stopped" : finished ? "done" : (job || status?.current_stage || "");
  return { status, job, assemble, stopped, finished, waiting, failed, working, showShots, label };
}

export default function App() {
  const [health, setHealth] = useState({});
  const [runs, setRuns] = useState([]);
  const [runId, setRunId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [board, setBoard] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [showCut, setShowCut] = useState(false);
  const [videoStamp, setVideoStamp] = useState(0);
  const [spokenDraft, setSpokenDraft] = useState("");
  const [ask, setAsk] = useState(null);
  const [notice, setNotice] = useState("");
  const confirm = (title, body, okLabel) => new Promise((resolve) => {
    setAsk({ title, body, okLabel, resolve });
  });

  const view = useMemo(() => runView(detail), [detail]);
  const shots = board?.shots || [];
  const selected = shots.find((shot) => shot.id === selectedId) || null;

  useEffect(() => {
    let stop = false;
    const load = () => request("/api/health").then((h) => { if (!stop) setHealth(h); }).catch(() => {});
    load();
    const timer = setInterval(load, 15000);
    return () => { stop = true; clearInterval(timer); };
  }, []);

  function refreshRuns() {
    request("/api/runs").then(setRuns).catch(() => {});
  }

  useEffect(() => { refreshRuns(); }, []);

  useEffect(() => {
    if (!runId) return undefined;
    let stop = false;
    const tick = async () => {
      try {
        const data = await request(`/api/runs/${runId}`);
        if (!stop) setDetail(data);
      } catch (err) {
        if (!stop) setNotice(err.message);
      }
    };
    tick();
    const timer = setInterval(tick, 2000);
    return () => { stop = true; clearInterval(timer); };
  }, [runId]);

  useEffect(() => {
    if (!runId || !view.showShots) return undefined;
    let stop = false;
    request(`/api/runs/${runId}/storyboard`)
      .then((next) => { if (!stop) setBoard(next); })
      .catch(() => { if (!stop) setBoard(null); });
    return () => { stop = true; };
  }, [runId, view.showShots, detail]);

  useEffect(() => {
    setBoard(null);
    setSelectedId(null);
    setShowCut(false);
    setVideoStamp(0);
    setSpokenDraft("");
    setNotice("");
  }, [runId]);

  useEffect(() => {
    if (shots.length && !shots.some((shot) => shot.id === selectedId)) setSelectedId(shots[0].id);
  }, [shots, selectedId]);

  const settled = useRef(false);
  useEffect(() => {
    settled.current = false;
  }, [runId]);

  useEffect(() => {
    if (view.finished && detail?.final_ready) setVideoStamp((stamp) => stamp || Date.now());
    const done = view.finished || view.stopped || view.failed;
    if (done && !settled.current) refreshRuns();
    settled.current = done;
  }, [view.finished, view.stopped, view.failed, detail?.final_ready]);

  async function openRun(id) {
    setRunId(id);
  }

  async function removeRun(run) {
    const name = run.topic ? `${run.run_id} (${run.topic})` : run.run_id;
    const ok = await confirm("Delete this run?", `${name} and its files will be removed.`, "Delete");
    if (!ok) return;
    try {
      await request(`/api/runs/${run.run_id}`, { method: "DELETE" });
      if (runId === run.run_id) setRunId(null);
      refreshRuns();
    } catch (err) {
      await confirm("Could not delete this run", err.message, "OK");
    }
  }

  async function startRun(body) {
    setNotice("");
    const data = await request("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    setRunId(data.run_id);
    refreshRuns();
  }

  async function stopRun() {
    try {
      await request(`/api/runs/${runId}/stop`, { method: "POST" });
    } catch (err) {
      setNotice(err.message);
    }
  }

  async function approve() {
    try {
      await request(`/api/runs/${runId}/approve`, { method: "POST" });
    } catch (err) {
      setNotice(err.message);
    }
  }

  async function rebuild() {
    try {
      await request(`/api/runs/${runId}/resume`, { method: "POST" });
      setShowCut(false);
    } catch (err) {
      setNotice(err.message);
    }
  }

  async function setMotion(motion) {
    try {
      await request(`/api/runs/${runId}/options`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ motion }),
      });
    } catch (err) {
      setNotice(err.message);
    }
  }

  async function applyDensity(density) {
    const ok = await confirm(
      "Change how many stills?",
      "This redraws the storyboard and the pictures.",
      "Redraw",
    );
    if (!ok) return;
    try {
      await request(`/api/runs/${runId}/options`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          motion: detail?.status?.meta?.motion !== false,
          still_density: density,
          replan: true,
        }),
      });
    } catch (err) {
      setNotice(err.message);
    }
  }

  const asset = (rel) => (rel ? `/api/runs/${runId}/asset?rel=${encodeURIComponent(rel)}` : "");
  const previewRel = selected?.frame_path || selected?.draft_path;
  const showVideo = showCut && view.finished && detail?.final_ready && videoStamp;
  const line = spokenDraft || selected?.narration_text || "";

  return (
    <div className="app">
      <header className="top">
        <span className="mark">Studio</span>
        <span className="dots" title="Services">
          {Object.entries(health).map(([name, ok]) => (
            <span key={name} className={`dot ${ok ? "on" : "off"}`} title={`${name}: ${ok ? "ok" : "down"}`} />
          ))}
        </span>
        <span className="muted grow">{runId ? stepLine(view, shots) : "New video"}</span>
        <div className="actions">
          {view.working && (
            <button className="btn" type="button" onClick={stopRun} disabled={view.job === "stopping"}>
              {view.job === "stopping" ? "Stopping…" : "Stop"}
            </button>
          )}
          {view.waiting && <button className="btn primary" type="button" onClick={approve}>Approve drafts</button>}
          {(detail?.status?.meta?.needs_rebuild || view.assemble === "failed") && (
            <button className="btn" type="button" onClick={rebuild}>Rebuild</button>
          )}
          {view.finished && detail?.final_ready && videoStamp ? (
            <a className="btn" href={`/api/runs/${runId}/video?t=${videoStamp}`}>Download</a>
          ) : null}
        </div>
      </header>

      {runId && <StageRail stages={view.status?.stages} shots={shots} />}

      <aside className="side">
        <div className="side-head">
          Projects
          <button className="btn" type="button" onClick={() => setRunId(null)}>New</button>
        </div>
        <div className="runs">
          {runs.map((run) => (
            <div key={run.run_id} className={`run-row ${run.run_id === runId ? "on" : ""}`}>
              <button type="button" className="run-main" onClick={() => openRun(run.run_id)}>
                <strong>{run.topic || "Untitled"}</strong>
                <small>{run.final_ready ? "Ready" : run.run_id.slice(0, 15)}</small>
              </button>
              <button className="icon" type="button" aria-label={`Delete ${run.topic || run.run_id}`} onClick={() => removeRun(run)}>
                <TrashIcon />
              </button>
            </div>
          ))}
          {!runs.length && <p className="muted pad">No videos yet.</p>}
        </div>
      </aside>

      <section className="stage">
        {notice && <p className="err">{notice}</p>}
        <div className="preview">
          {!runId && <Composer onStart={startRun} onError={setNotice} />}
          {runId && showVideo && (
            <video key={videoStamp} src={`/api/runs/${runId}/video?t=${videoStamp}`} controls autoPlay />
          )}
          {runId && !showVideo && previewRel && <img src={asset(previewRel)} alt={selected?.id || "still"} />}
          {runId && !showVideo && !previewRel && (
            <p className="muted">{view.label ? `Working on ${view.label}` : "Waiting for the first picture"}</p>
          )}
        </div>
        {runId && line && <p className="caption">{line}</p>}
      </section>

      <aside className="inspector">
        {!runId && <div className="pad muted">Pick a project, or start a new video in the center.</div>}
        {runId && !selected && (
          <div className="pad">
            <p className="muted">{detail?.status?.topic || runId}</p>
            {detail?.job?.error || detail?.status?.error ? <p className="err">{detail.job?.error || detail.status?.error}</p> : null}
          </div>
        )}
        {runId && selected && (
          <Inspector
            key={selected.id}
            shot={selected}
            runId={runId}
            meta={detail?.status?.meta}
            onSpoken={setSpokenDraft}
            onSaved={() => setBoard(null)}
            onMotion={setMotion}
            onDensity={applyDensity}
            onError={setNotice}
          />
        )}
        {runId && view.finished && detail?.final_ready && (
          <div className="pad">
            <button className="btn" type="button" onClick={() => setShowCut((value) => !value)}>
              {showCut ? "Show the still" : "Play the cut"}
            </button>
          </div>
        )}
      </aside>

      <footer className="film">
        <div className="film-head">
          <span>Stills</span>
        </div>
        <div className="strip">
          {shots.map((shot) => {
            const rel = shot.frame_path || shot.draft_path;
            return (
              <button key={shot.id} type="button" className={`frame ${shot.id === selectedId ? "on" : ""}`} onClick={() => { setSelectedId(shot.id); setShowCut(false); }}>
                {rel ? <img src={asset(rel)} alt="" /> : <span className="ph" />}
                <span>{shot.narration_text || shot.id}</span>
              </button>
            );
          })}
        </div>
      </footer>

      {ask && (
        <Confirm
          title={ask.title}
          body={ask.body}
          okLabel={ask.okLabel}
          onClose={(value) => {
            ask.resolve(value);
            setAsk(null);
          }}
        />
      )}
    </div>
  );
}

function stepCount(name, shots) {
  if (!shots.length) return "";
  if (name === "stills") {
    const finals = shots.filter((shot) => shot.frame_path).length;
    const drafts = shots.filter((shot) => shot.draft_path).length;
    const done = finals || drafts;
    return done ? `${finals || drafts}/${shots.length}` : "";
  }
  if (name === "qa") {
    const done = shots.filter((shot) => shot.qa_ok !== null && shot.qa_ok !== undefined).length;
    return done ? `${done}/${shots.length}` : "";
  }
  if (name === "motion") {
    const done = shots.filter((shot) => shot.clip_path).length;
    return done ? `${done}/${shots.length}` : "";
  }
  return "";
}

function stepLine(view, shots) {
  const name = view.status?.current_stage;
  if (!name || name === "done") return view.label || "done";
  const state = view.status?.stages?.[name] || view.label;
  const count = stepCount(name, shots);
  return [name, count, state].filter(Boolean).join(" · ");
}

function StageRail({ stages, shots }) {
  const map = stages || {};
  return (
    <div className="rail">
      {STAGES.map((name) => {
        const state = map[name] || "pending";
        const count = stepCount(name, shots);
        return (
          <span key={name} className={`chip ${state}`}>
            {name}{count ? ` ${count}` : ""} · {state}
          </span>
        );
      })}
    </div>
  );
}

function Composer({ onStart, onError }) {
  const [topic, setTopic] = useState("");
  const [duration, setDuration] = useState(45);
  const [style, setStyle] = useState("sketch");
  const [pick, setPick] = useState(false);
  const [skipQa, setSkipQa] = useState(false);
  const [review, setReview] = useState(true);
  const [motion, setMotion] = useState(true);
  const [density, setDensity] = useState("standard");
  const [busy, setBusy] = useState(false);

  async function submit(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await onStart({
        topic: topic.trim() || null,
        pick_topic: pick,
        duration_s: Number(duration),
        style_preset: style,
        skip_qa: skipQa,
        await_approval: review,
        motion,
        still_density: density,
      });
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <h1>New video</h1>
      <p className="muted">One topic. The studio researches it, draws the stills, and cuts the narration.</p>
      <label htmlFor="topic">Topic</label>
      <textarea id="topic" value={topic} onChange={(event) => setTopic(event.target.value)} placeholder="Why octopus arms can taste what they touch" />
      <div className="row">
        <div>
          <label htmlFor="duration">Seconds</label>
          <input id="duration" type="number" min="20" max="360" value={duration} onChange={(event) => setDuration(event.target.value)} />
        </div>
        <div>
          <label htmlFor="style">Look</label>
          <select id="style" value={style} onChange={(event) => setStyle(event.target.value)}>
            {STYLES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select>
        </div>
      </div>
      <div className="checks">
        <label><input type="checkbox" checked={pick} onChange={(event) => setPick(event.target.checked)} /> Pick a topic</label>
        <label><input type="checkbox" checked={review} onChange={(event) => setReview(event.target.checked)} /> Pause for review</label>
        <label><input type="checkbox" checked={motion} onChange={(event) => setMotion(event.target.checked)} /> Motion</label>
        <label><input type="checkbox" checked={skipQa} onChange={(event) => setSkipQa(event.target.checked)} /> Skip picture check</label>
      </div>
      <label htmlFor="density">How many stills</label>
      <select id="density" value={density} onChange={(event) => setDensity(event.target.value)}>
        <option value="fewer">Fewer, held longer</option>
        <option value="standard">Standard</option>
        <option value="more">More, shorter holds</option>
      </select>
      <div className="actions" style={{ marginTop: "0.9rem" }}>
        <button className="btn primary" type="submit" disabled={busy || (!topic.trim() && !pick)}>
          {busy ? "Starting…" : "Generate"}
        </button>
      </div>
    </form>
  );
}

function Inspector({ shot, runId, meta, onSpoken, onSaved, onMotion, onDensity, onError }) {
  const formRef = useRef(null);
  const [narration, setNarration] = useState(shot.narration_text || "");
  const [overlay, setOverlay] = useState(shot.overlay_text || "");
  const [prompt, setPrompt] = useState(shot.visual_prompt || "");
  const [busy, setBusy] = useState(false);
  const [density, setDensity] = useState(meta?.still_density || "standard");
  const [motionOn, setMotionOn] = useState(meta?.motion !== false);
  const serverKey = [shot.narration_text, shot.overlay_text, shot.visual_prompt].join("\u0001");

  useEffect(() => { onSpoken(narration); }, [narration, onSpoken]);

  useEffect(() => {
    if (formRef.current?.contains(document.activeElement)) return;
    setNarration(shot.narration_text || "");
    setOverlay(shot.overlay_text || "");
    setPrompt(shot.visual_prompt || "");
  }, [serverKey, shot.narration_text, shot.overlay_text, shot.visual_prompt]);

  useEffect(() => {
    if (document.activeElement?.id === "run-density") return;
    setDensity(meta?.still_density || "standard");
  }, [meta?.still_density]);

  useEffect(() => {
    if (document.activeElement?.id === "run-motion") return;
    setMotionOn(meta?.motion !== false);
  }, [meta?.motion]);

  async function save(event) {
    event.preventDefault();
    setBusy(true);
    try {
      await request(`/api/runs/${runId}/shots/${shot.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ visual_prompt: prompt, narration_text: narration, overlay_text: overlay }),
      });
      onSaved();
    } catch (err) {
      onError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form ref={formRef} onSubmit={save}>
      <div className="inspector-head">{shot.id}{shot.hero ? " · hero" : ""}</div>
      <label htmlFor="spoken">Spoken with this still</label>
      <textarea id="spoken" value={narration} onChange={(event) => setNarration(event.target.value)} />
      <label htmlFor="overlay">Words printed on the picture</label>
      <input id="overlay" value={overlay} placeholder="Leave empty when the picture needs no words" onChange={(event) => setOverlay(event.target.value)} />
      <label htmlFor="prompt">Picture prompt</label>
      <textarea id="prompt" value={prompt} onChange={(event) => setPrompt(event.target.value)} />
      <div className="actions" style={{ marginTop: "0.8rem" }}>
        <button className="btn primary" type="submit" disabled={busy}>{busy ? "Updating…" : "Update still"}</button>
      </div>
      <div className="checks">
        <label>
          <input id="run-motion" type="checkbox" checked={motionOn} onChange={(event) => { setMotionOn(event.target.checked); onMotion(event.target.checked); }} />
          Motion
        </label>
      </div>
      <label htmlFor="run-density">How many stills</label>
      <select id="run-density" value={density} onChange={(event) => setDensity(event.target.value)}>
        <option value="fewer">Fewer</option>
        <option value="standard">Standard</option>
        <option value="more">More</option>
      </select>
      <div className="actions" style={{ marginTop: "0.6rem" }}>
        <button className="btn" type="button" onClick={() => onDensity(density)}>Apply still count</button>
      </div>
      {meta?.needs_rebuild && <p className="muted">Changes saved. Rebuild the video to apply them.</p>}
    </form>
  );
}

function Confirm({ title, body, okLabel, onClose }) {
  useEffect(() => {
    const onKey = (event) => { if (event.key === "Escape") onClose(false); };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal" onClick={(event) => { if (event.target === event.currentTarget) onClose(false); }}>
      <div className="modal-card" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
        <h2 id="confirm-title">{title}</h2>
        <p>{body}</p>
        <div className="modal-actions">
          <button className="btn" type="button" onClick={() => onClose(false)}>Cancel</button>
          <button className="btn quiet-danger" type="button" onClick={() => onClose(true)}>{okLabel}</button>
        </div>
      </div>
    </div>
  );
}
