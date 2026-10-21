"use client";

import { FormEvent, useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Summary = { auroc?: number; auprc?: number; brier?: number; ece?: number; aurc?: number };
type Run = { name: string; clean: Summary };
type Prediction = {
  abstain: boolean;
  reason: string | null;
  p_match: number | null;
  u_image?: number | null;
  u_text?: number | null;
  w_image?: number | null;
  w_text?: number | null;
  disclaimer: string;
};

const fmt = (x?: number | null) => (x == null ? "–" : x.toFixed(3));
const REASONS: Record<string, string> = {
  missing_input: "An image and a report are both required.",
  unreadable_image: "The image could not be read.",
  low_confidence: "Calibrated probability is too close to 0.5.",
  unreliable_input: "Estimated input uncertainty is above the abstention threshold.",
};

export default function Home() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [run, setRun] = useState("");
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(`${API}/runs`)
      .then((r) => r.json())
      .then((rs: Run[]) => {
        setRuns(rs);
        if (rs.length) setRun(rs[0].name);
      })
      .catch(() => setError(`Cannot reach the API at ${API}`));
  }, []);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setError(null);
    const body = new FormData(e.currentTarget);
    const res = await fetch(`${API}/predict`, { method: "POST", body });
    if (!res.ok) return setError(`API error ${res.status}`);
    setResult(await res.json());
  }

  return (
    <>
      <h1>MediNova X research console</h1>
      <p className="muted">
        U-MERGE predicts whether a chest X-ray and a report belong to the same study, and abstains when unsure.
      </p>
      {error && <p className="muted">{error}</p>}

      <h2>Runs (clean test split, calibrated)</h2>
      <table>
        <thead>
          <tr><th>Run</th><th>AUROC</th><th>AUPRC</th><th>Brier</th><th>ECE</th><th>AURC</th></tr>
        </thead>
        <tbody>
          {runs.map((r) => (
            <tr key={r.name}>
              <td>{r.name}</td><td>{fmt(r.clean.auroc)}</td><td>{fmt(r.clean.auprc)}</td>
              <td>{fmt(r.clean.brier)}</td><td>{fmt(r.clean.ece)}</td><td>{fmt(r.clean.aurc)}</td>
            </tr>
          ))}
          {!runs.length && <tr><td colSpan={6} className="muted">No evaluated runs yet.</td></tr>}
        </tbody>
      </table>

      <h2>Try a pair (synthetic or permitted examples only)</h2>
      <form onSubmit={onSubmit}>
        <select name="run" value={run} onChange={(e) => setRun(e.target.value)}>
          {runs.map((r) => <option key={r.name}>{r.name}</option>)}
        </select>
        <input type="file" name="image" accept="image/png,image/jpeg" />
        <textarea name="report" rows={5} placeholder="Report text (Findings and Impression)" />
        <select name="coverage" defaultValue="0.9">
          {["1.0", "0.9", "0.8", "0.7"].map((c) => <option key={c} value={c}>Target coverage {c}</option>)}
        </select>
        <button type="submit" disabled={!run}>Predict</button>
      </form>

      {result && (
        <div className={`result ${result.abstain ? "abstain" : ""}`}>
          <strong>
            {result.abstain
              ? `Abstained: ${REASONS[result.reason ?? ""] ?? result.reason}`
              : `P(match) = ${fmt(result.p_match)}`}
          </strong>
          <table>
            <tbody>
              <tr><td>Calibrated P(match)</td><td>{fmt(result.p_match)}</td></tr>
              <tr><td>Uncertainty: image / text</td><td>{fmt(result.u_image)} / {fmt(result.u_text)}</td></tr>
              <tr><td>Gate weight: image / text</td><td>{fmt(result.w_image)} / {fmt(result.w_text)}</td></tr>
            </tbody>
          </table>
          <p className="muted">{result.disclaimer}</p>
        </div>
      )}
    </>
  );
}
