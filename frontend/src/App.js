import { useEffect, useRef, useState } from "react";
import "./App.css";

function formatStepTitle(key) {
  return key
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

function parseDecision(decision) {
  if (!decision || typeof decision !== "string") return [];
  return decision.split("+").map((s) => s.trim()).filter(Boolean);
}

function App() {
  const fileInputRef = useRef(null);
  const [steps, setSteps] = useState({});
  const [stepOrder, setStepOrder] = useState([]);
  const [decision, setDecision] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [fileName, setFileName] = useState("");
  const [cacheBust, setCacheBust] = useState(0);
  const [lightbox, setLightbox] = useState(null);
  const closeBtnRef = useRef(null);

  const orderedKeys =
    stepOrder.length > 0 ? stepOrder.filter((k) => steps[k]) : Object.keys(steps);

  const stepImageUrl = (key) => {
    const path = steps[key];
    if (!path) return "";
    return `${path}?v=${cacheBust}`;
  };

  useEffect(() => {
    if (!lightbox) return;
    const onKey = (e) => {
      if (e.key === "Escape") setLightbox(null);
    };
    document.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    closeBtnRef.current?.focus();
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [lightbox]);

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setError("");
    setLoading(true);
    setSteps({});
    setStepOrder([]);
    setDecision("");
    setFileName(file.name);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const res = await fetch("/process", {
        method: "POST",
        body: formData,
      });

      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        setError(data.error || `Request failed (${res.status})`);
        return;
      }

      setSteps(data.steps || {});
      setStepOrder(Array.isArray(data.step_order) ? data.step_order : []);
      setDecision(data.decision || "");
      setCacheBust(Date.now());
    } catch (err) {
      setError(
        err.message || "Network error. Is the backend running on port 5000?"
      );
    } finally {
      setLoading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const decisionParts = parseDecision(decision);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Adaptive document enhancement</h1>
        <p>
          Upload a scan or photo. The service picks a denoise / contrast /
          threshold path from image features, then shows each stage.
        </p>
      </header>

      <section className="upload-card" aria-label="Upload">
        <div className="upload-zone">
          <label className="upload-zone-label">
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleUpload}
              disabled={loading}
            />
            {loading ? "Processing…" : "Choose image"}
          </label>
          <div className="upload-meta">
            {fileName ? (
              <>
                <span>
                  Last file: <strong>{fileName}</strong>
                </span>
                <span>PNG, JPG, or other common formats</span>
              </>
            ) : (
              <span>No file selected yet</span>
            )}
          </div>
        </div>
        <div className="status-row">
          {loading && (
            <>
              <span className="spinner" aria-hidden />
              <span style={{ color: "#6b7280", fontSize: "0.9rem" }}>
                Running pipeline…
              </span>
            </>
          )}
        </div>
        {error ? (
          <p className="error-banner" role="alert">
            {error}
          </p>
        ) : null}
      </section>

      {(decision || decisionParts.length > 0) && (
        <section className="decision-panel" aria-label="Chosen pipeline">
          <h2>Model decision</h2>
          {decisionParts.length > 0 ? (
            <div className="decision-chips">
              {decisionParts.map((part) => (
                <span key={part} className="chip">
                  {formatStepTitle(part)}
                </span>
              ))}
            </div>
          ) : (
            <span className="chip chip-muted">—</span>
          )}
        </section>
      )}

      {orderedKeys.length === 0 && !loading ? (
        <div className="empty-hint">
          Choose an image to see original, deskew, grayscale, and enhancement
          steps side by side.
        </div>
      ) : (
        <>
          <section className="gallery-toolbar" aria-label="View all step images">
            <p className="gallery-toolbar-hint">
              Click any image for a large preview. Use the links to open each PNG in a
              new browser tab.
            </p>
            <ul className="gallery-link-list">
              {orderedKeys.map((key) => (
                <li key={key}>
                  <a
                    href={stepImageUrl(key)}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    {formatStepTitle(key)}
                  </a>
                </li>
              ))}
            </ul>
          </section>

          <div className="steps-grid">
            {orderedKeys.map((key) => {
              const src = stepImageUrl(key);
              const title = formatStepTitle(key);
              return (
                <article key={key} className="step-card">
                  <div className="step-card-header">
                    <span className="step-card-title">{title}</span>
                    <a
                      className="step-card-tab"
                      href={src}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Open in new tab
                    </a>
                  </div>
                  <div className="step-card-body">
                    <button
                      type="button"
                      className="step-thumb"
                      onClick={() => setLightbox({ src, title })}
                      aria-label={`View ${title} full size`}
                    >
                      <img src={src} alt="" loading="lazy" />
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        </>
      )}

      {lightbox ? (
        <div
          className="lightbox-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label={lightbox.title}
          onClick={() => setLightbox(null)}
        >
          <button
            ref={closeBtnRef}
            type="button"
            className="lightbox-close"
            aria-label="Close preview"
            onClick={(e) => {
              e.stopPropagation();
              setLightbox(null);
            }}
          >
            Close
          </button>
          <div
            className="lightbox-frame"
            onClick={(e) => e.stopPropagation()}
          >
            <p className="lightbox-caption">{lightbox.title}</p>
            <a
              className="lightbox-open-tab"
              href={lightbox.src}
              target="_blank"
              rel="noopener noreferrer"
            >
              Open this image in a new tab
            </a>
            <img
              className="lightbox-img"
              src={lightbox.src}
              alt={lightbox.title}
            />
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default App;
