// Tells the user when the audio engine stopped unexpectedly since they last looked, and where the
// crash report is (to attach to a bug report). Reports come from <data dir>/crashes.
import { useEffect, useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import { bridge } from '../engine/bridge';
import { t } from '../i18n';

const SEEN_KEY = 'tonesphere.crashSeen';
const WEEK = 7 * 24 * 3600 * 1000;

export function CrashNotice() {
  const [report, setReport] = useState<{ name: string; time: number } | null>(null);

  useEffect(() => {
    void bridge.crashReports().then((list) => {
      const newest = list[0];
      if (!newest) return;
      let seen = 0;
      try {
        seen = Number(localStorage.getItem(SEEN_KEY) ?? 0);
      } catch {
        /* storage unavailable */
      }
      // First run: only recent reports are news.
      if (newest.time > Math.max(seen, Date.now() - WEEK)) setReport(newest);
    });
  }, []);

  if (!report) return null;
  const dismiss = () => {
    try {
      localStorage.setItem(SEEN_KEY, String(report.time));
    } catch {
      /* storage unavailable */
    }
    setReport(null);
  };
  return (
    <div className="crash-notice" role="alert">
      <AlertTriangle size={16} aria-hidden />
      <span className="grow">
        The audio engine stopped unexpectedly on {new Date(report.time).toLocaleString()}. It started again by itself, and a crash report
        was saved.
      </span>
      <button className="btn small" onClick={() => void bridge.openCrashFolder()}>
        {t("Show crash report")}
      </button>
      <button className="icon-btn subtle" onClick={dismiss} aria-label={t("Dismiss")}>
        <X size={14} />
      </button>
    </div>
  );
}
