import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

/** Fixed-position popover rendered in a portal, anchored to an element, closes on outside click / Escape. */
export function Popover({
  anchor,
  onClose,
  children,
  width = 300,
  className = '',
  label,
}: {
  anchor: HTMLElement | null;
  onClose: () => void;
  children: ReactNode;
  width?: number;
  className?: string;
  label: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ left: number; top: number; maxHeight: number } | null>(null);

  useLayoutEffect(() => {
    if (!anchor) return;
    const place = () => {
      const r = anchor.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const left = Math.min(Math.max(8, r.right - width), vw - width - 8);
      const below = vh - r.bottom - 12;
      const above = r.top - 12;
      if (below >= 260 || below >= above) setPos({ left, top: r.bottom + 6, maxHeight: below });
      else setPos({ left, top: Math.max(8, r.top - 6 - Math.min(above, 480)), maxHeight: Math.min(above, 480) });
    };
    place();
    window.addEventListener('resize', place);
    return () => window.removeEventListener('resize', place);
  }, [anchor, width]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node) && !anchor?.contains(e.target as Node)) onClose();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
        anchor?.focus();
      }
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey, true);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey, true);
    };
  }, [anchor, onClose]);

  useEffect(() => {
    ref.current?.querySelector<HTMLElement>('button, input, [tabindex]')?.focus();
  }, [pos != null]);

  if (!pos) return null;
  return createPortal(
    <div
      ref={ref}
      className={`popover floating ${className}`}
      role="dialog"
      aria-label={label}
      style={{ left: pos.left, top: pos.top, width, maxHeight: pos.maxHeight }}
      onClick={(e) => e.stopPropagation()}
    >
      {children}
    </div>,
    document.body,
  );
}
