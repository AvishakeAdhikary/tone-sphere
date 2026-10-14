import { useEffect, useRef } from 'react';
import type { MeterSlot } from '@shared/protocol';
import { bindMeter } from '../engine/meterLoop';

/** Per-channel level meter. Motion is driven by the shared meter loop (no re-renders). */
export function Meter({
  slot,
  orientation = 'horizontal',
  maxChannels = 8,
  className = '',
}: {
  slot: MeterSlot;
  orientation?: 'horizontal' | 'vertical';
  maxChannels?: number;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const channels = Math.min(slot.channels, maxChannels);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const bars = Array.from(el.querySelectorAll<HTMLElement>('.meter-ch'));
    const unbind = bars.map((bar, c) => bindMeter(bar, slot.offset + c * 2));
    return () => unbind.forEach((u) => u());
  }, [slot.offset, channels]);

  return (
    <div ref={ref} className={`meter ${orientation} ${className}`} aria-hidden>
      {Array.from({ length: channels }, (_, c) => (
        <div key={c} className="meter-ch">
          <div className="meter-fill" />
          <div className="meter-hold" />
        </div>
      ))}
    </div>
  );
}
