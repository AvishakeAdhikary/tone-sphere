import type { Bin, NodeInfo } from '@shared/protocol';
import {
  AppWindow,
  AudioLines,
  Cable,
  Globe,
  Headphones,
  Mic,
  MonitorSpeaker,
  Music2,
  Radio,
  Speaker,
  Waves,
  type LucideIcon,
} from 'lucide-react';

export interface BinMeta {
  bin: Bin;
  title: string;
  /** One plain-language sentence explaining the bin. */
  hint: string;
  empty: string;
  icon: LucideIcon;
  hue: 'blue' | 'violet' | 'amber' | 'green';
}

export const BIN_META: Record<Bin, BinMeta> = {
  'physical-in': {
    bin: 'physical-in',
    title: 'Physical inputs',
    hint: 'Microphones, line-ins and audio interface inputs.',
    empty: 'No microphones or inputs found.',
    icon: Mic,
    hue: 'blue',
  },
  'virtual-out': {
    bin: 'virtual-out',
    title: 'Virtual outputs',
    hint: 'Virtual speakers. Apps play into these, and the sound comes here.',
    empty: 'No virtual speakers yet. Install the ToneSphere Driver Pack or enable Compatibility mode.',
    icon: Cable,
    hue: 'violet',
  },
  'app-out': {
    bin: 'app-out',
    title: 'Application outputs',
    hint: 'Sound coming from individual apps.',
    empty: 'Apps that play sound show up here automatically.',
    icon: AppWindow,
    hue: 'amber',
  },
  'physical-out': {
    bin: 'physical-out',
    title: 'Physical outputs',
    hint: 'Speakers, headphones and audio interface outputs.',
    empty: 'No speakers or headphones found.',
    icon: Speaker,
    hue: 'blue',
  },
  'virtual-in': {
    bin: 'virtual-in',
    title: 'Virtual inputs',
    hint: 'Virtual microphones. Anything sent here can be recorded by any app.',
    empty: 'No virtual microphones yet. Install the ToneSphere Driver Pack or enable Compatibility mode.',
    icon: Cable,
    hue: 'violet',
  },
  'app-in': {
    bin: 'app-in',
    title: 'Application inputs',
    hint: 'Send sound straight into an app’s microphone (Discord, Zoom, OBS, DAWs…).',
    empty: 'Apps that use a microphone show up here automatically.',
    icon: AppWindow,
    hue: 'amber',
  },
  bus: {
    bin: 'bus',
    title: 'Mixes',
    hint: 'Combine several sources into one mix, then send it anywhere.',
    empty: 'Create a mix to blend several sources together.',
    icon: Waves,
    hue: 'green',
  },
};

export const SOURCE_ORDER: Bin[] = ['physical-in', 'virtual-out', 'app-out'];
export const DEST_ORDER: Bin[] = ['physical-out', 'virtual-in', 'app-in'];

/** Picks a descriptive icon for a node. */
export function nodeIcon(n: NodeInfo): LucideIcon {
  if (n.kind === 'vban') return Globe;
  if (n.kind === 'bus') return Waves;
  if (n.kind === 'tone') return AudioLines;
  if (n.kind === 'asioclient') return Music2;
  if (n.kind === 'vad' || n.kind === 'compat') return Cable;
  if (n.kind === 'app') return AppWindow;
  const name = n.name.toLowerCase();
  if (n.bin === 'physical-in') {
    if (name.includes('line') || name.includes('stereo mix')) return Radio;
    return Mic;
  }
  if (n.bin === 'physical-out') {
    if (name.includes('headphone') || name.includes('headset')) return Headphones;
    if (name.includes('hdmi') || name.includes('display') || name.includes('monitor')) return MonitorSpeaker;
    return Speaker;
  }
  return BIN_META[n.bin].icon;
}

export function kindLabel(n: NodeInfo): string {
  switch (n.kind) {
    case 'asio':
      return 'ASIO';
    case 'wasapi':
      return n.settings.exclusive ? 'WASAPI exclusive' : 'Windows audio';
    case 'vad':
      return 'ToneSphere virtual device';
    case 'compat':
      return 'Virtual cable (compatibility)';
    case 'app':
      return 'Application';
    case 'asioclient':
      return 'ToneSphere ASIO client';
    case 'bus':
      return 'Mix';
    case 'vban':
      return 'Network (VBAN)';
    case 'tone':
      return 'Test tone';
    case 'pipewire':
      return 'PipeWire';
    case 'coreaudio':
      return 'Core Audio';
  }
}
