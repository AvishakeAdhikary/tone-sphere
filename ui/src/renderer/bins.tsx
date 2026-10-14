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
import { t } from './i18n';

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
    get title() {
      return t('Physical inputs');
    },
    get hint() {
      return t('Microphones, line-ins and audio interface inputs.');
    },
    get empty() {
      return t('No microphones or inputs found.');
    },
    icon: Mic,
    hue: 'blue',
  },
  'virtual-out': {
    bin: 'virtual-out',
    get title() {
      return t('Virtual outputs');
    },
    get hint() {
      return t('Virtual speakers. Apps play into these, and the sound comes here.');
    },
    get empty() {
      return t('No virtual speakers yet. Install VB-CABLE in Settings › Devices (Compatibility mode).');
    },
    icon: Cable,
    hue: 'violet',
  },
  'app-out': {
    bin: 'app-out',
    get title() {
      return t('Application outputs');
    },
    get hint() {
      return t('Sound coming from individual apps.');
    },
    get empty() {
      return t('Apps that play sound show up here automatically.');
    },
    icon: AppWindow,
    hue: 'amber',
  },
  'physical-out': {
    bin: 'physical-out',
    get title() {
      return t('Physical outputs');
    },
    get hint() {
      return t('Speakers, headphones and audio interface outputs.');
    },
    get empty() {
      return t('No speakers or headphones found.');
    },
    icon: Speaker,
    hue: 'blue',
  },
  'virtual-in': {
    bin: 'virtual-in',
    get title() {
      return t('Virtual inputs');
    },
    get hint() {
      return t('Virtual microphones. Anything sent here can be recorded by any app.');
    },
    get empty() {
      return t('No virtual microphones yet. Install VB-CABLE in Settings › Devices (Compatibility mode).');
    },
    icon: Cable,
    hue: 'violet',
  },
  'app-in': {
    bin: 'app-in',
    get title() {
      return t('Application inputs');
    },
    get hint() {
      return t('Send sound straight into an app’s microphone (Discord, Zoom, OBS, DAWs…).');
    },
    get empty() {
      return t('Apps that use a microphone show up here automatically.');
    },
    icon: AppWindow,
    hue: 'amber',
  },
  bus: {
    bin: 'bus',
    get title() {
      return t('Mixes');
    },
    get hint() {
      return t('Combine several sources into one mix, then send it anywhere.');
    },
    get empty() {
      return t('Create a mix to blend several sources together.');
    },
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
      return t('ASIO');
    case 'wasapi':
      return n.settings.exclusive ? t('WASAPI exclusive') : t('Windows audio');
    case 'vad':
      return t('ToneSphere virtual device');
    case 'compat':
      return t('Virtual cable (compatibility)');
    case 'app':
      return t('Application');
    case 'asioclient':
      return t('ToneSphere ASIO client');
    case 'bus':
      return t('Mix');
    case 'vban':
      return t('Network (VBAN)');
    case 'tone':
      return t('Test tone');
    case 'pipewire':
      return t('PipeWire');
    case 'coreaudio':
      return t('Core Audio');
  }
}
