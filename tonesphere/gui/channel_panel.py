"""
Channel Control Panel for GUI
Provides per-channel volume, mute, solo, pan controls
"""

import tkinter as tk
from tkinter import ttk


class ChannelControlPanel(tk.Toplevel):
    """Channel control panel window"""
    
    def __init__(self, parent, engine, device_id, device_name, colors):
        super().__init__(parent)
        self.engine = engine
        self.device_id = device_id
        self.colors = colors
        
        self.title(f"Channel Controls - {device_name}")
        self.geometry("600x500")
        self.configure(bg=colors['bg_primary'])
        
        # Get channel info
        self.channel_info = engine.get_device_channels(device_id)
        if not self.channel_info:
            tk.Label(self, text="Channel control not available for this device",
                    bg=colors['bg_primary'], fg=colors['text_primary'],
                    font=('Segoe UI', 12)).pack(pady=50)
            return
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create channel control widgets"""
        # Header
        header = tk.Frame(self, bg=self.colors['bg_secondary'], height=60)
        header.pack(fill=tk.X, padx=10, pady=10)
        header.pack_propagate(False)
        
        tk.Label(header, text=f"Device {self.device_id} Channel Controls",
                bg=self.colors['bg_secondary'], fg=self.colors['accent_orange'],
                font=('Segoe UI', 14, 'bold')).pack(expand=True)
        
        # Master controls
        master_frame = tk.LabelFrame(self, text="Master Controls",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_primary'],
                                    font=('Segoe UI', 11, 'bold'))
        master_frame.pack(fill=tk.X, padx=10, pady=5)
        
        master_inner = tk.Frame(master_frame, bg=self.colors['bg_secondary'])
        master_inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Master volume
        tk.Label(master_inner, text="Master Volume:",
                bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                font=('Segoe UI', 10)).grid(row=0, column=0, sticky='w', padx=5)
        
        self.master_volume_var = tk.DoubleVar(value=self.channel_info['master_volume'])
        master_volume_scale = tk.Scale(master_inner, from_=0.0, to=2.0, resolution=0.01,
                                      orient=tk.HORIZONTAL, variable=self.master_volume_var,
                                      bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                      command=self._on_master_volume_change, length=200)
        master_volume_scale.grid(row=0, column=1, padx=5)
        
        self.master_volume_label = tk.Label(master_inner, text=f"{self.channel_info['master_volume']:.2f}",
                                           bg=self.colors['bg_secondary'], fg=self.colors['accent_gold'],
                                           font=('Segoe UI', 10, 'bold'), width=6)
        self.master_volume_label.grid(row=0, column=2, padx=5)
        
        # Master mute
        self.master_mute_var = tk.BooleanVar(value=self.channel_info['master_muted'])
        master_mute_btn = tk.Checkbutton(master_inner, text="Mute All",
                                        variable=self.master_mute_var,
                                        command=self._on_master_mute_change,
                                        bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                                        selectcolor=self.colors['accent_red'],
                                        font=('Segoe UI', 10, 'bold'))
        master_mute_btn.grid(row=0, column=3, padx=10)
        
        # Swap channels button
        if self.channel_info['num_channels'] == 2:
            swap_btn = tk.Button(master_inner, text="⇄ Swap L/R",
                               command=self._on_swap_channels,
                               bg=self.colors['accent_orange'], fg=self.colors['text_primary'],
                               font=('Segoe UI', 9, 'bold'), padx=10)
            swap_btn.grid(row=0, column=4, padx=5)
            
            swap_status = "Swapped" if self.channel_info['channels_swapped'] else "Normal"
            self.swap_label = tk.Label(master_inner, text=swap_status,
                                      bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                      font=('Segoe UI', 9))
            self.swap_label.grid(row=0, column=5, padx=5)
        
        # Individual channel controls
        channels_frame = tk.LabelFrame(self, text="Individual Channels",
                                      bg=self.colors['bg_secondary'],
                                      fg=self.colors['text_primary'],
                                      font=('Segoe UI', 11, 'bold'))
        channels_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create scrollable frame for channels
        canvas = tk.Canvas(channels_frame, bg=self.colors['bg_secondary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(channels_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['bg_secondary'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create controls for each channel
        self.channel_controls = {}
        for ch_idx, ch_info in self.channel_info['channels'].items():
            self._create_channel_control(scrollable_frame, ch_idx, ch_info)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_channel_control(self, parent, ch_idx, ch_info):
        """Create controls for a single channel"""
        ch_frame = tk.Frame(parent, bg=self.colors['bg_tertiary'], relief='raised', bd=2)
        ch_frame.pack(fill=tk.X, padx=5, pady=5)
        
        inner = tk.Frame(ch_frame, bg=self.colors['bg_tertiary'])
        inner.pack(fill=tk.X, padx=10, pady=10)
        
        # Channel label
        ch_name = "Left" if ch_idx == 0 else "Right" if ch_idx == 1 else f"Ch {ch_idx}"
        tk.Label(inner, text=f"{ch_name}:",
                bg=self.colors['bg_tertiary'], fg=self.colors['accent_orange'],
                font=('Segoe UI', 10, 'bold'), width=8).grid(row=0, column=0, sticky='w')
        
        # Volume control
        volume_var = tk.DoubleVar(value=ch_info['volume'])
        volume_scale = tk.Scale(inner, from_=0.0, to=2.0, resolution=0.01,
                               orient=tk.HORIZONTAL, variable=volume_var,
                               bg=self.colors['bg_primary'], fg=self.colors['text_primary'],
                               command=lambda v, idx=ch_idx: self._on_channel_volume_change(idx, v),
                               length=150)
        volume_scale.grid(row=0, column=1, padx=5)
        
        volume_label = tk.Label(inner, text=f"{ch_info['volume']:.2f}",
                               bg=self.colors['bg_tertiary'], fg=self.colors['accent_gold'],
                               font=('Segoe UI', 9, 'bold'), width=6)
        volume_label.grid(row=0, column=2, padx=5)
        
        # Pan control (for stereo)
        if self.channel_info['num_channels'] == 2:
            pan_var = tk.DoubleVar(value=ch_info['pan'])
            pan_scale = tk.Scale(inner, from_=-1.0, to=1.0, resolution=0.01,
                                orient=tk.HORIZONTAL, variable=pan_var,
                                bg=self.colors['bg_primary'], fg=self.colors['text_primary'],
                                command=lambda v, idx=ch_idx: self._on_channel_pan_change(idx, v),
                                length=100)
            pan_scale.grid(row=0, column=3, padx=5)
            
            pan_label = tk.Label(inner, text="Pan",
                                bg=self.colors['bg_tertiary'], fg=self.colors['text_secondary'],
                                font=('Segoe UI', 8))
            pan_label.grid(row=0, column=4, padx=2)
        
        # Mute button
        mute_var = tk.BooleanVar(value=ch_info['muted'])
        mute_btn = tk.Checkbutton(inner, text="M", variable=mute_var,
                                 command=lambda idx=ch_idx, var=mute_var: self._on_channel_mute_change(idx, var.get()),
                                 bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                 selectcolor=self.colors['accent_red'],
                                 font=('Segoe UI', 9, 'bold'), width=3)
        mute_btn.grid(row=0, column=5, padx=2)
        
        # Solo button
        solo_var = tk.BooleanVar(value=ch_info['solo'])
        solo_btn = tk.Checkbutton(inner, text="S", variable=solo_var,
                                 command=lambda idx=ch_idx, var=solo_var: self._on_channel_solo_change(idx, var.get()),
                                 bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                 selectcolor=self.colors['success'],
                                 font=('Segoe UI', 9, 'bold'), width=3)
        solo_btn.grid(row=0, column=6, padx=2)
        
        # Invert button
        invert_var = tk.BooleanVar(value=ch_info['inverted'])
        invert_btn = tk.Checkbutton(inner, text="Ø", variable=invert_var,
                                   command=lambda idx=ch_idx, var=invert_var: self._on_channel_invert_change(idx, var.get()),
                                   bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                                   selectcolor=self.colors['warning'],
                                   font=('Segoe UI', 9, 'bold'), width=3)
        invert_btn.grid(row=0, column=7, padx=2)
        
        self.channel_controls[ch_idx] = {
            'volume_var': volume_var,
            'volume_label': volume_label,
            'mute_var': mute_var,
            'solo_var': solo_var,
            'invert_var': invert_var
        }
    
    def _on_master_volume_change(self, value):
        """Handle master volume change"""
        self.engine.set_device_master_volume(self.device_id, float(value))
        self.master_volume_label.config(text=f"{float(value):.2f}")
    
    def _on_master_mute_change(self):
        """Handle master mute change"""
        self.engine.set_device_master_mute(self.device_id, self.master_mute_var.get())
    
    def _on_swap_channels(self):
        """Handle channel swap"""
        self.engine.swap_channels(self.device_id)
        current = self.swap_label.cget("text")
        new_text = "Normal" if current == "Swapped" else "Swapped"
        self.swap_label.config(text=new_text)
    
    def _on_channel_volume_change(self, ch_idx, value):
        """Handle channel volume change"""
        self.engine.set_channel_volume(self.device_id, ch_idx, float(value))
        if ch_idx in self.channel_controls:
            self.channel_controls[ch_idx]['volume_label'].config(text=f"{float(value):.2f}")
    
    def _on_channel_pan_change(self, ch_idx, value):
        """Handle channel pan change"""
        self.engine.set_channel_pan(self.device_id, ch_idx, float(value))
    
    def _on_channel_mute_change(self, ch_idx, muted):
        """Handle channel mute change"""
        self.engine.set_channel_mute(self.device_id, ch_idx, muted)
    
    def _on_channel_solo_change(self, ch_idx, solo):
        """Handle channel solo change"""
        self.engine.set_channel_solo(self.device_id, ch_idx, solo)
    
    def _on_channel_invert_change(self, ch_idx, inverted):
        """Handle channel invert change"""
        self.engine.set_channel_pan(self.device_id, ch_idx, inverted)
