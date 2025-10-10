"""
Virtual Device Management Panel for GUI
CRUD operations for virtual audio devices
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class VirtualDevicePanel(tk.Toplevel):
    """Virtual device management panel"""
    
    def __init__(self, parent, engine, colors):
        super().__init__(parent)
        self.engine = engine
        self.colors = colors
        
        self.title("Virtual Device Manager")
        self.geometry("900x600")
        self.configure(bg=colors['bg_primary'])
        
        self._create_widgets()
        self._refresh_devices()
        
        # Auto-refresh every 3 seconds
        self.after(3000, self._auto_refresh)
    
    def _create_widgets(self):
        """Create panel widgets"""
        # Header
        header = tk.Frame(self, bg=self.colors['bg_secondary'], height=70)
        header.pack(fill=tk.X, padx=10, pady=10)
        header.pack_propagate(False)
        
        tk.Label(header, text="Virtual Device Manager",
                bg=self.colors['bg_secondary'], fg=self.colors['accent_primary'],
                font=('Segoe UI', 16, 'bold')).pack(side=tk.LEFT, padx=20, expand=True)
        
        # Limits info
        self.limits_label = tk.Label(header, text="",
                                     bg=self.colors['bg_secondary'], fg=self.colors['text_secondary'],
                                     font=('Segoe UI', 10))
        self.limits_label.pack(side=tk.RIGHT, padx=20)
        
        # Main content
        content = tk.Frame(self, bg=self.colors['bg_primary'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Device list
        left_frame = tk.LabelFrame(content, text="Virtual Devices",
                                   bg=self.colors['bg_secondary'],
                                   fg=self.colors['text_primary'],
                                   font=('Segoe UI', 12, 'bold'))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Device list
        list_frame = tk.Frame(left_frame, bg=self.colors['bg_secondary'])
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Treeview for devices
        columns = ('ID', 'Name', 'Type', 'Channels', 'Sample Rate', 'Status')
        self.device_tree = ttk.Treeview(list_frame, columns=columns,
                                       show='headings', height=15)
        
        # Configure columns
        self.device_tree.heading('ID', text='ID')
        self.device_tree.heading('Name', text='Name')
        self.device_tree.heading('Type', text='Type')
        self.device_tree.heading('Channels', text='Channels')
        self.device_tree.heading('Sample Rate', text='Sample Rate')
        self.device_tree.heading('Status', text='Status')
        
        self.device_tree.column('ID', width=60)
        self.device_tree.column('Name', width=200)
        self.device_tree.column('Type', width=100)
        self.device_tree.column('Channels', width=80)
        self.device_tree.column('Sample Rate', width=100)
        self.device_tree.column('Status', width=80)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                 command=self.device_tree.yview)
        self.device_tree.configure(yscrollcommand=scrollbar.set)
        
        self.device_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right side - Controls
        right_frame = tk.LabelFrame(content, text="Device Controls",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_primary'],
                                    font=('Segoe UI', 12, 'bold'))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        
        controls_inner = tk.Frame(right_frame, bg=self.colors['bg_secondary'])
        controls_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Create section
        create_frame = tk.LabelFrame(controls_inner, text="Create Device",
                                    bg=self.colors['bg_tertiary'],
                                    fg=self.colors['text_primary'],
                                    font=('Segoe UI', 10, 'bold'))
        create_frame.pack(fill=tk.X, pady=10)
        
        create_inner = tk.Frame(create_frame, bg=self.colors['bg_tertiary'])
        create_inner.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(create_inner, text="Channels:",
                bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                font=('Segoe UI', 9)).grid(row=0, column=0, sticky='w', pady=5)
        
        self.channels_var = tk.IntVar(value=2)
        channels_spin = tk.Spinbox(create_inner, from_=1, to=32,
                                   textvariable=self.channels_var,
                                   width=10, font=('Segoe UI', 9))
        channels_spin.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Button(create_inner, text="➕ Create Input",
                 command=self._create_input,
                 bg=self.colors['accent_primary'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 9, 'bold'), padx=15, pady=5).grid(row=1, column=0, columnspan=2, pady=5)
        
        tk.Button(create_inner, text="➕ Create Output",
                 command=self._create_output,
                 bg=self.colors['success'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 9, 'bold'), padx=15, pady=5).grid(row=2, column=0, columnspan=2, pady=5)
        
        # Modify section
        modify_frame = tk.LabelFrame(controls_inner, text="Modify Device",
                                     bg=self.colors['bg_tertiary'],
                                     fg=self.colors['text_primary'],
                                     font=('Segoe UI', 10, 'bold'))
        modify_frame.pack(fill=tk.X, pady=10)
        
        modify_inner = tk.Frame(modify_frame, bg=self.colors['bg_tertiary'])
        modify_inner.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(modify_inner, text="🔧 Change Sample Rate",
                 command=self._change_sample_rate,
                 bg=self.colors['bg_primary'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 9, 'bold'), padx=10, pady=5).pack(fill=tk.X, pady=3)
        
        tk.Button(modify_inner, text="🔧 Change Channels",
                 command=self._change_channels,
                 bg=self.colors['bg_primary'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 9, 'bold'), padx=10, pady=5).pack(fill=tk.X, pady=3)
        
        # Delete section
        delete_frame = tk.LabelFrame(controls_inner, text="Delete Device",
                                     bg=self.colors['bg_tertiary'],
                                     fg=self.colors['text_primary'],
                                     font=('Segoe UI', 10, 'bold'))
        delete_frame.pack(fill=tk.X, pady=10)
        
        delete_inner = tk.Frame(delete_frame, bg=self.colors['bg_tertiary'])
        delete_inner.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(delete_inner, text="🗑️ Delete Selected",
                 command=self._delete_device,
                 bg=self.colors['accent_red'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 9, 'bold'), padx=10, pady=5).pack(fill=tk.X, pady=3)
        
        # Refresh button
        tk.Button(controls_inner, text="🔄 Refresh",
                 command=self._refresh_devices,
                 bg=self.colors['bg_primary'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 9, 'bold'), padx=10, pady=8).pack(fill=tk.X, pady=10)
        
        # Info section
        info_frame = tk.LabelFrame(controls_inner, text="Information",
                                   bg=self.colors['bg_tertiary'],
                                   fg=self.colors['text_primary'],
                                   font=('Segoe UI', 10, 'bold'))
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.info_text = tk.Text(info_frame, height=8,
                                bg=self.colors['bg_primary'],
                                fg=self.colors['text_secondary'],
                                font=('Consolas', 8),
                                state='disabled', wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _refresh_devices(self):
        """Refresh device list"""
        try:
            # Clear tree
            for item in self.device_tree.get_children():
                self.device_tree.delete(item)
            
            # Get virtual devices
            devices = self.engine.list_virtual_devices()
            
            for device in devices:
                status = "Running" if device['is_running'] else "Stopped"
                self.device_tree.insert('', 'end', values=(
                    device['id'],
                    device['name'],
                    device['type'],
                    device['channels'],
                    f"{device['sample_rate']}Hz",
                    status
                ))
            
            # Update limits
            counts = self.engine.get_virtual_device_counts()
            limits_text = (f"Inputs: {counts['input_count']}/{counts['max_inputs']}  |  "
                          f"Outputs: {counts['output_count']}/{counts['max_outputs']}  |  "
                          f"Total: {counts['total_count']}")
            self.limits_label.config(text=limits_text)
            
            # Update info
            self._update_info(counts)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh devices: {e}")
    
    def _update_info(self, counts):
        """Update info text"""
        info = f"""
Virtual Device Statistics
{'='*35}

Inputs:  {counts['input_count']} / {counts['max_inputs']}
Outputs: {counts['output_count']} / {counts['max_outputs']}
Total:   {counts['total_count']}

Available:
  Inputs:  {counts['inputs_available']}
  Outputs: {counts['outputs_available']}

Note: Device names are auto-generated
and cannot be changed.
"""
        self.info_text.config(state='normal')
        self.info_text.delete('1.0', tk.END)
        self.info_text.insert('1.0', info)
        self.info_text.config(state='disabled')
    
    def _create_input(self):
        """Create virtual input"""
        try:
            channels = self.channels_var.get()
            device_id = self.engine.create_virtual_input("Virtual Input", channels)
            
            if device_id:
                messagebox.showinfo("Success", f"Created virtual input with ID: {device_id}")
                self._refresh_devices()
            else:
                messagebox.showwarning("Limit Reached", "Cannot create more virtual inputs. Limit reached.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create virtual input: {e}")
    
    def _create_output(self):
        """Create virtual output"""
        try:
            channels = self.channels_var.get()
            device_id = self.engine.create_virtual_output("Virtual Output", channels)
            
            if device_id:
                messagebox.showinfo("Success", f"Created virtual output with ID: {device_id}")
                self._refresh_devices()
            else:
                messagebox.showwarning("Limit Reached", "Cannot create more virtual outputs. Limit reached.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create virtual output: {e}")
    
    def _delete_device(self):
        """Delete selected device"""
        selection = self.device_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a device to delete")
            return
        
        item = self.device_tree.item(selection[0])
        device_id = item['values'][0]
        device_name = item['values'][1]
        
        confirm = messagebox.askyesno("Confirm Delete",
                                      f"Delete device {device_name} (ID: {device_id})?")
        if not confirm:
            return
        
        try:
            success = self.engine.delete_virtual_device(device_id)
            if success:
                messagebox.showinfo("Success", f"Deleted device {device_id}")
                self._refresh_devices()
            else:
                messagebox.showerror("Error", "Failed to delete device")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete device: {e}")
    
    def _change_sample_rate(self):
        """Change sample rate of selected device"""
        selection = self.device_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a device")
            return
        
        item = self.device_tree.item(selection[0])
        device_id = item['values'][0]
        current_rate = int(item['values'][4].replace('Hz', ''))
        
        new_rate = simpledialog.askinteger("Change Sample Rate",
                                          f"Current: {current_rate}Hz\n\nEnter new sample rate:",
                                          initialvalue=current_rate,
                                          minvalue=8000,
                                          maxvalue=192000,
                                          parent=self)
        
        if new_rate and new_rate != current_rate:
            try:
                success = self.engine.update_virtual_device_sample_rate(device_id, new_rate)
                if success:
                    messagebox.showinfo("Success", f"Sample rate changed to {new_rate}Hz")
                    self._refresh_devices()
                else:
                    messagebox.showerror("Error", "Failed to change sample rate")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to change sample rate: {e}")
    
    def _change_channels(self):
        """Change channels of selected device"""
        selection = self.device_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a device")
            return
        
        item = self.device_tree.item(selection[0])
        device_id = item['values'][0]
        current_channels = item['values'][3]
        
        new_channels = simpledialog.askinteger("Change Channels",
                                              f"Current: {current_channels}\n\nEnter new channel count:",
                                              initialvalue=current_channels,
                                              minvalue=1,
                                              maxvalue=32,
                                              parent=self)
        
        if new_channels and new_channels != current_channels:
            try:
                success = self.engine.update_virtual_device_channels(device_id, new_channels)
                if success:
                    messagebox.showinfo("Success", f"Channels changed to {new_channels}")
                    self._refresh_devices()
                else:
                    messagebox.showerror("Error", "Failed to change channels")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to change channels: {e}")
    
    def _auto_refresh(self):
        """Auto-refresh devices"""
        if self.winfo_exists():
            self._refresh_devices()
            self.after(3000, self._auto_refresh)
