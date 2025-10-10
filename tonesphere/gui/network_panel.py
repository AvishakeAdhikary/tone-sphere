"""
Network Audio Routing Panel for GUI
Provides network connection and routing controls
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog


class NetworkRoutingPanel(tk.Toplevel):
    """Network routing panel window"""
    
    def __init__(self, parent, engine, colors):
        super().__init__(parent)
        self.engine = engine
        self.colors = colors
        
        self.title("Network Audio Routing")
        self.geometry("800x600")
        self.configure(bg=colors['bg_primary'])
        
        self._create_widgets()
        self._update_connections()
        
        # Auto-update every 2 seconds
        self.after(2000, self._auto_update)
    
    def _create_widgets(self):
        """Create network panel widgets"""
        # Header
        header = tk.Frame(self, bg=self.colors['bg_secondary'], height=60)
        header.pack(fill=tk.X, padx=10, pady=10)
        header.pack_propagate(False)
        
        tk.Label(header, text="Network Audio Routing",
                bg=self.colors['bg_secondary'], fg=self.colors['accent_primary'],
                font=('Segoe UI', 16, 'bold')).pack(expand=True)
        
        # Main content
        content = tk.Frame(self, bg=self.colors['bg_primary'])
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left side - Connections
        left_frame = tk.LabelFrame(content, text="Network Connections",
                                  bg=self.colors['bg_secondary'],
                                  fg=self.colors['text_primary'],
                                  font=('Segoe UI', 12, 'bold'))
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # Connection buttons
        conn_buttons = tk.Frame(left_frame, bg=self.colors['bg_secondary'])
        conn_buttons.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(conn_buttons, text="➕ Connect to Instance",
                 command=self._connect_to_instance,
                 bg=self.colors['accent_primary'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 10, 'bold'), padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(conn_buttons, text="🔄 Refresh",
                 command=self._update_connections,
                 bg=self.colors['bg_tertiary'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 10, 'bold'), padx=10, pady=5).pack(side=tk.LEFT, padx=5)
        
        # Incoming connections
        incoming_frame = tk.LabelFrame(left_frame, text="Incoming Clients",
                                      bg=self.colors['bg_tertiary'],
                                      fg=self.colors['text_primary'],
                                      font=('Segoe UI', 10, 'bold'))
        incoming_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.incoming_listbox = tk.Listbox(incoming_frame,
                                          bg=self.colors['bg_primary'],
                                          fg=self.colors['text_primary'],
                                          font=('Consolas', 10),
                                          selectbackground=self.colors['accent_primary'])
        self.incoming_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Outgoing connections
        outgoing_frame = tk.LabelFrame(left_frame, text="Outgoing Connections",
                                      bg=self.colors['bg_tertiary'],
                                      fg=self.colors['text_primary'],
                                      font=('Segoe UI', 10, 'bold'))
        outgoing_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        outgoing_inner = tk.Frame(outgoing_frame, bg=self.colors['bg_tertiary'])
        outgoing_inner.pack(fill=tk.BOTH, expand=True)
        
        self.outgoing_listbox = tk.Listbox(outgoing_inner,
                                          bg=self.colors['bg_primary'],
                                          fg=self.colors['text_primary'],
                                          font=('Consolas', 10),
                                          selectbackground=self.colors['accent_primary'])
        self.outgoing_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        outgoing_buttons = tk.Frame(outgoing_inner, bg=self.colors['bg_tertiary'])
        outgoing_buttons.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        
        tk.Button(outgoing_buttons, text="❌\nDisconnect",
                 command=self._disconnect_selected,
                 bg=self.colors['accent_red'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 9, 'bold'), padx=5, pady=10).pack(pady=5)
        
        # Right side - Device Routing
        right_frame = tk.LabelFrame(content, text="Device Network Routing",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_primary'],
                                    font=('Segoe UI', 12, 'bold'))
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        # Device selection
        device_select = tk.Frame(right_frame, bg=self.colors['bg_secondary'])
        device_select.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(device_select, text="Device:",
                bg=self.colors['bg_secondary'], fg=self.colors['text_primary'],
                font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=5)
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_select, textvariable=self.device_var,
                                        state='readonly', width=30)
        self.device_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Routing buttons
        routing_buttons = tk.Frame(right_frame, bg=self.colors['bg_secondary'])
        routing_buttons.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(routing_buttons, text="📤 Send to Network",
                 command=self._send_device_to_network,
                 bg=self.colors['accent_primary'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 10, 'bold'), padx=15, pady=8).pack(side=tk.LEFT, padx=5)
        
        tk.Button(routing_buttons, text="📥 Receive from Network",
                 command=self._receive_from_network,
                 bg=self.colors['success'], fg=self.colors['text_primary'],
                 font=('Segoe UI', 10, 'bold'), padx=15, pady=8).pack(side=tk.LEFT, padx=5)
        
        # Statistics
        stats_frame = tk.LabelFrame(right_frame, text="Network Statistics",
                                   bg=self.colors['bg_tertiary'],
                                   fg=self.colors['text_primary'],
                                   font=('Segoe UI', 10, 'bold'))
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=15,
                                 bg=self.colors['bg_primary'],
                                 fg=self.colors['text_primary'],
                                 font=('Consolas', 9),
                                 state='disabled')
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Update devices
        self._update_devices()
    
    def _update_devices(self):
        """Update device list"""
        try:
            devices = self.engine.get_devices()
            device_list = [f"{d['id']}: {d['name']}" for d in devices]
            self.device_combo['values'] = device_list
            if device_list:
                self.device_combo.current(0)
        except:
            pass
    
    def _update_connections(self):
        """Update connection lists"""
        try:
            # Update incoming clients
            self.incoming_listbox.delete(0, tk.END)
            clients = self.engine.get_network_clients()
            for client in clients:
                self.incoming_listbox.insert(tk.END, client)
            
            # Update outgoing connections
            self.outgoing_listbox.delete(0, tk.END)
            connections = self.engine.get_network_connections()
            for conn in connections:
                self.outgoing_listbox.insert(tk.END, conn)
            
            # Update statistics
            self._update_statistics()
        except Exception as e:
            pass
    
    def _update_statistics(self):
        """Update network statistics"""
        try:
            stats = self.engine.get_network_statistics()
            
            self.stats_text.config(state='normal')
            self.stats_text.delete('1.0', tk.END)
            
            stats_text = f"""
Network Statistics
{'='*40}

Packets Sent:     {stats.get('packets_sent', 0):,}
Packets Received: {stats.get('packets_received', 0):,}

Bytes Sent:       {stats.get('bytes_sent', 0):,}
Bytes Received:   {stats.get('bytes_received', 0):,}

Quality:          {stats.get('quality', 'unknown').upper()}

Incoming Clients: {stats.get('connected_clients', 0)}
Outgoing Conns:   {stats.get('outgoing_connections', 0)}
"""
            
            self.stats_text.insert('1.0', stats_text)
            self.stats_text.config(state='disabled')
        except:
            pass
    
    def _connect_to_instance(self):
        """Connect to another ToneSphere instance"""
        host = simpledialog.askstring("Connect to Instance", 
                                     "Enter host address:",
                                     parent=self)
        if not host:
            return
        
        port = simpledialog.askinteger("Connect to Instance",
                                      "Enter port number:",
                                      initialvalue=9001,
                                      parent=self)
        if not port:
            return
        
        try:
            success = self.engine.connect_to_network(host, port)
            if success:
                messagebox.showinfo("Success", f"Connected to {host}:{port}")
                self._update_connections()
            else:
                messagebox.showerror("Error", f"Failed to connect to {host}:{port}")
        except Exception as e:
            messagebox.showerror("Error", f"Connection error: {e}")
    
    def _disconnect_selected(self):
        """Disconnect from selected connection"""
        selection = self.outgoing_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a connection to disconnect")
            return
        
        conn_id = self.outgoing_listbox.get(selection[0])
        
        try:
            self.engine.disconnect_from_network(conn_id)
            messagebox.showinfo("Disconnected", f"Disconnected from {conn_id}")
            self._update_connections()
        except Exception as e:
            messagebox.showerror("Error", f"Disconnect error: {e}")
    
    def _send_device_to_network(self):
        """Send device audio to network"""
        device_str = self.device_var.get()
        if not device_str:
            messagebox.showwarning("No Device", "Please select a device")
            return
        
        device_id = int(device_str.split(':')[0])
        
        try:
            self.engine.send_device_to_network(device_id)
            messagebox.showinfo("Success", f"Sending device {device_id} audio to network")
        except Exception as e:
            messagebox.showerror("Error", f"Send error: {e}")
    
    def _receive_from_network(self):
        """Receive network audio to device"""
        device_str = self.device_var.get()
        if not device_str:
            messagebox.showwarning("No Device", "Please select a device")
            return
        
        device_id = int(device_str.split(':')[0])
        
        try:
            self.engine.register_network_receive(device_id)
            messagebox.showinfo("Success", f"Device {device_id} registered for network receive")
        except Exception as e:
            messagebox.showerror("Error", f"Receive error: {e}")
    
    def _auto_update(self):
        """Auto-update connections and stats"""
        if self.winfo_exists():
            self._update_connections()
            self.after(2000, self._auto_update)
