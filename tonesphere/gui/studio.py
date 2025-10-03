import tkinter as tk
from tkinter import ttk, messagebox
from tonesphere.core.engine import AudioEngine
from tonesphere.utils.config import ConfigManager
import threading
import time

class ToneSphereStudioGUI:
    """Professional GUI for ToneSphere Studio"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ToneSphere Studio - Professional Audio Routing")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0a0a0a')
        
        # Modern color scheme
        self.colors = {
            'bg_primary': '#0a0a0a',
            'bg_secondary': '#1a1a1a', 
            'bg_tertiary': '#2a2a2a',
            'accent_red': '#ff4444',
            'accent_orange': '#ff8844',
            'accent_gold': '#ffaa44',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'text_muted': '#888888',
            'success': '#44ff44',
            'warning': '#ffaa44',
            'error': '#ff4444'
        }
        
        # Engine state
        self.engine = None
        self.is_running = False
        self.engine_toggle_state = False
        self.routing_window = None

        # Initialize combobox variables
        self.source_combo = None
        self.dest_combo = None
        self.volume_var = None
        self.volume_label = None
        
        self._create_modern_widgets()
        self._create_menu()
        self.start_updates()
    
    def _configure_styles(self):
        """Configure modern professional styles"""
        # Colors are already defined in __init__, no need to update them again
        pass
        
    def _create_menu(self):
        """Create modern application menu"""
        menubar = tk.Menu(self.root, bg=self.colors['bg_secondary'], 
                        fg=self.colors['text_primary'], 
                        activebackground=self.colors['accent_orange'])
        self.root.config(menu=menubar)
        
        # Engine menu
        engine_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'])
        menubar.add_cascade(label="Engine", menu=engine_menu)
        engine_menu.add_command(label="Toggle Engine", command=self.toggle_engine)
        engine_menu.add_separator()
        engine_menu.add_command(label="Exit", command=self.root.quit)
        
        # Routing menu
        routing_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'])
        menubar.add_cascade(label="Routing", menu=routing_menu)
        routing_menu.add_command(label="Open Routing Matrix", command=self.open_routing_window)
        
        # Network menu
        network_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_secondary'],
                            fg=self.colors['text_primary'])
        menubar.add_cascade(label="Network", menu=network_menu)
        network_menu.add_command(label="Start Streaming", command=self.start_network)
        network_menu.add_command(label="Stop Streaming", command=self.stop_network)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=self.colors['bg_secondary'],
                        fg=self.colors['text_primary'])
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _create_modern_widgets(self):
        """Create modern GUI widgets with professional styling"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title with modern styling
        title_frame = tk.Frame(main_frame, bg=self.colors['bg_primary'], height=80)
        title_frame.pack(fill=tk.X, pady=(0, 20))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="ToneSphere Studio", 
                            bg=self.colors['bg_primary'],
                            fg=self.colors['accent_orange'],
                            font=('Segoe UI', 28, 'bold'))
        title_label.pack(expand=True)
        
        # Status card with toggle button
        status_card = tk.Frame(main_frame, bg=self.colors['bg_secondary'], 
                            relief='raised', bd=2)
        status_card.pack(fill=tk.X, pady=(0, 20), ipady=15)
        
        status_inner = tk.Frame(status_card, bg=self.colors['bg_secondary'])
        status_inner.pack(fill=tk.X, padx=20, pady=10)
        
        # Main toggle button
        self.engine_button = tk.Button(status_inner, 
                                    text="⚡ START ENGINE",
                                    command=self.toggle_engine,
                                    bg=self.colors['bg_tertiary'],
                                    fg=self.colors['text_primary'],
                                    activebackground=self.colors['accent_red'],
                                    font=('Segoe UI', 14, 'bold'),
                                    relief='raised', bd=3,
                                    padx=40, pady=15,
                                    cursor='hand2')
        self.engine_button.pack(side=tk.LEFT)
        
        # Status info
        status_info_frame = tk.Frame(status_inner, bg=self.colors['bg_secondary'])
        status_info_frame.pack(side=tk.RIGHT)
        
        self.status_label = tk.Label(status_info_frame, text="Engine: Stopped",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_secondary'],
                                    font=('Segoe UI', 12, 'bold'))
        self.status_label.pack(anchor='e')
        
        self.performance_label = tk.Label(status_info_frame, text="CPU: 0% | Latency: 0ms",
                                        bg=self.colors['bg_secondary'],
                                        fg=self.colors['text_muted'],
                                        font=('Segoe UI', 10))
        self.performance_label.pack(anchor='e')

                
        # Control panel
        control_panel = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                                relief='raised', bd=2)
        control_panel.pack(fill=tk.X, pady=(0, 20), ipady=10)
        
        control_inner = tk.Frame(control_panel, bg=self.colors['bg_secondary'])
        control_inner.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(control_inner, text="Quick Actions", 
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_orange'],
                font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        button_frame = tk.Frame(control_inner, bg=self.colors['bg_secondary'])
        button_frame.pack(fill=tk.X)
        
        # Modern action buttons
        self._create_action_button(button_frame, "🔄 Refresh Devices", self.refresh_devices)
        self._create_action_button(button_frame, "🔗 Routing Matrix", self.open_routing_window)
        self._create_action_button(button_frame, "🌐 Network Panel", self.toggle_network_panel)
        
        # Network status panel (initially hidden)
        self.network_panel = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                                     relief='raised', bd=2)
        
        network_inner = tk.Frame(self.network_panel, bg=self.colors['bg_secondary'])
        network_inner.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Label(network_inner, text="Network Streaming", 
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_orange'],
                font=('Segoe UI', 14, 'bold')).pack(anchor='w', pady=(0, 10))
        
        network_buttons = tk.Frame(network_inner, bg=self.colors['bg_secondary'])
        network_buttons.pack(fill=tk.X)
        
        self._create_action_button(network_buttons, "▶️ Start Streaming", self.start_network)
        self._create_action_button(network_buttons, "⏹️ Stop Streaming", self.stop_network)
        
        self.client_count_label = tk.Label(network_buttons, text="Clients: 0",
                                          bg=self.colors['bg_secondary'],
                                          fg=self.colors['text_secondary'],
                                          font=('Segoe UI', 11, 'bold'))
        self.client_count_label.pack(side=tk.RIGHT, padx=10)
        
        # Devices panel
        devices_panel = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                                relief='raised', bd=2)
        devices_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        devices_header = tk.Frame(devices_panel, bg=self.colors['bg_secondary'])
        devices_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        tk.Label(devices_header, text="Audio Devices", 
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_orange'],
                font=('Segoe UI', 16, 'bold')).pack(side=tk.LEFT)
        
        # Modern devices treeview
        devices_frame = tk.Frame(devices_panel, bg=self.colors['bg_secondary'])
        devices_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        columns = ('ID', 'Name', 'Type', 'Channels', 'ASIO', 'Latency')
        self.devices_tree = ttk.Treeview(devices_frame, columns=columns, 
                                        show='headings', height=12)
        
        # Configure treeview colors
        style = ttk.Style()
        style.configure('Treeview', 
                       background=self.colors['bg_tertiary'],
                       foreground=self.colors['text_primary'],
                       fieldbackground=self.colors['bg_tertiary'])
        style.configure('Treeview.Heading',
                       background=self.colors['bg_secondary'],
                       foreground=self.colors['accent_orange'],
                       font=('Segoe UI', 11, 'bold'))
        
        for col in columns:
            self.devices_tree.heading(col, text=col)
            if col == 'Name':
                self.devices_tree.column(col, width=300)
            elif col == 'Type':
                self.devices_tree.column(col, width=150)
            else:
                self.devices_tree.column(col, width=80)
        
        # Scrollbars
        devices_scrollbar_y = ttk.Scrollbar(devices_frame, orient=tk.VERTICAL, 
                                           command=self.devices_tree.yview)
        self.devices_tree.configure(yscrollcommand=devices_scrollbar_y.set)
        
        self.devices_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        devices_scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log panel
        log_panel = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                            relief='raised', bd=2)
        log_panel.pack(fill=tk.BOTH, expand=True)
        
        log_header = tk.Frame(log_panel, bg=self.colors['bg_secondary'])
        log_header.pack(fill=tk.X, padx=20, pady=(15, 10))
        
        tk.Label(log_header, text="System Log", 
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_orange'],
                font=('Segoe UI', 14, 'bold')).pack(side=tk.LEFT)
        
        log_frame = tk.Frame(log_panel, bg=self.colors['bg_secondary'])
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        self.log_text = tk.Text(log_frame, height=8, 
                               bg=self.colors['bg_primary'], 
                               fg=self.colors['text_primary'],
                               font=('Consolas', 10),
                               insertbackground=self.colors['accent_orange'],
                               selectbackground=self.colors['accent_red'],
                               relief='sunken', bd=2)
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, 
                                     command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initially hide network panel
        self.network_panel_visible = False
        
    def toggle_engine(self):
        """Toggle engine start/stop with single button"""
        try:
            if not self.engine_toggle_state:
                # Start engine
                if not self.engine:
                    config = ConfigManager().load_config()
                    self.engine = AudioEngine(
                        sample_rate=config['engine']['sample_rate'],
                        buffer_size=config['engine']['buffer_size']
                    )
                    self.engine.initialize()
                
                self.engine.start_engine()
                self.is_running = True
                self.engine_toggle_state = True
                
                # Update button appearance
                self.engine_button.configure(
                    text="⏹️ STOP ENGINE",
                    bg=self.colors['accent_red'],
                    activebackground=self.colors['accent_orange']
                )
                
                self.log_message("✅ Audio engine started successfully", "success")
                self.refresh_devices()
                
            else:
                # Stop engine
                if self.engine:
                    self.engine.stop_engine()
                    self.is_running = False
                    self.engine_toggle_state = False
                    
                    # Update button appearance
                    self.engine_button.configure(
                        text="⚡ START ENGINE",
                        bg=self.colors['bg_tertiary'],
                        activebackground=self.colors['accent_red']
                    )
                    
                    self.log_message("⏹️ Audio engine stopped", "warning")
                    
        except Exception as e:
            self.log_message(f"❌ Engine error: {e}", "error")
            messagebox.showerror("Engine Error", f"Failed to toggle engine: {e}")

    def _create_action_button(self, parent, text, command):
        """Create a modern action button"""
        btn = tk.Button(parent, text=text, command=command,
                       bg=self.colors['bg_tertiary'],
                       fg=self.colors['text_primary'],
                       activebackground=self.colors['accent_orange'],
                       activeforeground=self.colors['text_primary'],
                       font=('Segoe UI', 10, 'bold'),
                       relief='raised', bd=2,
                       padx=15, pady=8,
                       cursor='hand2')
        btn.pack(side=tk.LEFT, padx=(0, 10))
        return btn
    
    def toggle_network_panel(self):
        """Toggle network panel visibility"""
        if self.network_panel_visible:
            self.network_panel.pack_forget()
            self.network_panel_visible = False
        else:
            self.network_panel.pack(fill=tk.X, pady=(0, 20))
            self.network_panel_visible = True
    
    def open_routing_window(self):
        """Open the routing matrix in a separate window"""
        if self.routing_window and self.routing_window.winfo_exists():
            self.routing_window.lift()
            return
            
        self.routing_window = tk.Toplevel(self.root)
        self.routing_window.title("ToneSphere - Routing Matrix")
        self.routing_window.geometry("1000x700")
        self.routing_window.configure(bg=self.colors['bg_primary'])
        
        # Create routing interface
        self._create_routing_interface(self.routing_window)
    
    def _create_routing_interface(self, parent):
        """Create the visual routing interface with interactive canvas"""
        from tonesphere.gui.interactive_canvas import InteractiveRoutingCanvas
        
        main_frame = tk.Frame(parent, bg=self.colors['bg_primary'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title and controls bar
        title_frame = tk.Frame(main_frame, bg=self.colors['bg_primary'])
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(title_frame, text="Audio Routing Matrix",
                              bg=self.colors['bg_primary'],
                              fg=self.colors['accent_orange'],
                              font=('Segoe UI', 20, 'bold'))
        title_label.pack(side=tk.LEFT)
        
        # Canvas controls
        controls_frame = tk.Frame(title_frame, bg=self.colors['bg_primary'])
        controls_frame.pack(side=tk.RIGHT)
        
        # Zoom controls
        tk.Label(controls_frame, text="Zoom:",
                bg=self.colors['bg_primary'],
                fg=self.colors['text_secondary'],
                font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        zoom_out_btn = tk.Button(controls_frame, text="−",
                                command=lambda: self.interactive_canvas.zoom_out() if hasattr(self, 'interactive_canvas') else None,
                                bg=self.colors['bg_tertiary'],
                                fg=self.colors['text_primary'],
                                font=('Segoe UI', 14, 'bold'),
                                width=2, relief='raised', bd=2,
                                cursor='hand2')
        zoom_out_btn.pack(side=tk.LEFT, padx=2)
        
        zoom_in_btn = tk.Button(controls_frame, text="+",
                               command=lambda: self.interactive_canvas.zoom_in() if hasattr(self, 'interactive_canvas') else None,
                               bg=self.colors['bg_tertiary'],
                               fg=self.colors['text_primary'],
                               font=('Segoe UI', 14, 'bold'),
                               width=2, relief='raised', bd=2,
                               cursor='hand2')
        zoom_in_btn.pack(side=tk.LEFT, padx=2)
        
        # Center button
        center_btn = tk.Button(controls_frame, text="⊙ Center",
                              command=lambda: self.interactive_canvas.center_content() if hasattr(self, 'interactive_canvas') else None,
                              bg=self.colors['accent_orange'],
                              fg=self.colors['text_primary'],
                              activebackground=self.colors['accent_red'],
                              font=('Segoe UI', 10, 'bold'),
                              relief='raised', bd=2,
                              padx=15, pady=5,
                              cursor='hand2')
        center_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # Canvas container with border
        canvas_container = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                                   relief='raised', bd=2)
        canvas_container.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Create canvas
        self.routing_canvas = tk.Canvas(canvas_container, 
                                       bg=self.colors['bg_primary'],
                                       highlightthickness=0,
                                       cursor="crosshair")
        self.routing_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize interactive canvas controller
        self.interactive_canvas = InteractiveRoutingCanvas(self.routing_canvas, self.colors)
        
        # Set up callbacks
        self.interactive_canvas.on_connection_created = self._on_canvas_connection_created
        self.interactive_canvas.on_connection_deleted = self._on_canvas_connection_deleted
        self.interactive_canvas.on_device_moved = self._on_canvas_device_moved
        
        # Instructions panel
        instructions_frame = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                                     relief='raised', bd=2)
        instructions_frame.pack(fill=tk.X, pady=(0, 15))
        
        instructions_inner = tk.Frame(instructions_frame, bg=self.colors['bg_secondary'])
        instructions_inner.pack(fill=tk.X, padx=15, pady=10)
        
        tk.Label(instructions_inner, text="💡 Controls:",
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_orange'],
                font=('Segoe UI', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 15))
        
        instructions_text = (
            "🖱️ Left-Click & Drag: Move devices  |  "
            "🔗 Click connection point & drag: Create route  |  "
            "🗑️ Right-Click connection: Delete  |  "
            "🖱️ Middle-Click & Drag: Pan  |  "
            "🔍 Mouse Wheel: Zoom  |  "
            "⊙ Center button: Reset view"
        )
        
        tk.Label(instructions_inner, text=instructions_text,
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_secondary'],
                font=('Segoe UI', 9)).pack(side=tk.LEFT)
        
        # Control panel for manual routing
        control_frame = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                                relief='raised', bd=2)
        control_frame.pack(fill=tk.X, ipady=10)
        
        control_inner = tk.Frame(control_frame, bg=self.colors['bg_secondary'])
        control_inner.pack(fill=tk.X, padx=20, pady=10)
        
        # Routing controls
        tk.Label(control_inner, text="Manual Routing",
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_orange'],
                font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 8))
        
        routing_controls = tk.Frame(control_inner, bg=self.colors['bg_secondary'])
        routing_controls.pack(fill=tk.X)
        
        tk.Label(routing_controls, text="Source:", 
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.source_var = tk.StringVar()
        self.source_combo = ttk.Combobox(routing_controls, textvariable=self.source_var, 
                                        width=20, font=('Segoe UI', 9))
        self.source_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(routing_controls, text="→", 
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_orange'],
                font=('Segoe UI', 14, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(routing_controls, text="Destination:", 
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.dest_var = tk.StringVar()
        self.dest_combo = ttk.Combobox(routing_controls, textvariable=self.dest_var, 
                                      width=20, font=('Segoe UI', 9))
        self.dest_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(routing_controls, text="Volume:", 
                bg=self.colors['bg_secondary'],
                fg=self.colors['text_primary'],
                font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=(0, 5))
        
        self.volume_var = tk.DoubleVar(value=1.0)
        volume_scale = ttk.Scale(routing_controls, from_=0.0, to=2.0, 
                                variable=self.volume_var, length=80)
        volume_scale.pack(side=tk.LEFT, padx=(0, 8))
        
        self.volume_label = tk.Label(routing_controls, text="1.00",
                                    bg=self.colors['bg_secondary'],
                                    fg=self.colors['text_secondary'],
                                    font=('Segoe UI', 10, 'bold'),
                                    width=4)
        self.volume_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Update volume label
        volume_scale.configure(command=lambda v: self.volume_label.configure(text=f"{float(v):.2f}"))
        
        # Create routing button
        create_btn = tk.Button(routing_controls, text="🔗 Create",
                              command=self.create_routing,
                              bg=self.colors['accent_orange'],
                              fg=self.colors['text_primary'],
                              activebackground=self.colors['accent_red'],
                              font=('Segoe UI', 10, 'bold'),
                              relief='raised', bd=2,
                              padx=15, pady=4,
                              cursor='hand2')
        create_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Clear all button
        clear_btn = tk.Button(routing_controls, text="🗑️ Clear All",
                             command=self.clear_all_routes,
                             bg=self.colors['error'],
                             fg=self.colors['text_primary'],
                             activebackground='#cc0000',
                             font=('Segoe UI', 10, 'bold'),
                             relief='raised', bd=2,
                             padx=15, pady=4,
                             cursor='hand2')
        clear_btn.pack(side=tk.RIGHT)
        
        # Active routes list
        routes_frame = tk.Frame(main_frame, bg=self.colors['bg_secondary'],
                               relief='raised', bd=2)
        routes_frame.pack(fill=tk.BOTH, expand=True)
        
        routes_header = tk.Frame(routes_frame, bg=self.colors['bg_secondary'])
        routes_header.pack(fill=tk.X, padx=15, pady=(10, 8))
        
        tk.Label(routes_header, text="Active Routes",
                bg=self.colors['bg_secondary'],
                fg=self.colors['accent_orange'],
                font=('Segoe UI', 12, 'bold')).pack(side=tk.LEFT)
        
        # Routes list
        routes_list_frame = tk.Frame(routes_frame, bg=self.colors['bg_secondary'])
        routes_list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))
        
        # Create treeview for routes
        routes_columns = ('Source', 'Destination', 'Volume', 'Status')
        self.routes_tree = ttk.Treeview(routes_list_frame, columns=routes_columns,
                                       show='headings', height=6)
        
        # Configure routes treeview
        for col in routes_columns:
            self.routes_tree.heading(col, text=col)
            if col in ['Volume', 'Status']:
                self.routes_tree.column(col, width=80)
            else:
                self.routes_tree.column(col, width=200)
        
        # Routes scrollbar
        routes_scrollbar = ttk.Scrollbar(routes_list_frame, orient=tk.VERTICAL,
                                        command=self.routes_tree.yview)
        self.routes_tree.configure(yscrollcommand=routes_scrollbar.set)
        
        self.routes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        routes_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind double-click to delete route
        self.routes_tree.bind('<Double-1>', self.on_route_double_click)
        
        # Context menu for routes
        self.routes_context_menu = tk.Menu(self.routing_window, tearoff=0,
                                          bg=self.colors['bg_secondary'],
                                          fg=self.colors['text_primary'])
        self.routes_context_menu.add_command(label="Delete Route", command=self.delete_selected_route)
        self.routes_context_menu.add_command(label="Toggle Mute", command=self.toggle_route_mute)
        self.routes_context_menu.add_separator()
        self.routes_context_menu.add_command(label="Set Volume...", command=self.set_route_volume)
        
        # Bind right-click to show context menu
        self.routes_tree.bind('<Button-3>', self.show_routes_context_menu)
        
        # Initial load
        self.refresh_routing_display()

    def _on_canvas_connection_created(self, source_id: int, dest_id: int):
        """Handle connection created from canvas"""
        if self.engine:
            success, message = self.engine.create_routing(source_id, dest_id, 1.0)
            if success:
                self.log_message(f"✅ {message}: {source_id} → {dest_id}", "success")
                self.refresh_routing_display()
                self.refresh_active_routes()
            else:
                self.log_message(f"⚠️ {message}", "warning")
    
    def _on_canvas_connection_deleted(self, source_id: int, dest_id: int):
        """Handle connection deleted from canvas"""
        if self.engine:
            if messagebox.askyesno("Delete Connection", 
                                  f"Delete routing from device {source_id} to {dest_id}?"):
                if self.engine.remove_routing(source_id, dest_id):
                    self.log_message(f"🗑️ Deleted routing: {source_id} → {dest_id}", "warning")
                    self.refresh_routing_display()
                    self.refresh_active_routes()
    
    def _on_canvas_device_moved(self):
        """Handle device moved on canvas"""
        # Optional: Save device positions to config
        pass
    
    def start_network(self):
        """Start network streaming"""
        try:
            if self.engine:
                self.engine.start_network_streaming()
                self.log_message("✓ Network streaming started on port 9001")
        except Exception as e:
            self.log_message(f"✗ Failed to start network streaming: {e}")
    
    def stop_network(self):
        """Stop network streaming"""
        try:
            if self.engine:
                self.engine.stop_network_streaming()
                self.log_message("✓ Network streaming stopped")
        except Exception as e:
            self.log_message(f"✗ Error stopping network streaming: {e}")
    
    def refresh_devices(self):
        """Refresh device list"""
        if not self.engine:
            return
            
        try:
            # Clear existing items
            for item in self.devices_tree.get_children():
                self.devices_tree.delete(item)
            
            # Get devices and populate tree
            devices = self.engine.get_devices()
            device_options = []
            
            for device in devices:
                self.devices_tree.insert('', 'end', values=(
                    device['id'],
                    device['name'][:30] + '...' if len(device['name']) > 30 else device['name'],
                    device['type'],
                    device['channels'],
                    '✓' if device['is_asio'] else '✗',
                    f"{device['latency_ms']:.1f}ms"
                ))
                device_options.append(f"{device['id']}: {device['name']}")
            
            # Update routing comboboxes if they exist
            try:
                if hasattr(self, 'source_combo') and self.source_combo.winfo_exists():
                    self.source_combo['values'] = device_options
                if hasattr(self, 'dest_combo') and self.dest_combo.winfo_exists():
                    self.dest_combo['values'] = device_options
            except (tk.TclError, AttributeError):
                # Comboboxes don't exist or routing window is closed
                pass
            
            self.log_message(f"✓ Found {len(devices)} audio devices")
            
        except Exception as e:
            self.log_message(f"✗ Error refreshing devices: {e}")
    
    def create_routing(self):
        """Create audio routing with improved feedback"""
        try:
            source_text = self.source_var.get()
            dest_text = self.dest_var.get()
            
            if not source_text or not dest_text:
                messagebox.showwarning("Warning", "Please select both source and destination devices")
                return
            
            # Extract device IDs
            source_id = int(source_text.split(':')[0])
            dest_id = int(dest_text.split(':')[0])
            volume = self.volume_var.get()
            
            if self.engine:
                success, message = self.engine.create_routing(source_id, dest_id, volume)
                if success:
                    self.log_message(f"✅ {message}: {source_id} -> {dest_id} (Volume: {volume:.2f})", "success")
                    # Refresh routing display if window is open
                    if hasattr(self, 'routing_canvas'):
                        self.refresh_routing_display()
                    if hasattr(self, 'routes_tree'):
                        self.refresh_active_routes()
                else:
                    self.log_message(f"⚠️ {message}", "warning")
            else:
                self.log_message("❌ Engine not initialized", "error")
                
        except Exception as e:
            self.log_message(f"❌ Error creating routing: {e}", "error")
            messagebox.showerror("Error", f"Error creating routing: {e}")
    
    def log_message(self, message: str, msg_type: str = "info"):
        """Add message to log with color coding"""
        timestamp = time.strftime("%H:%M:%S")
        
        # Color coding based on message type
        if msg_type == "success":
            color = self.colors['success']
        elif msg_type == "warning":
            color = self.colors['warning']
        elif msg_type == "error":
            color = self.colors['error']
        else:
            color = self.colors['text_primary']
        
        # Insert message with timestamp
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        
        # Apply color to the last line
        line_start = f"{self.log_text.index(tk.END)}-1l"
        line_end = f"{self.log_text.index(tk.END)}-1c"
        
        # Create tag for this message type if it doesn't exist
        tag_name = f"msg_{msg_type}"
        self.log_text.tag_configure(tag_name, foreground=color)
        self.log_text.tag_add(tag_name, line_start, line_end)
        
        self.log_text.see(tk.END)
    
    def refresh_routing_display(self):
        """Refresh the visual routing display with interactive canvas"""
        if not hasattr(self, 'interactive_canvas'):
            return
        
        if not self.engine:
            return
        
        # Clear canvas
        self.interactive_canvas.clear()
        
        # Get devices
        devices = self.engine.get_devices()
        device_options = []
        
        for device in devices:
            device_options.append(f"{device['id']}: {device['name']}")
        
        # Update comboboxes
        try:
            if hasattr(self, 'source_combo') and self.source_combo.winfo_exists():
                self.source_combo['values'] = device_options
            if hasattr(self, 'dest_combo') and self.dest_combo.winfo_exists():
                self.dest_combo['values'] = device_options
        except tk.TclError:
            pass
        
        # Separate inputs and outputs
        inputs = [d for d in devices if 'input' in d['type']]
        outputs = [d for d in devices if 'output' in d['type']]
        
        # Position devices on canvas
        # Inputs on the left
        start_y = 100
        spacing_y = 120
        
        for i, device in enumerate(inputs):
            x = 100
            y = start_y + (i * spacing_y)
            self.interactive_canvas.add_device(
                device['id'], 
                device['name'], 
                device['type'],
                x, y
            )
        
        # Outputs on the right
        for i, device in enumerate(outputs):
            x = 600
            y = start_y + (i * spacing_y)
            self.interactive_canvas.add_device(
                device['id'], 
                device['name'], 
                device['type'],
                x, y
            )
        
        # Add connections
        routing_matrix = self.engine.get_routing_matrix()
        for connection in routing_matrix.values():
            self.interactive_canvas.add_connection(
                connection['source_id'],
                connection['destination_id'],
                connection['volume'],
                connection['muted'],
                connection['solo']
            )
        
        # Redraw and center
        self.interactive_canvas.redraw()
        self.routing_canvas.after(100, self.interactive_canvas.center_content)
        
        # Update routes tree
        if hasattr(self, 'routes_tree'):
            self.refresh_active_routes()
    
    def _draw_device_block(self, x, y, device, device_type):
        """Draw a device block with modern styling"""
        width, height = 160, 60
        
        # Choose color based on device type and status
        if 'virtual' in device['type']:
            if device['is_active']:
                color = self.colors['accent_orange']
                border_color = self.colors['accent_gold']
            else:
                color = '#663311'  # Darker orange for inactive virtual
                border_color = self.colors['accent_orange']
        elif device['is_asio']:
            if device['is_active']:
                color = self.colors['accent_red']
                border_color = '#ff6666'
            else:
                color = '#661111'  # Darker red for inactive ASIO
                border_color = self.colors['accent_red']
        else:
            if device['is_active']:
                color = self.colors['bg_tertiary']
                border_color = self.colors['text_secondary']
            else:
                color = '#1a1a1a'  # Very dark for inactive regular devices
                border_color = self.colors['text_muted']
        
        # Draw main block with rounded corners effect
        self.routing_canvas.create_rectangle(x+2, y+2, x + width-2, y + height-2,
                                           fill='#000000', outline='', width=0)  # Shadow
        self.routing_canvas.create_rectangle(x, y, x + width, y + height,
                                           fill=color, outline=border_color,
                                           width=2)
        
        # Draw device name
        name = device['name'][:18] + '...' if len(device['name']) > 18 else device['name']
        self.routing_canvas.create_text(x + width//2, y + height//2 - 8,
                                      text=name, fill=self.colors['text_primary'],
                                      font=('Segoe UI', 9, 'bold'))
        
        # Draw device info
        info_text = f"Ch:{device['channels']} | {device['latency_ms']:.0f}ms"
        if device['is_asio']:
            info_text = "ASIO | " + info_text
        
        self.routing_canvas.create_text(x + width//2, y + height//2 + 8,
                                      text=info_text, fill=self.colors['text_secondary'],
                                      font=('Segoe UI', 7))
        
        # Draw connection point
        if device_type == 'input':
            # Output connection point on the right
            self.routing_canvas.create_oval(x + width - 8, y + height//2 - 4,
                                          x + width, y + height//2 + 4,
                                          fill=self.colors['accent_gold'], outline=border_color, width=2)
        else:
            # Input connection point on the left
            self.routing_canvas.create_oval(x - 8, y + height//2 - 4,
                                          x, y + height//2 + 4,
                                          fill=self.colors['accent_gold'], outline=border_color, width=2)
    
    def _draw_connection(self, start_pos, end_pos, connection):
        """Draw a routing connection line like a thread/cable"""
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Choose line color and style based on connection state
        if connection['muted']:
            color = self.colors['text_muted']
            width = 2
            dash = (5, 5)  # Dashed line for muted
        elif connection['solo']:
            color = self.colors['accent_gold']
            width = 5
            dash = ()
        else:
            # Color based on volume level
            volume = connection['volume']
            if volume > 1.5:
                color = self.colors['accent_red']  # High volume - red
            elif volume > 1.0:
                color = self.colors['accent_orange']  # Medium-high volume - orange
            elif volume > 0.5:
                color = self.colors['success']  # Normal volume - green
            else:
                color = self.colors['text_secondary']  # Low volume - gray
            width = max(2, int(volume * 3))  # Width based on volume
            dash = ()
        
        # Calculate control points for smooth curve
        mid_x = (x1 + x2) // 2
        control_x1 = x1 + (mid_x - x1) * 0.7
        control_x2 = x2 - (x2 - mid_x) * 0.7
        
        # Draw smooth curved connection using multiple line segments
        points = []
        segments = 20
        for i in range(segments + 1):
            t = i / segments
            # Cubic Bezier curve calculation
            x = (1-t)**3 * x1 + 3*(1-t)**2*t * control_x1 + 3*(1-t)*t**2 * control_x2 + t**3 * x2
            y = (1-t)**3 * y1 + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y2
            points.extend([x, y])
        
        # Draw the curved line
        if dash:
            self.routing_canvas.create_line(points, fill=color, width=width, 
                                          smooth=True, dash=dash, capstyle='round')
        else:
            self.routing_canvas.create_line(points, fill=color, width=width, 
                                          smooth=True, capstyle='round')
        
        # Draw volume indicator near the middle of the connection
        mid_point_x = (x1 + x2) // 2
        mid_point_y = (y1 + y2) // 2
        
        # Volume text
        volume_text = f"{connection['volume']:.1f}x"
        text_bg = self.colors['bg_primary']
        
        # Create background for volume text
        self.routing_canvas.create_rectangle(mid_point_x - 15, mid_point_y - 8,
                                           mid_point_x + 15, mid_point_y + 8,
                                           fill=text_bg, outline=color, width=1)
        
        self.routing_canvas.create_text(mid_point_x, mid_point_y,
                                      text=volume_text, fill=color,
                                      font=('Segoe UI', 8, 'bold'))

    def refresh_active_routes(self):
        """Refresh the active routes display"""
        if not hasattr(self, 'routes_tree'):
            return
            
        # Clear existing items
        for item in self.routes_tree.get_children():
            self.routes_tree.delete(item)
        
        if not self.engine:
            return
            
        # Get current routing matrix
        routing_matrix = self.engine.get_routing_matrix()
        devices = self.engine.get_devices()
        
        # Create device lookup for names
        device_lookup = {d['id']: d['name'] for d in devices}
        
        # Populate routes tree
        for connection in routing_matrix.values():
            source_name = device_lookup.get(connection['source_id'], f"ID:{connection['source_id']}")
            dest_name = device_lookup.get(connection['destination_id'], f"ID:{connection['destination_id']}")
            
            # Determine status
            if connection['muted']:
                status = "MUTED"
                status_color = self.colors['error']
            elif connection['solo']:
                status = "SOLO"
                status_color = self.colors['accent_gold']
            else:
                status = "ACTIVE"
                status_color = self.colors['success']
            
            # Insert route into tree
            route_id = f"{connection['source_id']}_{connection['destination_id']}"
            self.routes_tree.insert('', 'end', iid=route_id, values=(
                source_name[:25] + '...' if len(source_name) > 25 else source_name,
                dest_name[:25] + '...' if len(dest_name) > 25 else dest_name,
                f"{connection['volume']:.2f}x",
                status,
                "Double-click to delete"
            ))
    
    def delete_selected_route(self):
        """Delete the selected route"""
        selection = self.routes_tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a route to delete")
            return
            
        route_id = selection[0]
        source_id, dest_id = map(int, route_id.split('_'))
        
        # Confirm deletion
        if messagebox.askyesno("Confirm Deletion", 
                              f"Delete routing from device {source_id} to device {dest_id}?"):
            if self.engine and self.engine.remove_routing(source_id, dest_id):
                self.log_message(f"🗑️ Deleted routing: {source_id} -> {dest_id}", "warning")
                self.refresh_active_routes()
                self.refresh_routing_display()
            else:
                self.log_message(f"❌ Failed to delete routing: {source_id} -> {dest_id}", "error")
    
    def clear_all_routes(self):
        """Clear all active routes"""
        if not self.engine:
            return
            
        routing_matrix = self.engine.get_routing_matrix()
        if not routing_matrix:
            messagebox.showinfo("No Routes", "No active routes to clear")
            return
            
        # Confirm clearing all routes
        if messagebox.askyesno("Confirm Clear All", 
                              f"Delete all {len(routing_matrix)} active routes?"):
            cleared_count = 0
            for connection in list(routing_matrix.values()):
                if self.engine.remove_routing(connection['source_id'], connection['destination_id']):
                    cleared_count += 1
            
            self.log_message(f"🗑️ Cleared {cleared_count} routes", "warning")
            self.refresh_active_routes()
            self.refresh_routing_display()
    
    def on_route_double_click(self, event):
        """Handle double-click on route"""
        self.delete_selected_route()
    
    def show_routes_context_menu(self, event):
        """Show context menu for routes"""
        # Select the item under cursor
        item = self.routes_tree.identify('item', event.x, event.y)
        if item:
            self.routes_tree.selection_set(item)
            self.routes_context_menu.post(event.x_root, event.y_root)
    
    def toggle_route_mute(self):
        """Toggle mute for selected route"""
        selection = self.routes_tree.selection()
        if not selection:
            return
            
        route_id = selection[0]
        source_id, dest_id = map(int, route_id.split('_'))
        
        if self.engine:
            self.engine.routing_matrix.toggle_mute(source_id, dest_id)
            self.log_message(f"🔇 Toggled mute for routing: {source_id} -> {dest_id}", "info")
            self.refresh_active_routes()
            self.refresh_routing_display()
    
    def set_route_volume(self):
        """Set volume for selected route"""
        selection = self.routes_tree.selection()
        if not selection:
            return
            
        route_id = selection[0]
        source_id, dest_id = map(int, route_id.split('_'))
        
        # Get current volume
        routing_matrix = self.engine.get_routing_matrix()
        current_volume = routing_matrix.get(route_id, {}).get('volume', 1.0)
        
        # Ask for new volume
        new_volume = tk.simpledialog.askfloat("Set Volume", 
                                             f"Enter volume for route {source_id} -> {dest_id}:",
                                             initialvalue=current_volume,
                                             minvalue=0.0, maxvalue=2.0)
        
        if new_volume is not None:
            self.engine.set_routing_volume(source_id, dest_id, new_volume)
            self.log_message(f"🔊 Set volume for routing {source_id} -> {dest_id}: {new_volume:.2f}x", "info")
            self.refresh_active_routes()
            self.refresh_routing_display()

    def update_status(self):
        """Update status information"""
        if self.engine:
            if self.engine.is_running:
                self.status_label.configure(text="Engine: Running", 
                                          fg=self.colors['success'])
                
                # Update performance stats
                stats = self.engine.get_performance_stats()
                self.performance_label.configure(text=f"CPU: {stats['cpu_usage']:.1f}% | Latency: {stats['latency_ms']:.1f}ms")
                
                # Update network client count
                clients = self.engine.get_network_clients()
                self.client_count_label.configure(text=f"Clients: {len(clients)}")
            else:
                self.status_label.configure(text="Engine: Stopped",
                                          fg=self.colors['text_secondary'])
        else:
            self.status_label.configure(text="Engine: Not Initialized",
                                      fg=self.colors['error'])
    
    def start_updates(self):
        """Start update thread"""
        def update_loop():
            while True:
                try:
                    self.root.after(0, self.update_status)
                    time.sleep(1.0)
                except:
                    break
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About ToneSphere Studio", 
                          "ToneSphere Studio v1.0\n\n"
                          "Professional Audio Routing Engine\n"
                          "Inspired by VoiceMeeter Potato\n\n"
                          "Features:\n"
                          "• Low-latency ASIO support\n"
                          "• Virtual audio devices\n"
                          "• Network audio streaming\n"
                          "• Professional routing matrix\n"
                          "• Real-time audio effects")
    
    def run(self):
        """Run the GUI"""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()
    
    def on_closing(self):
        """Handle application closing"""
        if self.engine:
            self.engine.stop_engine()
            self.engine.stop_network_streaming()
        self.root.destroy()