"""
Interactive Routing Canvas for ToneSphere
Provides zoomable, pannable canvas with drag-and-drop functionality
"""

import tkinter as tk
from typing import Dict, List, Tuple, Optional, Callable
import math


class DeviceBlock:
    """Represents a draggable device block on the canvas"""
    
    def __init__(self, device_id: int, name: str, device_type: str, 
                 x: float, y: float, width: float = 180, height: float = 70):
        self.device_id = device_id
        self.name = name
        self.device_type = device_type
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.canvas_ids = []  # Store canvas item IDs
        self.is_selected = False
        self.connection_point = None  # (x, y) for connection
        
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside the device block"""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def get_connection_point(self, is_input: bool) -> Tuple[float, float]:
        """Get the connection point coordinates"""
        center_y = self.y + self.height / 2
        if is_input:
            return (self.x, center_y)
        else:
            return (self.x + self.width, center_y)


class Connection:
    """Represents a routing connection between devices"""
    
    def __init__(self, source_id: int, dest_id: int, volume: float = 1.0,
                 muted: bool = False, solo: bool = False):
        self.source_id = source_id
        self.dest_id = dest_id
        self.volume = volume
        self.muted = muted
        self.solo = solo
        self.canvas_ids = []  # Store canvas item IDs
        self.is_hovered = False
        
    def get_key(self) -> str:
        """Get unique key for this connection"""
        return f"{self.source_id}_{self.dest_id}"


class InteractiveRoutingCanvas:
    """
    Interactive canvas for audio routing with zoom, pan, and drag-and-drop
    """
    
    def __init__(self, canvas: tk.Canvas, colors: Dict[str, str]):
        self.canvas = canvas
        self.colors = colors
        
        # Canvas state
        self.zoom_level = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 3.0
        self.zoom_step = 0.1
        
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        
        # Device blocks and connections
        self.device_blocks: Dict[int, DeviceBlock] = {}
        self.connections: Dict[str, Connection] = {}
        
        # Interaction state
        self.is_panning = False
        self.pan_start_x = 0
        self.pan_start_y = 0
        
        self.dragging_device = None
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        self.drawing_connection = False
        self.connection_start_device = None
        self.temp_connection_line = None
        
        self.hovered_connection = None
        
        # Callbacks
        self.on_connection_created: Optional[Callable] = None
        self.on_connection_deleted: Optional[Callable] = None
        self.on_device_moved: Optional[Callable] = None
        
        # Bind events
        self._bind_events()
        
    def _bind_events(self):
        """Bind mouse and keyboard events"""
        # Mouse wheel for zoom
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Button-4>", self._on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mouse_wheel)  # Linux scroll down
        
        # Middle mouse button for panning
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        
        # Left mouse button for dragging and selection
        self.canvas.bind("<ButtonPress-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        
        # Right mouse button for context menu / delete
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)
        
        # Mouse motion for hover effects
        self.canvas.bind("<Motion>", self._on_mouse_move)
        
        # Keyboard shortcuts
        self.canvas.bind("<Control-0>", lambda e: self.center_content())
        self.canvas.bind("<plus>", lambda e: self.zoom_in())
        self.canvas.bind("<minus>", lambda e: self.zoom_out())
        
    def _screen_to_canvas(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen coordinates to canvas coordinates"""
        canvas_x = (screen_x - self.pan_offset_x) / self.zoom_level
        canvas_y = (screen_y - self.pan_offset_y) / self.zoom_level
        return (canvas_x, canvas_y)
    
    def _canvas_to_screen(self, canvas_x: float, canvas_y: float) -> Tuple[float, float]:
        """Convert canvas coordinates to screen coordinates"""
        screen_x = canvas_x * self.zoom_level + self.pan_offset_x
        screen_y = canvas_y * self.zoom_level + self.pan_offset_y
        return (screen_x, screen_y)
    
    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming"""
        # Get mouse position
        mouse_x = event.x
        mouse_y = event.y
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:
            # Zoom out
            self.zoom_at_point(mouse_x, mouse_y, -self.zoom_step)
        elif event.num == 4 or event.delta > 0:
            # Zoom in
            self.zoom_at_point(mouse_x, mouse_y, self.zoom_step)
    
    def zoom_at_point(self, screen_x: float, screen_y: float, delta: float):
        """Zoom in/out at a specific point"""
        # Calculate canvas coordinates before zoom
        canvas_x, canvas_y = self._screen_to_canvas(screen_x, screen_y)
        
        # Update zoom level
        old_zoom = self.zoom_level
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, self.zoom_level + delta))
        
        # Adjust pan offset to keep the point under the mouse
        self.pan_offset_x = screen_x - canvas_x * self.zoom_level
        self.pan_offset_y = screen_y - canvas_y * self.zoom_level
        
        # Redraw
        self.redraw()
    
    def zoom_in(self):
        """Zoom in centered"""
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2
        self.zoom_at_point(center_x, center_y, self.zoom_step)
    
    def zoom_out(self):
        """Zoom out centered"""
        center_x = self.canvas.winfo_width() / 2
        center_y = self.canvas.winfo_height() / 2
        self.zoom_at_point(center_x, center_y, -self.zoom_step)
    
    def _on_pan_start(self, event):
        """Start panning"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        self.canvas.config(cursor="fleur")
    
    def _on_pan_move(self, event):
        """Pan the canvas"""
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            
            self.redraw()
    
    def _on_pan_end(self, event):
        """End panning"""
        self.is_panning = False
        self.canvas.config(cursor="")
    
    def _on_left_click(self, event):
        """Handle left mouse button press"""
        canvas_x, canvas_y = self._screen_to_canvas(event.x, event.y)
        
        # Check if clicking on a device
        clicked_device = self._get_device_at_point(canvas_x, canvas_y)
        
        if clicked_device:
            # Check if clicking on connection point
            if self._is_near_connection_point(clicked_device, canvas_x, canvas_y):
                # Start drawing connection
                self.drawing_connection = True
                self.connection_start_device = clicked_device
                self.temp_connection_line = None
            else:
                # Start dragging device
                self.dragging_device = clicked_device
                self.drag_start_x = canvas_x - clicked_device.x
                self.drag_start_y = canvas_y - clicked_device.y
                self.canvas.config(cursor="hand2")
    
    def _on_left_drag(self, event):
        """Handle left mouse button drag"""
        canvas_x, canvas_y = self._screen_to_canvas(event.x, event.y)
        
        if self.dragging_device:
            # Move the device
            self.dragging_device.x = canvas_x - self.drag_start_x
            self.dragging_device.y = canvas_y - self.drag_start_y
            self.redraw()
            
        elif self.drawing_connection:
            # Draw temporary connection line
            if self.connection_start_device:
                start_point = self.connection_start_device.get_connection_point(
                    'input' not in self.connection_start_device.device_type
                )
                screen_start = self._canvas_to_screen(*start_point)
                
                # Delete old temp line
                if self.temp_connection_line:
                    self.canvas.delete(self.temp_connection_line)
                
                # Draw new temp line
                self.temp_connection_line = self.canvas.create_line(
                    screen_start[0], screen_start[1], event.x, event.y,
                    fill=self.colors['accent_orange'], width=3, dash=(5, 5)
                )
    
    def _on_left_release(self, event):
        """Handle left mouse button release"""
        canvas_x, canvas_y = self._screen_to_canvas(event.x, event.y)
        
        if self.dragging_device:
            # Finish dragging
            self.dragging_device = None
            self.canvas.config(cursor="")
            if self.on_device_moved:
                self.on_device_moved()
                
        elif self.drawing_connection:
            # Finish drawing connection
            target_device = self._get_device_at_point(canvas_x, canvas_y)
            
            if target_device and target_device != self.connection_start_device:
                # Create connection
                if 'input' in self.connection_start_device.device_type:
                    source = self.connection_start_device
                    dest = target_device
                else:
                    source = target_device
                    dest = self.connection_start_device
                
                if self.on_connection_created:
                    self.on_connection_created(source.device_id, dest.device_id)
            
            # Clean up
            if self.temp_connection_line:
                self.canvas.delete(self.temp_connection_line)
                self.temp_connection_line = None
            
            self.drawing_connection = False
            self.connection_start_device = None
    
    def _on_right_click(self, event):
        """Handle right mouse button click"""
        canvas_x, canvas_y = self._screen_to_canvas(event.x, event.y)
        
        # Check if clicking on a connection
        connection = self._get_connection_at_point(canvas_x, canvas_y)
        
        if connection and self.on_connection_deleted:
            # Delete the connection
            self.on_connection_deleted(connection.source_id, connection.dest_id)
    
    def _on_mouse_move(self, event):
        """Handle mouse movement for hover effects"""
        canvas_x, canvas_y = self._screen_to_canvas(event.x, event.y)
        
        # Check for connection hover
        connection = self._get_connection_at_point(canvas_x, canvas_y)
        
        if connection != self.hovered_connection:
            self.hovered_connection = connection
            self.redraw()
    
    def _get_device_at_point(self, x: float, y: float) -> Optional[DeviceBlock]:
        """Get device block at the given point"""
        for device in self.device_blocks.values():
            if device.contains_point(x, y):
                return device
        return None
    
    def _is_near_connection_point(self, device: DeviceBlock, x: float, y: float, threshold: float = 15) -> bool:
        """Check if point is near the connection point of a device"""
        is_input = 'input' not in device.device_type
        conn_point = device.get_connection_point(is_input)
        distance = math.sqrt((x - conn_point[0])**2 + (y - conn_point[1])**2)
        return distance < threshold / self.zoom_level
    
    def _get_connection_at_point(self, x: float, y: float, threshold: float = 10) -> Optional[Connection]:
        """Get connection at the given point"""
        for connection in self.connections.values():
            source_device = self.device_blocks.get(connection.source_id)
            dest_device = self.device_blocks.get(connection.dest_id)
            
            if source_device and dest_device:
                start = source_device.get_connection_point(False)
                end = dest_device.get_connection_point(True)
                
                # Check distance to line segment
                if self._point_to_line_distance(x, y, start[0], start[1], end[0], end[1]) < threshold / self.zoom_level:
                    return connection
        
        return None
    
    def _point_to_line_distance(self, px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
        """Calculate distance from point to line segment"""
        line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
        if line_len_sq == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)
        
        return math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    def add_device(self, device_id: int, name: str, device_type: str, x: float, y: float):
        """Add a device block to the canvas"""
        device = DeviceBlock(device_id, name, device_type, x, y)
        self.device_blocks[device_id] = device
    
    def add_connection(self, source_id: int, dest_id: int, volume: float = 1.0, 
                      muted: bool = False, solo: bool = False):
        """Add a connection to the canvas"""
        connection = Connection(source_id, dest_id, volume, muted, solo)
        self.connections[connection.get_key()] = connection
    
    def remove_connection(self, source_id: int, dest_id: int):
        """Remove a connection from the canvas"""
        key = f"{source_id}_{dest_id}"
        if key in self.connections:
            del self.connections[key]
    
    def clear(self):
        """Clear all devices and connections"""
        self.device_blocks.clear()
        self.connections.clear()
        try:
            self.canvas.delete("all")
        except tk.TclError:
            # Canvas might be destroyed, ignore
            pass
    
    def center_content(self):
        """Center the view on all content"""
        if not self.device_blocks:
            return
        
        # Calculate bounding box of all devices
        min_x = min(d.x for d in self.device_blocks.values())
        max_x = max(d.x + d.width for d in self.device_blocks.values())
        min_y = min(d.y for d in self.device_blocks.values())
        max_y = max(d.y + d.height for d in self.device_blocks.values())
        
        content_width = max_x - min_x
        content_height = max_y - min_y
        content_center_x = (min_x + max_x) / 2
        content_center_y = (min_y + max_y) / 2
        
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate zoom to fit
        zoom_x = canvas_width / (content_width + 100)  # Add padding
        zoom_y = canvas_height / (content_height + 100)
        target_zoom = min(zoom_x, zoom_y, self.max_zoom)
        target_zoom = max(target_zoom, self.min_zoom)
        
        # Center the content
        self.zoom_level = target_zoom
        self.pan_offset_x = canvas_width / 2 - content_center_x * self.zoom_level
        self.pan_offset_y = canvas_height / 2 - content_center_y * self.zoom_level
        
        self.redraw()
    
    def redraw(self):
        """Redraw all content"""
        self.canvas.delete("all")
        
        # Draw grid
        self._draw_grid()
        
        # Draw connections first (behind devices)
        for connection in self.connections.values():
            self._draw_connection(connection)
        
        # Draw devices
        for device in self.device_blocks.values():
            self._draw_device(device)
        
        # Draw zoom level indicator
        self._draw_zoom_indicator()
    
    def _draw_grid(self):
        """Draw background grid"""
        grid_size = 50 * self.zoom_level
        
        if grid_size < 10:  # Don't draw if too small
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate grid start positions
        start_x = self.pan_offset_x % grid_size
        start_y = self.pan_offset_y % grid_size
        
        # Draw vertical lines
        x = start_x
        while x < canvas_width:
            self.canvas.create_line(x, 0, x, canvas_height, 
                                   fill=self.colors['bg_tertiary'], width=1)
            x += grid_size
        
        # Draw horizontal lines
        y = start_y
        while y < canvas_height:
            self.canvas.create_line(0, y, canvas_width, y, 
                                   fill=self.colors['bg_tertiary'], width=1)
            y += grid_size
    
    def _draw_device(self, device: DeviceBlock):
        """Draw a device block"""
        # Convert to screen coordinates
        screen_x, screen_y = self._canvas_to_screen(device.x, device.y)
        screen_width = device.width * self.zoom_level
        screen_height = device.height * self.zoom_level
        
        # Choose colors based on device type
        if 'virtual' in device.device_type:
            color = self.colors['accent_orange']
            border_color = self.colors['accent_gold']
        elif 'input' in device.device_type:
            color = self.colors['success']
            border_color = '#66ff66'
        else:
            color = self.colors['accent_red']
            border_color = '#ff6666'
        
        # Draw shadow
        self.canvas.create_rectangle(
            screen_x + 3, screen_y + 3,
            screen_x + screen_width + 3, screen_y + screen_height + 3,
            fill='#000000', outline='', width=0
        )
        
        # Draw main rectangle
        self.canvas.create_rectangle(
            screen_x, screen_y,
            screen_x + screen_width, screen_y + screen_height,
            fill=color, outline=border_color, width=2
        )
        
        # Draw device name (scale font with zoom)
        font_size = max(8, int(10 * self.zoom_level))
        name = device.name[:15] + '...' if len(device.name) > 15 else device.name
        self.canvas.create_text(
            screen_x + screen_width / 2, screen_y + screen_height / 2,
            text=name, fill=self.colors['text_primary'],
            font=('Segoe UI', font_size, 'bold')
        )
        
        # Draw connection point
        is_input = 'input' not in device.device_type
        conn_point = device.get_connection_point(is_input)
        screen_conn = self._canvas_to_screen(*conn_point)
        
        point_size = 8 * self.zoom_level
        self.canvas.create_oval(
            screen_conn[0] - point_size, screen_conn[1] - point_size,
            screen_conn[0] + point_size, screen_conn[1] + point_size,
            fill=self.colors['accent_gold'], outline=border_color, width=2
        )
    
    def _draw_connection(self, connection: Connection):
        """Draw a routing connection"""
        source_device = self.device_blocks.get(connection.source_id)
        dest_device = self.device_blocks.get(connection.dest_id)
        
        if not source_device or not dest_device:
            return
        
        # Get connection points
        start = source_device.get_connection_point(False)
        end = dest_device.get_connection_point(True)
        
        # Convert to screen coordinates
        screen_start = self._canvas_to_screen(*start)
        screen_end = self._canvas_to_screen(*end)
        
        # Choose color and style
        if connection.muted:
            color = self.colors['text_muted']
            width = 2
            dash = (5, 5)
        elif connection.solo:
            color = self.colors['accent_gold']
            width = max(3, int(5 * self.zoom_level))
            dash = ()
        else:
            volume = connection.volume
            if volume > 1.5:
                color = self.colors['accent_red']
            elif volume > 1.0:
                color = self.colors['accent_orange']
            elif volume > 0.5:
                color = self.colors['success']
            else:
                color = self.colors['text_secondary']
            width = max(2, int(volume * 3 * self.zoom_level))
            dash = ()
        
        # Highlight if hovered
        if connection == self.hovered_connection:
            width += 2
        
        # Draw curved line using Bezier curve
        points = self._calculate_bezier_curve(screen_start, screen_end)
        
        if dash:
            self.canvas.create_line(*points, fill=color, width=width, 
                                   smooth=True, dash=dash, capstyle='round')
        else:
            self.canvas.create_line(*points, fill=color, width=width, 
                                   smooth=True, capstyle='round')
        
        # Draw volume indicator
        mid_x = (screen_start[0] + screen_end[0]) / 2
        mid_y = (screen_start[1] + screen_end[1]) / 2
        
        volume_text = f"{connection.volume:.1f}x"
        font_size = max(7, int(8 * self.zoom_level))
        
        # Background for text
        self.canvas.create_rectangle(
            mid_x - 20, mid_y - 10,
            mid_x + 20, mid_y + 10,
            fill=self.colors['bg_primary'], outline=color, width=1
        )
        
        self.canvas.create_text(
            mid_x, mid_y,
            text=volume_text, fill=color,
            font=('Segoe UI', font_size, 'bold')
        )
    
    def _calculate_bezier_curve(self, start: Tuple[float, float], 
                                end: Tuple[float, float], segments: int = 20) -> List[float]:
        """Calculate Bezier curve points"""
        x1, y1 = start
        x2, y2 = end
        
        mid_x = (x1 + x2) / 2
        control_x1 = x1 + (mid_x - x1) * 0.7
        control_x2 = x2 - (x2 - mid_x) * 0.7
        
        points = []
        for i in range(segments + 1):
            t = i / segments
            x = (1-t)**3 * x1 + 3*(1-t)**2*t * control_x1 + 3*(1-t)*t**2 * control_x2 + t**3 * x2
            y = (1-t)**3 * y1 + 3*(1-t)**2*t * y1 + 3*(1-t)*t**2 * y2 + t**3 * y2
            points.extend([x, y])
        
        return points
    
    def _draw_zoom_indicator(self):
        """Draw zoom level indicator"""
        zoom_text = f"Zoom: {self.zoom_level:.0%}"
        self.canvas.create_text(
            10, 10,
            text=zoom_text, fill=self.colors['text_secondary'],
            font=('Segoe UI', 9), anchor='nw'
        )
    
    def _screen_to_canvas(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Convert screen coordinates to canvas coordinates (fixed method name)"""
        canvas_x = (screen_x - self.pan_offset_x) / self.zoom_level
        canvas_y = (screen_y - self.pan_offset_y) / self.zoom_level
        return (canvas_x, canvas_y)