"""
Modern GUI Theme for ToneSphere
Professional dark theme with smooth animations and modern styling
"""

import tkinter as tk
from tkinter import ttk


class ModernTheme:
    """Modern professional theme for ToneSphere"""
    
    # Color palette
    COLORS = {
        # Backgrounds
        'bg_primary': '#0d1117',
        'bg_secondary': '#161b22',
        'bg_tertiary': '#21262d',
        'bg_hover': '#30363d',
        'bg_active': '#3d444d',
        
        # Accents
        'accent_primary': '#ff8844',
        'accent_secondary': '#ffaa66',
        'accent_tertiary': '#ffcc88',
        'accent_red': '#ff4444',
        'accent_green': '#44ff88',
        'accent_blue': '#4488ff',
        'accent_purple': '#aa44ff',
        'accent_gold': '#ffdd44',
        
        # Text
        'text_primary': '#f0f6fc',
        'text_secondary': '#c9d1d9',
        'text_muted': '#8b949e',
        'text_disabled': '#484f58',
        
        # Status colors
        'success': '#3fb950',
        'warning': '#d29922',
        'error': '#f85149',
        'info': '#58a6ff',
        
        # Borders
        'border': '#30363d',
        'border_hover': '#484f58',
        'border_active': '#ff8844',
    }
    
    # Fonts
    FONTS = {
        'title': ('Segoe UI', 32, 'bold'),
        'heading': ('Segoe UI', 20, 'bold'),
        'subheading': ('Segoe UI', 16, 'bold'),
        'body': ('Segoe UI', 11),
        'body_bold': ('Segoe UI', 11, 'bold'),
        'small': ('Segoe UI', 9),
        'small_bold': ('Segoe UI', 9, 'bold'),
        'code': ('Consolas', 10),
        'button': ('Segoe UI', 11, 'bold'),
        'button_large': ('Segoe UI', 14, 'bold'),
    }
    
    @classmethod
    def apply_to_root(cls, root: tk.Tk):
        """Apply theme to root window"""
        root.configure(bg=cls.COLORS['bg_primary'])
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure Treeview
        style.configure('Treeview',
                       background=cls.COLORS['bg_tertiary'],
                       foreground=cls.COLORS['text_primary'],
                       fieldbackground=cls.COLORS['bg_tertiary'],
                       borderwidth=0,
                       font=cls.FONTS['body'])
        
        style.configure('Treeview.Heading',
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['accent_primary'],
                       borderwidth=1,
                       relief='flat',
                       font=cls.FONTS['body_bold'])
        
        style.map('Treeview.Heading',
                 background=[('active', cls.COLORS['bg_hover'])])
        
        style.map('Treeview',
                 background=[('selected', cls.COLORS['accent_primary'])],
                 foreground=[('selected', cls.COLORS['bg_primary'])])
        
        # Configure Combobox
        style.configure('TCombobox',
                       fieldbackground=cls.COLORS['bg_tertiary'],
                       background=cls.COLORS['bg_secondary'],
                       foreground=cls.COLORS['text_primary'],
                       arrowcolor=cls.COLORS['accent_primary'],
                       borderwidth=1,
                       relief='flat')
        
        # Configure Scale
        style.configure('TScale',
                       background=cls.COLORS['bg_secondary'],
                       troughcolor=cls.COLORS['bg_tertiary'],
                       borderwidth=0,
                       sliderthickness=20)
        
        # Configure Scrollbar
        style.configure('Vertical.TScrollbar',
                       background=cls.COLORS['bg_secondary'],
                       troughcolor=cls.COLORS['bg_primary'],
                       borderwidth=0,
                       arrowcolor=cls.COLORS['text_secondary'])
        
        style.map('Vertical.TScrollbar',
                 background=[('active', cls.COLORS['bg_hover'])])
    
    @classmethod
    def create_card(cls, parent, **kwargs) -> tk.Frame:
        """Create a modern card frame"""
        defaults = {
            'bg': cls.COLORS['bg_secondary'],
            'relief': 'flat',
            'bd': 0,
            'highlightthickness': 1,
            'highlightbackground': cls.COLORS['border'],
            'highlightcolor': cls.COLORS['border_active']
        }
        defaults.update(kwargs)
        return tk.Frame(parent, **defaults)
    
    @classmethod
    def create_button(cls, parent, text, command, style='primary', **kwargs) -> tk.Button:
        """Create a modern styled button"""
        if style == 'primary':
            bg = cls.COLORS['accent_primary']
            fg = cls.COLORS['bg_primary']
            active_bg = cls.COLORS['accent_secondary']
        elif style == 'secondary':
            bg = cls.COLORS['bg_tertiary']
            fg = cls.COLORS['text_primary']
            active_bg = cls.COLORS['bg_hover']
        elif style == 'success':
            bg = cls.COLORS['success']
            fg = cls.COLORS['bg_primary']
            active_bg = '#2ea043'
        elif style == 'danger':
            bg = cls.COLORS['error']
            fg = cls.COLORS['text_primary']
            active_bg = '#da3633'
        else:
            bg = cls.COLORS['bg_tertiary']
            fg = cls.COLORS['text_primary']
            active_bg = cls.COLORS['bg_hover']
        
        defaults = {
            'bg': bg,
            'fg': fg,
            'activebackground': active_bg,
            'activeforeground': fg,
            'font': cls.FONTS['button'],
            'relief': 'flat',
            'bd': 0,
            'padx': 20,
            'pady': 10,
            'cursor': 'hand2',
            'highlightthickness': 0
        }
        defaults.update(kwargs)
        
        btn = tk.Button(parent, text=text, command=command, **defaults)
        
        # Add hover effects
        def on_enter(e):
            btn['bg'] = active_bg
        
        def on_leave(e):
            btn['bg'] = bg
        
        btn.bind('<Enter>', on_enter)
        btn.bind('<Leave>', on_leave)
        
        return btn
    
    @classmethod
    def create_label(cls, parent, text, style='body', **kwargs) -> tk.Label:
        """Create a modern styled label"""
        defaults = {
            'bg': cls.COLORS['bg_secondary'],
            'fg': cls.COLORS['text_primary'],
            'font': cls.FONTS.get(style, cls.FONTS['body'])
        }
        defaults.update(kwargs)
        return tk.Label(parent, text=text, **defaults)
    
    @classmethod
    def create_entry(cls, parent, **kwargs) -> tk.Entry:
        """Create a modern styled entry"""
        defaults = {
            'bg': cls.COLORS['bg_tertiary'],
            'fg': cls.COLORS['text_primary'],
            'insertbackground': cls.COLORS['accent_primary'],
            'selectbackground': cls.COLORS['accent_primary'],
            'selectforeground': cls.COLORS['bg_primary'],
            'font': cls.FONTS['body'],
            'relief': 'flat',
            'bd': 0,
            'highlightthickness': 1,
            'highlightbackground': cls.COLORS['border'],
            'highlightcolor': cls.COLORS['border_active']
        }
        defaults.update(kwargs)
        return tk.Entry(parent, **defaults)
    
    @classmethod
    def create_text(cls, parent, **kwargs) -> tk.Text:
        """Create a modern styled text widget"""
        defaults = {
            'bg': cls.COLORS['bg_tertiary'],
            'fg': cls.COLORS['text_primary'],
            'insertbackground': cls.COLORS['accent_primary'],
            'selectbackground': cls.COLORS['accent_primary'],
            'selectforeground': cls.COLORS['bg_primary'],
            'font': cls.FONTS['code'],
            'relief': 'flat',
            'bd': 0,
            'highlightthickness': 1,
            'highlightbackground': cls.COLORS['border'],
            'highlightcolor': cls.COLORS['border_active'],
            'wrap': 'word'
        }
        defaults.update(kwargs)
        return tk.Text(parent, **defaults)
    
    @classmethod
    def create_separator(cls, parent, orient='horizontal') -> ttk.Separator:
        """Create a styled separator"""
        return ttk.Separator(parent, orient=orient)
    
    @classmethod
    def create_progress_bar(cls, parent, **kwargs) -> ttk.Progressbar:
        """Create a modern progress bar"""
        style = ttk.Style()
        style.configure('Modern.Horizontal.TProgressbar',
                       troughcolor=cls.COLORS['bg_tertiary'],
                       background=cls.COLORS['accent_primary'],
                       borderwidth=0,
                       thickness=8)
        
        defaults = {
            'style': 'Modern.Horizontal.TProgressbar',
            'mode': 'determinate'
        }
        defaults.update(kwargs)
        return ttk.Progressbar(parent, **defaults)
    
    @classmethod
    def create_menu(cls, parent) -> tk.Menu:
        """Create a modern styled menu"""
        return tk.Menu(parent,
                      bg=cls.COLORS['bg_secondary'],
                      fg=cls.COLORS['text_primary'],
                      activebackground=cls.COLORS['accent_primary'],
                      activeforeground=cls.COLORS['bg_primary'],
                      borderwidth=0,
                      relief='flat')
    
    @classmethod
    def create_tooltip(cls, widget, text):
        """Add a tooltip to a widget"""
        tooltip = None
        
        def show_tooltip(event):
            nonlocal tooltip
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            
            tooltip = tk.Toplevel(widget)
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{x}+{y}")
            
            label = tk.Label(tooltip,
                           text=text,
                           bg=cls.COLORS['bg_hover'],
                           fg=cls.COLORS['text_primary'],
                           font=cls.FONTS['small'],
                           relief='flat',
                           bd=0,
                           padx=8,
                           pady=4)
            label.pack()
        
        def hide_tooltip(event):
            nonlocal tooltip
            if tooltip:
                tooltip.destroy()
                tooltip = None
        
        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)
