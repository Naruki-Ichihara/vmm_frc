"""
Theme color palette and stylesheet for VMM-FRC application.
Supports automatic light/dark mode switching based on system settings.
"""

import darkdetect


def is_dark_mode():
    """Detect if system is using dark mode."""
    try:
        return darkdetect.isDark()
    except Exception:
        return True  # Default to dark mode if detection fails


# Dark Mode Color Palette
COLORS_DARK = {
    # Background colors
    'bg_primary': '#2b2b2b',       # Main background
    'bg_secondary': '#353535',     # Ribbon, panels
    'bg_tertiary': '#404040',      # Selected items, hover
    'bg_input': '#3c3c3c',         # Input fields
    'bg_hover': '#4a4a4a',         # Hover state
    'bg_tab_hover': '#3a3a3a',     # Tab hover

    # Border colors
    'border': '#555555',
    'border_hover': '#666666',

    # Text colors
    'text_primary': '#e0e0e0',
    'text_secondary': '#999999',
    'text_muted': '#888888',
    'text_disabled': '#666666',
    'text_white': '#ffffff',

    # Accent colors
    'accent': '#0078d4',           # Primary accent (blue)
    'accent_hover': '#1084d8',
    'accent_success': '#4caf50',   # Green for success/import
    'accent_success_hover': '#45a049',
    'accent_success_border': '#2e7d32',
    'accent_info': '#7fb3d5',      # Light blue for info

    # Chart/Plot colors
    'chart_bg': '#2a2a2a',
    'chart_text': '#e0e0e0',
    'chart_grid': '#444444',
    'chart_spine': '#555555',

    # 3D viewer colors
    'viewer_bg': '#2a2a2a',
    'viewer_bg_export': '#808080',

    # Scrollbar colors
    'scrollbar_handle': '#505050',
    'scrollbar_handle_hover': '#606060',
    'slider_handle': '#606060',
    'slider_handle_hover': '#707070',

    # ROI colors (for multi-ROI visualization)
    'roi_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],

    # Axis colors
    'axis_x': '#FF0000',
    'axis_y': '#00FF00',
    'axis_z': '#0000FF',
}

# Light Mode Color Palette
COLORS_LIGHT = {
    # Background colors
    'bg_primary': '#f5f5f5',       # Main background
    'bg_secondary': '#e8e8e8',     # Ribbon, panels
    'bg_tertiary': '#d0d0d0',      # Selected items, hover
    'bg_input': '#ffffff',         # Input fields
    'bg_hover': '#e0e0e0',         # Hover state
    'bg_tab_hover': '#dcdcdc',     # Tab hover

    # Border colors
    'border': '#c0c0c0',
    'border_hover': '#a0a0a0',

    # Text colors
    'text_primary': '#1a1a1a',
    'text_secondary': '#555555',
    'text_muted': '#666666',
    'text_disabled': '#999999',
    'text_white': '#ffffff',

    # Accent colors
    'accent': '#0078d4',           # Primary accent (blue)
    'accent_hover': '#1084d8',
    'accent_success': '#4caf50',   # Green for success/import
    'accent_success_hover': '#45a049',
    'accent_success_border': '#2e7d32',
    'accent_info': '#5a9fd4',      # Light blue for info

    # Chart/Plot colors
    'chart_bg': '#ffffff',
    'chart_text': '#1a1a1a',
    'chart_grid': '#cccccc',
    'chart_spine': '#999999',

    # 3D viewer colors
    'viewer_bg': '#e0e0e0',
    'viewer_bg_export': '#808080',

    # Scrollbar colors
    'scrollbar_handle': '#c0c0c0',
    'scrollbar_handle_hover': '#a0a0a0',
    'slider_handle': '#b0b0b0',
    'slider_handle_hover': '#909090',

    # ROI colors (for multi-ROI visualization)
    'roi_colors': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],

    # Axis colors
    'axis_x': '#FF0000',
    'axis_y': '#00FF00',
    'axis_z': '#0000FF',
}


def get_colors():
    """Get the appropriate color palette based on system theme."""
    return COLORS_DARK if is_dark_mode() else COLORS_LIGHT


# For backward compatibility - this will be set dynamically
COLORS = get_colors()


def get_stylesheet():
    """Generate the theme stylesheet based on system settings."""
    c = get_colors()
    return f"""
        QMainWindow, QWidget {{
            background-color: {c['bg_primary']};
            color: {c['text_primary']};
        }}
        QGroupBox {{
            border: 1px solid {c['border']};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
            color: {c['text_primary']};
        }}
        QPushButton {{
            background-color: {c['bg_tertiary']};
            border: 1px solid {c['border']};
            border-radius: 4px;
            padding: 5px 10px;
            color: {c['text_primary']};
        }}
        QPushButton:hover {{
            background-color: {c['bg_hover']};
            border-color: {c['border_hover']};
        }}
        QPushButton:pressed {{
            background-color: {c['bg_secondary']};
        }}
        QPushButton:disabled {{
            background-color: {c['bg_secondary']};
            color: {c['text_disabled']};
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
            background-color: {c['bg_input']};
            border: 1px solid {c['border']};
            border-radius: 3px;
            padding: 3px;
            color: {c['text_primary']};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 16px;
            border-left: 1px solid {c['border']};
            border-bottom: 1px solid {c['border']};
            background-color: {c['bg_secondary']};
        }}
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
            background-color: {c['bg_tertiary']};
        }}
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {{
            background-color: {c['accent']};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 4px solid {c['text_primary']};
            width: 0px;
            height: 0px;
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 16px;
            border-left: 1px solid {c['border']};
            border-top: 1px solid {c['border']};
            background-color: {c['bg_secondary']};
        }}
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {c['bg_tertiary']};
        }}
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{
            background-color: {c['accent']};
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            image: none;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 4px solid {c['text_primary']};
            width: 0px;
            height: 0px;
        }}
        QComboBox {{
            padding-right: 20px;
        }}
        QComboBox:hover {{
            border-color: {c['accent']};
        }}
        QComboBox:focus {{
            border-color: {c['accent']};
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid {c['border']};
            background-color: {c['bg_secondary']};
        }}
        QComboBox::drop-down:hover {{
            background-color: {c['bg_tertiary']};
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {c['text_primary']};
        }}
        QComboBox QAbstractItemView {{
            background-color: {c['bg_input']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
            selection-background-color: {c['accent']};
            selection-color: {c['text_white']};
            outline: none;
        }}
        QComboBox QAbstractItemView::item {{
            padding: 4px 8px;
            min-height: 20px;
        }}
        QComboBox QAbstractItemView::item:hover {{
            background-color: {c['bg_tertiary']};
        }}
        QComboBox QAbstractItemView::item:selected {{
            background-color: {c['accent']};
            color: {c['text_white']};
        }}
        QSlider::groove:horizontal {{
            background: {c['bg_tertiary']};
            height: 6px;
            border-radius: 3px;
        }}
        QSlider::handle:horizontal {{
            background: {c['slider_handle']};
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }}
        QSlider::handle:horizontal:hover {{
            background: {c['slider_handle_hover']};
        }}
        QCheckBox, QRadioButton {{
            color: {c['text_primary']};
            spacing: 6px;
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border: 2px solid {c['border']};
            border-radius: 3px;
            background-color: {c['bg_input']};
        }}
        QCheckBox::indicator:hover {{
            border-color: {c['accent']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {c['accent']};
            border-color: {c['accent']};
        }}
        QCheckBox::indicator:disabled {{
            background-color: {c['bg_secondary']};
            border-color: {c['text_disabled']};
        }}
        QRadioButton::indicator {{
            width: 16px;
            height: 16px;
            border: 2px solid {c['border']};
            border-radius: 9px;
            background-color: {c['bg_input']};
        }}
        QRadioButton::indicator:hover {{
            border-color: {c['accent']};
        }}
        QRadioButton::indicator:checked {{
            background-color: {c['accent']};
            border-color: {c['accent']};
        }}
        QRadioButton::indicator:disabled {{
            background-color: {c['bg_secondary']};
            border-color: {c['text_disabled']};
        }}
        QTabBar::tab {{
            background-color: {c['bg_secondary']};
            border: 1px solid {c['border']};
            padding: 6px 12px;
            color: {c['text_primary']};
        }}
        QTabBar::tab:selected {{
            background-color: {c['bg_tertiary']};
            border-bottom: 2px solid {c['accent']};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {c['bg_tab_hover']};
        }}
        QScrollBar:vertical {{
            background: {c['bg_primary']};
            width: 12px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background: {c['scrollbar_handle']};
            border-radius: 6px;
            min-height: 20px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {c['scrollbar_handle_hover']};
        }}
        QScrollBar:horizontal {{
            background: {c['bg_primary']};
            height: 12px;
            border-radius: 6px;
        }}
        QScrollBar::handle:horizontal {{
            background: {c['scrollbar_handle']};
            border-radius: 6px;
            min-width: 20px;
        }}
        QScrollBar::add-line, QScrollBar::sub-line {{
            width: 0;
            height: 0;
        }}
        QProgressBar {{
            background-color: {c['bg_secondary']};
            border: 1px solid {c['border']};
            border-radius: 3px;
            text-align: center;
            color: {c['text_primary']};
        }}
        QProgressBar::chunk {{
            background-color: {c['accent']};
            border-radius: 2px;
        }}
        QStatusBar {{
            background-color: {c['bg_primary']};
            border-top: 1px solid {c['border']};
            color: {c['text_primary']};
        }}
        QMenuBar {{
            background-color: {c['bg_primary']};
            color: {c['text_primary']};
        }}
        QMenuBar::item:selected {{
            background-color: {c['bg_tertiary']};
        }}
        QMenu {{
            background-color: {c['bg_primary']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
        }}
        QMenu::item:selected {{
            background-color: {c['bg_tertiary']};
        }}
        QLabel {{
            color: {c['text_primary']};
        }}
        QFrame {{
            color: {c['text_primary']};
        }}
        QSplitter::handle {{
            background-color: {c['bg_tertiary']};
        }}
        QToolTip {{
            background-color: {c['bg_input']};
            color: {c['text_primary']};
            border: 1px solid {c['border']};
        }}
    """


def get_ribbon_button_style():
    """Get stylesheet for ribbon buttons."""
    c = get_colors()
    return f"""
        QPushButton {{
            text-align: center;
            padding: 3px 4px;
            border: 1px solid {c['border']};
            background-color: {c['bg_tertiary']};
            color: {c['text_primary']};
            font-size: 10px;
            border-radius: 3px;
        }}
        QPushButton:hover {{
            background-color: {c['bg_hover']};
            border: 1px solid {c['accent']};
        }}
        QPushButton:pressed {{
            background-color: {c['bg_secondary']};
            border: 1px solid {c['accent']};
        }}
    """


def get_ribbon_combobox_style():
    """Get stylesheet for ribbon comboboxes."""
    c = get_colors()
    return f"""
        QComboBox {{
            border: 1px solid {c['border']};
            border-radius: 3px;
            padding: 3px 5px;
            background-color: {c['bg_input']};
            color: {c['text_primary']};
            font-size: 11px;
        }}
        QComboBox:hover {{
            border: 1px solid {c['accent']};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-style: solid;
            border-width: 4px 3px 0px 3px;
            border-color: {c['text_primary']} transparent transparent transparent;
        }}
    """


def get_import_button_style():
    """Get stylesheet for import/action buttons."""
    c = get_colors()
    return f"""
        QPushButton {{
            text-align: center;
            padding: 8px 16px;
            border: 1px solid {c['accent_success_border']};
            background-color: {c['accent_success']};
            color: {c['text_white']};
            font-size: 12px;
            font-weight: bold;
            border-radius: 4px;
        }}
        QPushButton:hover {{
            background-color: {c['accent_success_hover']};
        }}
        QPushButton:disabled {{
            background-color: {c['bg_input']};
            border: 1px solid {c['border']};
            color: {c['text_disabled']};
        }}
    """


def get_toolbar_style():
    """Get stylesheet for toolbar frames."""
    c = get_colors()
    return f"QFrame {{ background-color: {c['bg_secondary']}; border-bottom: 1px solid {c['border']}; border-radius: 5px; padding: 10px; }}"


def get_ribbon_stack_style():
    """Get stylesheet for ribbon stack widget."""
    c = get_colors()
    return f"QStackedWidget {{ background-color: {c['bg_secondary']}; border-bottom: 1px solid {c['border']}; }}"


def get_ribbon_frame_style():
    """Get stylesheet for ribbon frames."""
    c = get_colors()
    return f"""
        QFrame {{ background-color: {c['bg_secondary']}; }}
        QGroupBox {{
            margin-top: 6px;
            padding-top: 8px;
            padding-bottom: 2px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 6px;
            padding: 0 3px;
        }}
    """


def get_splitter_style():
    """Get stylesheet for splitter."""
    c = get_colors()
    return f"""
        QSplitter::handle {{
            background-color: {c['border']};
            width: 1px;
            margin: 0px;
            padding: 0px;
        }}
        QSplitter::handle:hover {{
            background-color: {c['border_hover']};
        }}
    """


def get_progress_bar_style():
    """Get stylesheet for progress bar."""
    c = get_colors()
    return f"""
        QProgressBar {{
            border: 1px solid {c['border']};
            border-radius: 4px;
            text-align: center;
            font-size: 12px;
            height: 20px;
            background-color: {c['bg_secondary']};
            color: {c['text_primary']};
        }}
        QProgressBar::chunk {{
            background-color: {c['accent']};
            border-radius: 3px;
        }}
    """


def get_status_bar_style():
    """Get stylesheet for status bar."""
    c = get_colors()
    return f"QStatusBar {{ border-top: 1px solid {c['border']}; }}"


def get_main_window_style():
    """Get stylesheet for main window."""
    c = get_colors()
    return f"""
        QMainWindow {{
            background-color: {c['bg_primary']};
        }}
        QTabBar::tab {{
            background-color: {c['bg_secondary']};
            color: {c['text_primary']};
            padding: 8px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }}
        QTabBar::tab:selected {{
            background-color: {c['bg_tertiary']};
            border-bottom: 2px solid {c['accent']};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {c['bg_tab_hover']};
        }}
    """


def get_group_style():
    """Get stylesheet for group boxes."""
    c = get_colors()
    return f"""
        QGroupBox {{
            font-weight: bold;
            border: 1px solid {c['border']};
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }}
    """


def get_viewer_frame_style():
    """Get stylesheet for 3D viewer frames."""
    c = get_colors()
    return f"QFrame {{ background-color: {c['chart_bg']}; border: 1px solid {c['border']}; }}"


def get_viewer_title_style():
    """Get stylesheet for viewer title labels."""
    c = get_colors()
    return f"color: {c['text_white']}; font-weight: bold; background-color: {c['chart_grid']}; padding: 2px;"


def get_mpl_theme():
    """Get matplotlib theme settings based on system theme."""
    c = get_colors()
    return {
        'figure.facecolor': c['bg_primary'],
        'axes.facecolor': c['chart_bg'],
        'axes.edgecolor': c['border'],
        'axes.labelcolor': c['text_primary'],
        'text.color': c['text_primary'],
        'xtick.color': c['text_primary'],
        'ytick.color': c['text_primary'],
        'grid.color': c['chart_grid'],
        'legend.facecolor': c['bg_secondary'],
        'legend.edgecolor': c['border'],
    }


def apply_mpl_theme():
    """Apply theme to matplotlib based on system settings."""
    import matplotlib.pyplot as plt
    for key, value in get_mpl_theme().items():
        plt.rcParams[key] = value


def get_splash_colors():
    """Get colors for splash screen based on system theme."""
    if is_dark_mode():
        return {
            'bg': '#0d0d0d',
            'bg_bottom': '#0a0a0a',
            'accent': '#00a8ff',
            'text_title': '#ffffff',
            'text_subtitle': '#aaaaaa',
            'text_version': '#00a8ff',
            'text_loading': '#777777',
            'text_copyright': '#666666',
            'progress_bg': '#1a1a1a',
            'progress_chunk': '#00a8ff',
        }
    else:
        return {
            'bg': '#f8f8f8',
            'bg_bottom': '#e8e8e8',
            'accent': '#0078d4',
            'text_title': '#1a1a1a',
            'text_subtitle': '#555555',
            'text_version': '#0078d4',
            'text_loading': '#666666',
            'text_copyright': '#888888',
            'progress_bg': '#e0e0e0',
            'progress_chunk': '#0078d4',
        }
