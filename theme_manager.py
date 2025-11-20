import streamlit as st

class ThemeManager:
    """Manages light and dark theme switching with enhanced color palette"""
    
    LIGHT_THEME = {
        'name': 'light',

        # Background + text
        'background': '#F7FAFC',        
        'primary_text': '#1A2A33',      
        'secondary_text': '#4B5F6A',    

        # Logo-matching accents
        'accent': '#0D3B66',            
        'accent_secondary': '#1BA9F5',  

        'logo': 'assets/athena_logo_light.png',

        # UI boxes
        'card_bg': '#FFFFFF',
        'border': '#C7D7E5',            
        'input_bg': '#FFFFFF',          
        'input_border': '#8BB6D9',      
        'hover_bg': '#E9F4FF',          

        # Buttons 
        'button_bg': '#0D3B66',         
        'button_text': '#FFFFFF',

        # Additional UI elements
        'success_color': '#06D6A0',
        'error_color': '#EF476F',
        'warning_color': '#FFB627',
        'info_color': '#1BA9F5',
}


    
    DARK_THEME = {
        'name': 'dark',
        
        # Background + text
        'background': '#0A191C',
        'primary_text': '#DCEFF0',
        'secondary_text': '#3DCCC7',
        
        # Teal accents
        'accent': '#3DCCC7',
        'accent_secondary': '#6AA84F',
        
        'logo': 'assets/athena_logo_dark.png',
        
        # UI boxes
        'card_bg': '#142B30',
        'border': '#1F3A3F',
        'input_bg': '#1A2F33',
        'input_border': '#2A4A4F',
        'hover_bg': '#1F3A40',
        
        # Additional UI elements
        'success_color': '#06D6A0',
        'error_color': '#EF476F',
        'warning_color': '#FFB627',
        'info_color': '#3DCCC7',
    }
    
    @staticmethod
    def initialize():
        """Initialize theme in session state"""
        if 'theme' not in st.session_state:
            st.session_state.theme = 'dark'
    
    @staticmethod
    def get_current_theme():
        """Get the current theme dictionary"""
        ThemeManager.initialize()
        if st.session_state.theme == 'dark':
            return ThemeManager.DARK_THEME
        return ThemeManager.LIGHT_THEME
    
    @staticmethod
    def toggle_theme():
        """Toggle between light and dark theme"""
        ThemeManager.initialize()
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'
    
    @staticmethod
    def is_dark():
        """Check if dark theme is active"""
        ThemeManager.initialize()
        return st.session_state.theme == 'dark'
    
    @staticmethod
    def is_light():
        """Check if light theme is active"""
        return not ThemeManager.is_dark()
    
    @staticmethod
    def get_status_color(status: str):
        """Get color for status indicators"""
        theme = ThemeManager.get_current_theme()
        status_map = {
            'success': theme['success_color'],
            'error': theme['error_color'],
            'warning': theme['warning_color'],
            'info': theme['info_color'],
        }
        return status_map.get(status.lower(), theme['accent'])