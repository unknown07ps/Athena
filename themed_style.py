# themed_style.py

from theme_manager import ThemeManager

def get_themed_style():
    theme = ThemeManager.get_current_theme()

    background = theme['background']
    primary_text = theme['primary_text']
    secondary_text = theme['secondary_text']
    accent = theme['accent']
    card_bg = theme['card_bg']
    border = theme['border']
    input_bg = theme['input_bg']
    input_border = theme['input_border']
    hover_bg = theme['hover_bg']

    return f"""
    <style>

    /* ===========================
        GLOBAL PAGE STYLE
    ============================ */
    body, .stApp {{
        background: {background} !important;
        color: {primary_text} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {accent} !important;
        font-weight: 700 !important;
    }}

    p, label, span {{
        color: {primary_text} !important;
    }}

    /* ===========================
        CARDS + BOXES
    ============================ */
    .result-box, .answer-box, .rag-box, .comparison-box {{
        background: {card_bg};
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid {border};
        color: {primary_text};
        line-height: 1.6;
    }}

    /* ===========================
        INPUT FIELDS
    ============================ */
    .stTextInput > div > input {{
        background: {input_bg} !important;
        color: {primary_text} !important;
        border: 1px solid {input_border} !important;
        border-radius: 6px !important;
    }}

    .stTextInput > div > input:hover {{
        border-color: {accent} !important;
    }}

    /* ===========================
        BUTTONS
    ============================ */
    .stButton > button {{
        background: {accent} !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        border: none !important;
    }}

    .stButton > button:hover {{
        background: {hover_bg} !important;
        color: {primary_text} !important;
        border: 1px solid {accent} !important;
    }}

    /* ===========================
        HR (DIVIDERS)
    ============================ */
    hr {{
        border: none;
        border-top: 1px solid {border} !important;
    }}

    /* ===========================
        SCROLLBAR
    ============================ */
    ::-webkit-scrollbar {{
        width: 10px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {accent};
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-track {{
        background: {background};
    }}
    

    </style>
    """
