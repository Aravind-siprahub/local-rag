import os
from PIL import Image, ImageDraw, ImageFont

ASSETS_DIR = r"d:\local-rag\docs\presentation_assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# Colors matching SipraHub Sales Deck
BRAND_RED = (206, 33, 36)      # #CE2124
DARK_BG = (15, 23, 42)          # Slate-900
SIDEBAR_BG = (24, 33, 47)      # Slate-800
MAIN_BG = (248, 250, 252)       # Slate-50
CARD_BG = (255, 255, 255)       # White
TEXT_DARK = (15, 23, 42)        # Slate-900
TEXT_MUTED = (100, 116, 139)    # Slate-500
BORDER_COLOR = (226, 232, 240)  # Slate-200
USER_BUBBLE = (241, 245, 249)   # Slate-100
TAG_BG = (254, 242, 242)        # Red-50
SUCCESS_GREEN = (22, 163, 74)   # Green-600
SUCCESS_BG = (240, 253, 244)    # Green-50

def get_font(size, bold=False):
    font_names = ["calibrib.ttf" if bold else "calibri.ttf", "segoeuib.ttf" if bold else "segoeui.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in font_names:
        font_path = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", name)
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def create_hero_screenshot():
    w, h = 1600, 960
    img = Image.new("RGB", (w, h), MAIN_BG)
    d = ImageDraw.Draw(img)

    # 1. Sidebar (Width 320)
    d.rectangle([(0, 0), (320, h)], fill=SIDEBAR_BG)
    
    # Sidebar Header / Logo
    d.rectangle([(24, 30), (52, 58)], fill=BRAND_RED)
    d.text((32, 34), "S", font=get_font(22, bold=True), fill=(255, 255, 255))
    d.text((64, 34), "SipraHub", font=get_font(20, bold=True), fill=(255, 255, 255))
    d.text((156, 37), "LOCAL RAG", font=get_font(12, bold=True), fill=BRAND_RED)
    
    # Air-Gapped Sovereign Badge
    d.rounded_rectangle([(24, 80), (296, 120)], radius=6, fill=(30, 41, 59))
    d.text((36, 92), "DEPLOYMENT: 100% AIR-GAPPED", font=get_font(11, bold=True), fill=(148, 163, 184))
    
    # Sidebar Nav Items
    nav_items = [
        ("Chat Assistant", True),
        ("Document Registry", False),
        ("Upload Files", False),
        ("Evaluation Suite", False),
        ("Local Settings", False),
    ]
    y_nav = 145
    for item, active in nav_items:
        if active:
            d.rounded_rectangle([(20, y_nav), (300, y_nav + 44)], radius=8, fill=(30, 41, 59))
            d.rectangle([(20, y_nav + 8), (24, y_nav + 36)], fill=BRAND_RED)
            d.text((44, y_nav + 12), item, font=get_font(15, bold=True), fill=(255, 255, 255))
        else:
            d.text((44, y_nav + 12), item, font=get_font(15), fill=(148, 163, 184))
        y_nav += 52

    # Sidebar Local Inference Card
    d.rounded_rectangle([(20, h - 140), (300, h - 24)], radius=8, fill=(30, 41, 59))
    d.text((32, h - 126), "LOCAL INFERENCE ENGINE", font=get_font(11, bold=True), fill=(148, 163, 184))
    d.ellipse([(32, h - 98), (42, h - 88)], fill=SUCCESS_GREEN)
    d.text((48, h - 102), "Ollama Local Engine", font=get_font(14, bold=True), fill=(255, 255, 255))
    d.text((32, h - 74), "Model: qwen3:8b (100% On-Premise)", font=get_font(12), fill=(148, 163, 184))
    d.text((32, h - 52), "Data Egress: ZERO (Air-Gapped)", font=get_font(12, bold=True), fill=SUCCESS_GREEN)

    # 2. Main Header
    d.rectangle([(320, 0), (w, 70)], fill=CARD_BG)
    d.line([(320, 70), (w, 70)], fill=BORDER_COLOR, width=1)
    d.text((350, 24), "Internal Document Chat  /  100% On-Premise Knowledge Base", font=get_font(16, bold=True), fill=TEXT_DARK)
    
    # Status badges top right
    d.rounded_rectangle([(w - 410, 18), (w - 220, 52)], radius=6, fill=SUCCESS_BG)
    d.text((w - 395, 26), "Local pgvector: Connected", font=get_font(12, bold=True), fill=SUCCESS_GREEN)
    d.rounded_rectangle([(w - 200, 18), (w - 30, 52)], radius=6, fill=TAG_BG)
    d.text((w - 185, 26), "100% Air-Gapped", font=get_font(12, bold=True), fill=BRAND_RED)

    # 3. Main Chat Conversation Area
    user_y = 110
    d.rounded_rectangle([(w - 680, user_y), (w - 60, user_y + 64)], radius=12, fill=USER_BUBBLE, outline=BORDER_COLOR)
    d.text((w - 655, user_y + 12), "User (Enterprise Employee)", font=get_font(11, bold=True), fill=TEXT_MUTED)
    d.text((w - 655, user_y + 32), "what are leave polices there in siprahub ?", font=get_font(15, bold=True), fill=TEXT_DARK)

    # Assistant Response Box
    asst_y = 200
    d.rounded_rectangle([(360, asst_y), (w - 120, asst_y + 540)], radius=12, fill=CARD_BG, outline=BORDER_COLOR)
    
    # Assistant Header
    d.rectangle([(380, asst_y + 18), (404, asst_y + 42)], fill=BRAND_RED)
    d.text((387, asst_y + 20), "S", font=get_font(16, bold=True), fill=(255, 255, 255))
    d.text((416, asst_y + 22), "SipraHub Local AI Assistant", font=get_font(15, bold=True), fill=TEXT_DARK)
    d.rounded_rectangle([(610, asst_y + 20), (810, asst_y + 44)], radius=4, fill=SUCCESS_BG)
    d.text((620, asst_y + 24), "100% LOCALLY GROUNDED", font=get_font(11, bold=True), fill=SUCCESS_GREEN)

    # Content Text
    content_y = asst_y + 64
    d.text((380, content_y), "Based strictly on the local HR Framework document, here are the verified leave policies:", font=get_font(14), fill=TEXT_DARK)
    
    # Policy Cards
    policies = [
        ("1. Casual Leave (CL) Entitlement", "Employees are credited with 1 (one) Casual Leave per month during their employment. Intended for short-term personal requirements."),
        ("2. Carry Forward Rules", "Unused Casual Leave for a given month will be carried forward to subsequent months within the same calendar year."),
        ("3. Year-End Lapsing Policy", "All accumulated Casual Leave must be utilized within the same calendar year. Any unused balance at year-end strictly lapses and will NOT carry over."),
        ("4. Strict Anti-Hallucination Guard", "Notice: Sick leave, maternity leave, and paternity leave are not mentioned in the local files. Unstated rules are strictly refused.")
    ]
    card_y = content_y + 35
    for title, desc in policies:
        d.rounded_rectangle([(380, card_y), (w - 150, card_y + 70)], radius=8, fill=MAIN_BG, outline=BORDER_COLOR)
        d.text((400, card_y + 10), title, font=get_font(13, bold=True), fill=TEXT_DARK)
        d.text((400, card_y + 34), desc, font=get_font(12), fill=TEXT_MUTED)
        card_y += 82

    # Citation provenance bar
    cite_y = asst_y + 440
    d.line([(380, cite_y), (w - 150, cite_y)], fill=BORDER_COLOR, width=1)
    d.text((380, cite_y + 15), "LOCAL PROVENANCE & CITATIONS:", font=get_font(11, bold=True), fill=TEXT_MUTED)
    
    d.rounded_rectangle([(380, cite_y + 38), (840, cite_y + 78)], radius=6, fill=TAG_BG, outline=BRAND_RED)
    d.text((395, cite_y + 48), "[Source] New HR Framework (3) 1.docx  |  Chunk #2  (Cosine: 0.89)", font=get_font(12, bold=True), fill=BRAND_RED)
    
    d.rounded_rectangle([(860, cite_y + 38), (1260, cite_y + 78)], radius=6, fill=USER_BUBBLE, outline=BORDER_COLOR)
    d.text((875, cite_y + 48), "Local Reranker: FlashRank Score 0.94", font=get_font(12), fill=TEXT_DARK)

    # 4. Bottom Prompt Input Bar
    bar_y = h - 95
    d.rounded_rectangle([(360, bar_y), (w - 120, bar_y + 60)], radius=10, fill=CARD_BG, outline=BORDER_COLOR)
    d.text((385, bar_y + 20), "Ask any question from local company documents (100% private & on-premise)...", font=get_font(14), fill=TEXT_MUTED)
    d.rounded_rectangle([(w - 230, bar_y + 10), (w - 135, bar_y + 50)], radius=6, fill=BRAND_RED)
    d.text((w - 200, bar_y + 18), "Ask", font=get_font(14, bold=True), fill=(255, 255, 255))

    img.save(os.path.join(ASSETS_DIR, "hero_app.png"), "PNG")
    print("Updated hero_app.png (Fully Local)")

def create_doc_management_screenshot():
    w, h = 1600, 960
    img = Image.new("RGB", (w, h), MAIN_BG)
    d = ImageDraw.Draw(img)

    # Sidebar (Width 320)
    d.rectangle([(0, 0), (320, h)], fill=SIDEBAR_BG)
    d.rectangle([(24, 30), (52, 58)], fill=BRAND_RED)
    d.text((32, 34), "S", font=get_font(22, bold=True), fill=(255, 255, 255))
    d.text((64, 34), "SipraHub", font=get_font(20, bold=True), fill=(255, 255, 255))
    d.text((156, 37), "LOCAL RAG", font=get_font(12, bold=True), fill=BRAND_RED)

    # Header
    d.rectangle([(320, 0), (w, 70)], fill=CARD_BG)
    d.line([(320, 70), (w, 70)], fill=BORDER_COLOR, width=1)
    d.text((350, 24), "On-Premise Document Registry & Vector Storage", font=get_font(18, bold=True), fill=TEXT_DARK)
    
    # Upload Button Top Right
    d.rounded_rectangle([(w - 240, 16), (w - 40, 54)], radius=6, fill=BRAND_RED)
    d.text((w - 215, 24), "+ Ingest Local File", font=get_font(13, bold=True), fill=(255, 255, 255))

    # Metric Cards Row
    cards = [
        ("On-Premise Documents", "14 Files", "Indexed locally with zero cloud upload"),
        ("Local Vector Chunks", "342 Chunks", "1500 chars / 300 overlap window"),
        ("Local Embedding Model", "nomic-embed", "768-dim vectors running on Ollama"),
        ("Air-Gapped Security", "100% Isolated", "Zero network requests outside LAN")
    ]
    x_card = 350
    for title, val, sub in cards:
        d.rounded_rectangle([(x_card, 95), (x_card + 280, 185)], radius=8, fill=CARD_BG, outline=BORDER_COLOR)
        d.text((x_card + 20, 110), title, font=get_font(11, bold=True), fill=TEXT_MUTED)
        d.text((x_card + 20, 130), val, font=get_font(22, bold=True), fill=TEXT_DARK)
        d.text((x_card + 20, 162), sub, font=get_font(11), fill=BRAND_RED)
        x_card += 305

    # Table Container
    table_y = 210
    d.rounded_rectangle([(350, table_y), (w - 40, h - 40)], radius=8, fill=CARD_BG, outline=BORDER_COLOR)
    
    d.rectangle([(350, table_y), (w - 40, table_y + 45)], fill=USER_BUBBLE)
    d.text((375, table_y + 14), "LOCAL DOCUMENT NAME", font=get_font(12, bold=True), fill=TEXT_MUTED)
    d.text((750, table_y + 14), "FORMAT", font=get_font(12, bold=True), fill=TEXT_MUTED)
    d.text((870, table_y + 14), "CHUNKS", font=get_font(12, bold=True), fill=TEXT_MUTED)
    d.text((990, table_y + 14), "FILE SIZE", font=get_font(12, bold=True), fill=TEXT_MUTED)
    d.text((1110, table_y + 14), "STATUS", font=get_font(12, bold=True), fill=TEXT_MUTED)
    d.text((1280, table_y + 14), "STORAGE", font=get_font(12, bold=True), fill=TEXT_MUTED)

    rows = [
        ("New HR Framework (3) 1.docx", ".DOCX", "10 Chunks", "45.2 KB", "LOCAL INDEXED", "On-Premise"),
        ("SipraHub_PRD_v1.1 1.docx", ".DOCX", "24 Chunks", "112.8 KB", "LOCAL INDEXED", "On-Premise"),
        ("Engineering_Architecture_v2.pdf", ".PDF", "18 Chunks", "84.5 KB", "LOCAL INDEXED", "On-Premise"),
        ("Company_Security_Guidelines.docx", ".DOCX", "8 Chunks", "38.1 KB", "LOCAL INDEXED", "On-Premise"),
        ("Employee_Onboarding_Handbook.pdf", ".PDF", "32 Chunks", "154.0 KB", "LOCAL INDEXED", "On-Premise"),
        ("Product_Roadmap_2026.xlsx", ".XLSX", "14 Chunks", "62.4 KB", "LOCAL INDEXED", "On-Premise"),
        ("API_Integration_Guide.md", ".MD", "12 Chunks", "28.6 KB", "LOCAL INDEXED", "On-Premise")
    ]
    row_y = table_y + 55
    for name, fmt, chunks, size, status, stor in rows:
        d.line([(350, row_y - 8), (w - 40, row_y - 8)], fill=BORDER_COLOR, width=1)
        d.text((375, row_y + 12), name, font=get_font(13, bold=True), fill=TEXT_DARK)
        d.text((750, row_y + 12), fmt, font=get_font(12), fill=TEXT_MUTED)
        d.text((870, row_y + 12), chunks, font=get_font(12), fill=TEXT_DARK)
        d.text((990, row_y + 12), size, font=get_font(12), fill=TEXT_MUTED)
        
        d.rounded_rectangle([(1105, row_y + 8), (1240, row_y + 36)], radius=12, fill=SUCCESS_BG)
        d.text((1120, row_y + 14), status, font=get_font(10, bold=True), fill=SUCCESS_GREEN)
        
        d.text((1285, row_y + 12), stor, font=get_font(12, bold=True), fill=TEXT_DARK)
        row_y += 58

    img.save(os.path.join(ASSETS_DIR, "doc_management.png"), "PNG")
    print("Updated doc_management.png (Fully Local)")

def create_architecture_diagram():
    w, h = 1600, 960
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)

    # Title Banner
    d.text((50, 30), "SipraHub 100% Fully Local RAG — Air-Gapped Technical Architecture", font=get_font(24, bold=True), fill=TEXT_DARK)
    d.text((50, 70), "Zero Cloud Dependencies • Zero External API Calls • 100% On-Premise Data Sovereignty", font=get_font(14, bold=True), fill=BRAND_RED)
    d.line([(50, 105), (w - 50, 105)], fill=BORDER_COLOR, width=2)

    cols = [
        ("1. LOCAL CLIENT LAYER", "Enterprise React 18 UI", [
            ("Local Web Browser", "React 18 + Vite running on intranet / localhost"),
            ("Real-Time SSE Client", "Immediate token streaming directly over LAN"),
            ("Citation Inspector", "Clickable source badges showing local chunk IDs"),
            ("Local File Ingest UI", "Direct file upload to local private server")
        ], BRAND_RED),
        ("2. LOCAL API & ROUTER", "FastAPI Asynchronous Engine", [
            ("FastAPI Backend", "Local Python 3.12 service with asyncpg connection"),
            ("Local Query Normalizer", "In-memory typo correction and shorthand expansion"),
            ("Local Intent Router", "Instant rule-based document QA classification"),
            ("Token Budget Controller", "Caps local generation to 1024 tokens for high speed")
        ], (30, 41, 59)),
        ("3. LOCAL SEARCH & RERANK", "On-Premise Hybrid Search", [
            ("Local pgvector Store", "768d Cosine distance (<=>) search on local PostgreSQL"),
            ("Local Lexical tsvector", "PostgreSQL full-text keyword indexing on local SSD"),
            ("FlashRank Reranker", "In-process neural cross-encoder (TinyBERT) on CPU"),
            ("Strict Grounding Engine", "Prompt injection shielding & anti-hallucination rules")
        ], BRAND_RED),
        ("4. LOCAL INFERENCE & AI", "100% On-Premise Execution", [
            ("Local Ollama Server", "qwen3:8b running completely on private hardware"),
            ("Local Embeddings", "nomic-embed-text (768d) embedding model in Ollama"),
            ("Local Vector Storage", "On-premise PostgreSQL database with pgvector"),
            ("Zero Data Egress", "100% Air-Gapped: Zero packets leave the local network")
        ], (30, 41, 59))
    ]

    x = 50
    col_w = 345
    for col_title, col_sub, items, border_c in cols:
        d.rounded_rectangle([(x, 130), (x + col_w, h - 60)], radius=10, fill=MAIN_BG, outline=border_c, width=2)
        
        # Header Box
        d.rounded_rectangle([(x + 15, 150), (x + col_w - 15, 210)], radius=6, fill=border_c)
        d.text((x + 25, 162), col_title, font=get_font(13, bold=True), fill=(255, 255, 255))
        d.text((x + 25, 184), col_sub, font=get_font(11), fill=(226, 232, 240))

        y_item = 230
        for it_title, it_desc in items:
            d.rounded_rectangle([(x + 15, y_item), (x + col_w - 15, y_item + 125)], radius=8, fill=CARD_BG, outline=BORDER_COLOR)
            d.text((x + 28, y_item + 16), it_title, font=get_font(13, bold=True), fill=TEXT_DARK)
            d.text((x + 28, y_item + 46), it_desc, font=get_font(11), fill=TEXT_MUTED)
            d.line([(x + 28, y_item + 100), (x + col_w - 28, y_item + 100)], fill=TAG_BG, width=3)
            y_item += 145

        if x + col_w < w - 100:
            arrow_x = x + col_w + 12
            d.text((arrow_x, 480), "-->", font=get_font(18, bold=True), fill=BRAND_RED)

        x += col_w + 45

    img.save(os.path.join(ASSETS_DIR, "architecture_diagram.png"), "PNG")
    print("Updated architecture_diagram.png (Fully Local)")

if __name__ == "__main__":
    create_hero_screenshot()
    create_doc_management_screenshot()
    create_architecture_diagram()
