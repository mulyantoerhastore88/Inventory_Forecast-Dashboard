import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date, timedelta
import gspread
from google.oauth2.service_account import Credentials
from dateutil.relativedelta import relativedelta
import warnings
from tenacity import retry, stop_after_attempt, wait_exponential
import math
from streamlit_echarts import st_echarts
ECHARTS_AVAILABLE = True
warnings.filterwarnings('ignore')

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Inventory Intelligence Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS KHUSUS PRINT PDF (FIX BLANK PAGE) ---
st.markdown("""
<style>
    @media print {
        /* FIX UTAMA: Reset SEMUA element ke block/visible */
        * {
            overflow: visible !important;
            position: static !important;
            display: block !important;
            float: none !important;
            height: auto !important;
            max-height: none !important;
            width: auto !important;
            max-width: none !important;
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
            break-inside: avoid !important;
        }

        /* Hide unnecessary elements */
        [data-testid="stSidebar"],
        [data-testid="stHeader"],
        .stButton,
        .stDeployButton,
        footer,
        .stDownloadButton,
        .stActionButton,
        button,
        [data-testid="baseButton-secondary"],
        [data-testid="baseButton-primary"],
        .stAlert,
        .stMarkdown:has(button) {
            display: none !important;
            height: 0 !important;
            width: 0 !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }

        /* Force main container to be visible */
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            position: static !important;
            width: 100vw !important;
            height: auto !important;
            margin: 0 !important;
            padding: 0 !important;
            overflow: visible !important;
            display: block !important;
        }

        /* Force all content to be visible */
        section[data-testid="stMain"] > div,
        [data-testid="block-container"] {
            overflow: visible !important;
            height: auto !important;
            max-height: none !important;
            display: block !important;
            position: static !important;
            break-inside: avoid;
        }

        /* Charts and tables - force visibility */
        .element-container,
        .stDataFrame,
        .stPlotlyChart,
        .stAltairChart,
        [data-testid="stHorizontalBlock"] {
            break-inside: avoid-page !important;
            page-break-inside: avoid !important;
            overflow: visible !important;
        }

        /* Ensure text is black for printing */
        body, h1, h2, h3, h4, h5, h6, p, div, span {
            color: #000000 !important;
            background-color: white !important;
        }

        /* Remove shadows and gradients for print */
        .status-indicator,
        .inventory-card,
        .metric-highlight {
            box-shadow: none !important;
            background: white !important;
            border: 1px solid #ccc !important;
        }

        /* Fix for Plotly charts */
        .js-plotly-plot,
        .plotly,
        .plot-container {
            width: 100% !important;
            height: auto !important;
        }

        /* Add page breaks between major sections */
        .stTabs {
            break-after: page !important;
        }

        /* Ensure all content fits page width */
        .row {
            display: block !important;
        }

        .column {
            width: 100% !important;
            float: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- Custom CSS Premium ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .status-indicator {
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .status-indicator:hover {
        transform: translateY(-5px);
    }
    .status-under { 
        background: linear-gradient(135deg, #FF5252 0%, #FF1744 100%);
        color: white;
        border-left: 5px solid #D32F2F;
    }
    .status-accurate { 
        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
        color: white;
        border-left: 5px solid #1B5E20;
    }
    .status-over { 
        background: linear-gradient(135deg, #FF9800 0%, #F57C00 100%);
        color: white;
        border-left: 5px solid #E65100;
    }
    
    .inventory-card {
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .inventory-card:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.12);
    }
    .card-replenish { 
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        color: #EF6C00;
        border: 2px solid #FF9800;
    }
    .card-ideal { 
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        color: #2E7D32;
        border: 2px solid #4CAF50;
    }
    .card-high { 
        background: linear-gradient(135deg, #FFEBEE 0%, #FFCDD2 100%);
        color: #C62828;
        border: 2px solid #F44336;
    }
    
    .metric-highlight {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.15);
        border-top: 5px solid #667eea;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        padding: 10px 0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #F8F9FA 0%, #E9ECEF 100%);
        border-radius: 10px 10px 0 0;
        padding: 12px 24px;
        font-weight: 700;
        font-size: 1rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 2px solid #5a67d8 !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    
    .sankey-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* New CSS */
    .monthly-performance-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid;
    }
    
    .performance-under { border-left-color: #F44336; }
    .performance-accurate { border-left-color: #4CAF50; }
    .performance-over { border-left-color: #FF9800; }
    
    .highlight-row {
        background-color: #FFF9C4 !important;
        font-weight: bold !important;
    }
    
    .warning-badge {
        background: #FF5252;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    .success-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
    }
    
    /* Compact metrics */
    .compact-metric {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 0.5rem 0;
    }
    
    /* Brand performance */
    .brand-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
    }
    
    /* Financial cards */
    .financial-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-top: 4px solid;
        transition: all 0.3s ease;
    }
    .financial-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .card-revenue { border-top-color: #667eea; }
    .card-margin { border-top-color: #4CAF50; }
    .card-cost { border-top-color: #FF9800; }
    .card-inventory { border-top-color: #9C27B0; }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background-color: #0E1117;
            color: #FFFFFF;
        }
        .financial-card, .brand-card, .compact-metric {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
    }
    
    /* Progress bar animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)

# --- Judul Dashboard ---
st.markdown('<h1 class="main-header">💰 FORECAST & INVENTORY CONTROL PRO DASHBOARD</h1>', unsafe_allow_html=True)
st.caption(f"🚀 Inventory Control & Forecast Analytics - D2C Demand Planner Mulyanto | Real-time Insights | Updated: {datetime.now().strftime('%d %B %Y %H:%M')}")

# --- ====================================================== ---
# ---                KONEKSI & LOAD DATA                    ---
# --- ====================================================== ---

@st.cache_resource(show_spinner=False)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_gsheet_connection():
    """Inisialisasi koneksi ke Google Sheets dengan retry mechanism"""
    try:
        skey = st.secrets["gcp_service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(skey, scopes=scopes)
        client = gspread.authorize(credentials)
        return client
    except Exception as e:
        st.error(f"❌ Koneksi Gagal: {str(e)}")
        return None

def validate_month_format(month_str):
    """Validate and standardize month formats"""
    if pd.isna(month_str):
        return datetime.now()
    
    month_str = str(month_str).strip().upper()
    
    # Mapping bulan
    month_map = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    
    formats_to_try = ['%b-%Y', '%b-%y', '%B %Y', '%m/%Y', '%Y-%m']
    
    for fmt in formats_to_try:
        try:
            return datetime.strptime(month_str, fmt)
        except:
            continue
    
    # Fallback: cari bulan dalam string
    for month_name, month_num in month_map.items():
        if month_name in month_str:
            # Cari tahun
            year_part = month_str.replace(month_name, '').replace('-', '').replace(' ', '').strip()
            if year_part and year_part.isdigit():
                year = int('20' + year_part) if len(year_part) == 2 else int(year_part)
            else:
                year = datetime.now().year
            
            return datetime(year, month_num, 1)
    
    return datetime.now()

def add_product_info_to_data(df, df_product):
    """Add Product_Name, Brand, SKU_Tier, Prices from Product_Master to any dataframe"""
    if df.empty or df_product.empty or 'SKU_ID' not in df.columns:
        return df
    
    # Get product info from Product_Master (including prices)
    price_cols = ['Floor_Price', 'Net_Order_Price'] if 'Floor_Price' in df_product.columns and 'Net_Order_Price' in df_product.columns else []
    
    product_info_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status'] + price_cols
    product_info_cols = [col for col in product_info_cols if col in df_product.columns]
    
    product_info = df_product[product_info_cols].copy()
    product_info = product_info.drop_duplicates(subset=['SKU_ID'])
    
    # Remove existing columns if they exist (except SKU_ID)
    cols_to_remove = []
    for col in ['Product_Name', 'Brand', 'SKU_Tier', 'Status', 'Floor_Price', 'Net_Order_Price']:
        if col in df.columns and col != 'SKU_ID':
            cols_to_remove.append(col)
    
    if cols_to_remove:
        df_temp = df.drop(columns=cols_to_remove)
    else:
        df_temp = df.copy()
    
    # Merge with product info
    df_result = pd.merge(df_temp, product_info, on='SKU_ID', how='left')
    return df_result

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
def _safe_read_sheet(spreadsheet, sheet_name, use_records=True):
    ws = spreadsheet.worksheet(sheet_name)
    if use_records:
        return pd.DataFrame(ws.get_all_records())
    else:
        raw_data = ws.get_all_values()
        if len(raw_data) < 2: return pd.DataFrame()
        headers = [str(h).strip() for h in raw_data[0]]
        df = pd.DataFrame(raw_data[1:], columns=headers)
        df = df.loc[:, df.columns != '']
        return df

@st.cache_data(ttl=600, max_entries=3, show_spinner=False)
def load_and_process_data(_client):
    """
    Load semua data termasuk sheet baru: BS_Fullfilment_Cost
    """
    
    gsheet_url = st.secrets["gsheet_url"]
    data = {}

    try:
        spreadsheet = _client.open_by_url(gsheet_url)

        # 1. PRODUCT MASTER
        ws_prod = spreadsheet.worksheet("Product_Master")
        df_product = pd.DataFrame(ws_prod.get_all_records())
        df_product.columns = [col.strip().replace(' ', '_') for col in df_product.columns]
        
        for col in ['Floor_Price', 'Net_Order_Price']:
            if col in df_product.columns:
                df_product[col] = pd.to_numeric(df_product[col], errors='coerce').fillna(0)
        
        if 'Status' not in df_product.columns: df_product['Status'] = 'Active'
        df_product_active = df_product[df_product['Status'].str.upper() == 'ACTIVE'].copy()
        active_skus = df_product_active['SKU_ID'].tolist()
        
        data['product'] = df_product
        data['product_active'] = df_product_active

        # 2. SALES DATA
        ws_sales = spreadsheet.worksheet("Sales")
        df_sales_raw = pd.DataFrame(ws_sales.get_all_records())
        df_sales_raw.columns = [col.strip() for col in df_sales_raw.columns]
        month_cols = [c for c in df_sales_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        if month_cols and 'SKU_ID' in df_sales_raw.columns:
            id_cols = ['SKU_ID']
            for col in ['SKU_Name', 'Product_Name', 'Brand', 'SKU_Tier']:
                if col in df_sales_raw.columns: id_cols.append(col)
            df_sales_long = df_sales_raw.melt(id_vars=id_cols, value_vars=month_cols, var_name='Month_Label', value_name='Sales_Qty')
            df_sales_long['Sales_Qty'] = pd.to_numeric(df_sales_long['Sales_Qty'], errors='coerce').fillna(0)
            df_sales_long['Month'] = df_sales_long['Month_Label'].apply(validate_month_format)
            df_sales_long = df_sales_long[df_sales_long['SKU_ID'].isin(active_skus)]
            df_sales_long = add_product_info_to_data(df_sales_long, df_product)
            data['sales'] = df_sales_long.sort_values('Month')

        # 3. ROFO DATA
        ws_rofo = spreadsheet.worksheet("Rofo")
        df_rofo_raw = pd.DataFrame(ws_rofo.get_all_records())
        df_rofo_raw.columns = [col.strip() for col in df_rofo_raw.columns]
        month_cols_rofo = [c for c in df_rofo_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        if month_cols_rofo:
            id_cols_rofo = ['SKU_ID']
            for col in ['Product_Name', 'Brand']:
                if col in df_rofo_raw.columns: id_cols_rofo.append(col)
            df_rofo_long = df_rofo_raw.melt(id_vars=id_cols_rofo, value_vars=month_cols_rofo, var_name='Month_Label', value_name='Forecast_Qty')
            df_rofo_long['Forecast_Qty'] = pd.to_numeric(df_rofo_long['Forecast_Qty'], errors='coerce').fillna(0)
            df_rofo_long['Month'] = df_rofo_long['Month_Label'].apply(validate_month_format)
            df_rofo_long = df_rofo_long[df_rofo_long['SKU_ID'].isin(active_skus)]
            df_rofo_long = add_product_info_to_data(df_rofo_long, df_product)
            data['forecast'] = df_rofo_long

        # 4. PO DATA
        ws_po = spreadsheet.worksheet("PO")
        df_po_raw = pd.DataFrame(ws_po.get_all_records())
        df_po_raw.columns = [col.strip() for col in df_po_raw.columns]
        month_cols_po = [c for c in df_po_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        if month_cols_po and 'SKU_ID' in df_po_raw.columns:
            df_po_long = df_po_raw.melt(id_vars=['SKU_ID'], value_vars=month_cols_po, var_name='Month_Label', value_name='PO_Qty')
            df_po_long['PO_Qty'] = pd.to_numeric(df_po_long['PO_Qty'], errors='coerce').fillna(0)
            df_po_long['Month'] = df_po_long['Month_Label'].apply(validate_month_format)
            df_po_long = df_po_long[df_po_long['SKU_ID'].isin(active_skus)]
            df_po_long = add_product_info_to_data(df_po_long, df_product)
            data['po'] = df_po_long

        # 5. STOCK DATA
        try:
            df_stock_raw = _safe_read_sheet(spreadsheet, "Stock_Onhand", use_records=False)
        except:
            df_stock_raw = pd.DataFrame()
        if not df_stock_raw.empty:
            col_mapping = {
                'SKU_ID': 'SKU_ID', 'Qty_Available': 'Stock_Qty', 'Product_Code': 'Anchanto_Code',
                'Stock_Category': 'Stock_Category', 'Expiry_Date': 'Expiry_Date', 'Product_Name': 'Product_Name'
            }
            if 'SKU_ID' in df_stock_raw.columns and 'Qty_Available' in df_stock_raw.columns:
                cols_to_use = [c for c in col_mapping.keys() if c in df_stock_raw.columns]
                df_stock = df_stock_raw[cols_to_use].copy()
                df_stock = df_stock.rename(columns=col_mapping)
                df_stock['Stock_Qty'] = pd.to_numeric(df_stock['Stock_Qty'], errors='coerce').fillna(0)
                df_stock['SKU_ID'] = df_stock['SKU_ID'].astype(str).str.strip()
                if 'Floor_Price' in df_product.columns:
                    df_stock = pd.merge(df_stock, df_product[['SKU_ID', 'Floor_Price', 'Net_Order_Price']], on='SKU_ID', how='left')
                data['stock'] = df_stock
            else:
                data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])
        else:
            data['stock'] = pd.DataFrame(columns=['SKU_ID', 'Stock_Qty'])

        # 6. FORECAST 2026 ECOMM
        try:
            ws_ecomm = spreadsheet.worksheet("Forecast_2026_Ecomm")
            df_ecomm_raw = pd.DataFrame(ws_ecomm.get_all_records())
            df_ecomm_raw.columns = [col.strip().replace(' ', '_') for col in df_ecomm_raw.columns]
            month_cols_ecomm = [c for c in df_ecomm_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            for col in month_cols_ecomm:
                df_ecomm_raw[col] = pd.to_numeric(df_ecomm_raw[col], errors='coerce').fillna(0)
            data['ecomm_forecast'] = df_ecomm_raw
            data['ecomm_forecast_month_cols'] = month_cols_ecomm
        except:
            data['ecomm_forecast'] = pd.DataFrame()
            data['ecomm_forecast_month_cols'] = []
        
        # 7. FORECAST 2026 RESELLER
        try:
            ws_reseller = spreadsheet.worksheet("Forecast_2026_Reseller")
            df_reseller_raw = pd.DataFrame(ws_reseller.get_all_records())
            df_reseller_raw.columns = [col.strip().replace(' ', '_') for col in df_reseller_raw.columns]
            all_month_cols_res = [c for c in df_reseller_raw.columns if any(m in c.upper() for m in ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            for col in all_month_cols_res:
                df_reseller_raw[col] = pd.to_numeric(df_reseller_raw[col], errors='coerce').fillna(0)
            
            forecast_start_date = datetime(2026, 1, 1)
            def is_forecast_month(month_str):
                try:
                    month_str = str(month_str).upper().replace('_', ' ').replace('-', ' ')
                    if ' ' in month_str:
                        month_part, year_part = month_str.split(' ')
                        month_num = datetime.strptime(month_part[:3], '%b').month
                        year_clean = ''.join(filter(str.isdigit, year_part))
                        year = 2000 + int(year_clean) if len(year_clean) == 2 else int(year_clean)
                        return datetime(year, month_num, 1) >= forecast_start_date
                except: return False
                return False
            
            hist_cols = [c for c in all_month_cols_res if not is_forecast_month(c)]
            fcst_cols = [c for c in all_month_cols_res if is_forecast_month(c)]
            data['reseller_forecast'] = df_reseller_raw
            data['reseller_all_month_cols'] = all_month_cols_res
            data['reseller_historical_cols'] = hist_cols
            data['reseller_forecast_cols'] = fcst_cols
        except:
            data['reseller_forecast'] = pd.DataFrame()
            data['reseller_all_month_cols'] = []
            data['reseller_historical_cols'] = []
            data['reseller_forecast_cols'] = []

        # ==============================================================================
        # 8. BS FULLFILMENT COST (NEW SHEET)
        # ==============================================================================
        try:
            ws_bs = spreadsheet.worksheet("BS_Fullfilment_Cost")
            df_bs = pd.DataFrame(ws_bs.get_all_records())
            
            # Cleaning Headers & Data
            # Hapus spasi di nama kolom
            df_bs.columns = [c.strip() for c in df_bs.columns]
            
            # Helper untuk bersihkan angka (hapus koma dan persen)
            def clean_currency(x):
                if isinstance(x, str):
                    return pd.to_numeric(x.replace(',', '').replace('%', ''), errors='coerce')
                return x

            # List kolom angka yang perlu dibersihkan
            numeric_cols = ['Total Order(BS)', 'GMV (Fullfil By BS)', 'GMV Total (MP)', 'Total Cost', 'BSA', '%Cost']
            
            for col in numeric_cols:
                if col in df_bs.columns:
                    df_bs[col] = df_bs[col].apply(clean_currency).fillna(0)
            
            # Parse Date (Apr-25)
            df_bs['Month_Date'] = pd.to_datetime(df_bs['Month'], format='%b-%y', errors='coerce')
            df_bs = df_bs.sort_values('Month_Date')
            
            data['fulfillment'] = df_bs
            
        except Exception as e:
            st.warning(f"Gagal load BS_Fullfilment_Cost: {e}")
            data['fulfillment'] = pd.DataFrame()

        # ==============================================================================
        # 9. EXPANSI FULLFILMENT COST (NEW SHEET)
        # ==============================================================================
        try:
            ws_exp = spreadsheet.worksheet("Expansi_Fullfilment_Cost")
            df_exp = pd.DataFrame(ws_exp.get_all_records())
            
            # Cleaning Headers
            df_exp.columns = [c.strip() for c in df_exp.columns]
            
            # Helper untuk bersihkan angka
            def clean_currency_exp(x):
                if isinstance(x, str):
                    return pd.to_numeric(x.replace(',', '').replace('%', ''), errors='coerce')
                return x
        
            # List kolom angka yang perlu dibersihkan
            numeric_cols_exp = ['Total Cost', 'GMV', 'Order_Qty', 'BSA', 'Cost_per_Order']
            
            for col in numeric_cols_exp:
                if col in df_exp.columns:
                    df_exp[col] = df_exp[col].apply(clean_currency_exp).fillna(0)
            
            # Parse %Cost
            if '%Cost' in df_exp.columns:
                df_exp['%Cost'] = df_exp['%Cost'].apply(clean_currency_exp).fillna(0)
            
            # Parse Date (Jan 26, Feb 26, ...) — STRIP dulu karena ada spasi di depan (" Feb 26")
            # yang sebelumnya bikin Month_Date jadi NaT -> sumbu-x chart ke-sort abjad.
            df_exp['Month'] = df_exp['Month'].astype(str).str.strip()
            df_exp['Month_Date'] = pd.to_datetime(df_exp['Month'], format='%b %y', errors='coerce')
            _mask_na = df_exp['Month_Date'].isna()
            if _mask_na.any():
                df_exp.loc[_mask_na, 'Month_Date'] = pd.to_datetime(df_exp.loc[_mask_na, 'Month'], format='%b-%y', errors='coerce')
            _mask_na = df_exp['Month_Date'].isna()
            if _mask_na.any():
                df_exp.loc[_mask_na, 'Month_Date'] = pd.to_datetime(df_exp.loc[_mask_na, 'Month'], errors='coerce')

            df_exp = df_exp.sort_values(['Store', 'Month_Date'])
            
            data['expansi_fulfillment'] = df_exp
            
        except Exception as e:
            st.warning(f"Gagal load Expansi_Fullfilment_Cost: {e}")
            data['expansi_fulfillment'] = pd.DataFrame()

        # ==============================================================================
        # 10. INVENTORY BRANCH (NEW SHEET) - snapshot kesehatan stok per cabang
        # ==============================================================================
        try:
            ws_invb = spreadsheet.worksheet("Inventory Branch")
            df_invb = pd.DataFrame(ws_invb.get_all_records())
            df_invb.columns = [str(c).strip() for c in df_invb.columns]
            df_invb = df_invb.loc[:, [c for c in df_invb.columns if c != '']]

            # Map kolom robust (header 'Replenishment (...)' bisa terpotong di sheet)
            rename_invb = {}
            for c in df_invb.columns:
                cl = c.lower()
                if cl.startswith('branch') or cl == 'store':
                    rename_invb[c] = 'Store'
                elif 'onhand' in cl or cl.startswith('stock'):
                    rename_invb[c] = 'Stock_Onhand'
                elif cl.startswith('replenish'):
                    rename_invb[c] = 'Replenishment'
                elif 'avg' in cl and 'sales' in cl:
                    rename_invb[c] = 'Avg_Sales'
                elif 'cover' in cl:
                    rename_invb[c] = 'Month_Cover'
            df_invb = df_invb.rename(columns=rename_invb)
            if 'Store' not in df_invb.columns and len(df_invb.columns) > 0:
                df_invb = df_invb.rename(columns={df_invb.columns[0]: 'Store'})

            def _clean_num_invb(x):
                if isinstance(x, str):
                    return pd.to_numeric(x.replace(',', '').strip(), errors='coerce')
                return x
            for col in ['Stock_Onhand', 'Replenishment', 'Avg_Sales', 'Month_Cover']:
                if col in df_invb.columns:
                    df_invb[col] = df_invb[col].apply(_clean_num_invb).fillna(0)
                else:
                    df_invb[col] = 0

            df_invb['Store'] = df_invb['Store'].astype(str).str.strip()
            df_invb = df_invb[df_invb['Store'] != '']
            # Recompute cover biar konsisten: stock / avg sales
            df_invb['Month_Cover'] = np.where(df_invb['Avg_Sales'] > 0, df_invb['Stock_Onhand'] / df_invb['Avg_Sales'], 0)

            data['inventory_branch'] = df_invb
        except Exception as e:
            st.warning(f"Gagal load Inventory Branch: {e}")
            data['inventory_branch'] = pd.DataFrame()

        # ==============================================================================
        # 11. EXPANSI REPLENISHMENT (OUTBOUND) (NEW SHEET) - qty kiriman bulanan per cabang
        # ==============================================================================
        try:
            ws_out = spreadsheet.worksheet("Expansi Replenishment(Outbound)")
            df_out_raw = pd.DataFrame(ws_out.get_all_records())
            df_out_raw.columns = [str(c).strip() for c in df_out_raw.columns]
            df_out_raw = df_out_raw.loc[:, [c for c in df_out_raw.columns if c != '']]
            df_out_raw = df_out_raw.rename(columns={df_out_raw.columns[0]: 'Store'})
            df_out_raw['Store'] = df_out_raw['Store'].astype(str).str.strip()
            df_out_raw = df_out_raw[df_out_raw['Store'] != '']

            month_cols_out = [c for c in df_out_raw.columns if any(m in c.upper() for m in
                              ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])]
            for c in month_cols_out:
                df_out_raw[c] = df_out_raw[c].apply(
                    lambda x: pd.to_numeric(str(x).replace(',', '').strip(), errors='coerce') if isinstance(x, str) else x
                )
                df_out_raw[c] = pd.to_numeric(df_out_raw[c], errors='coerce').fillna(0)

            df_out_long = df_out_raw.melt(id_vars=['Store'], value_vars=month_cols_out,
                                          var_name='Month_Label', value_name='Outbound_Qty')
            df_out_long['Outbound_Qty'] = pd.to_numeric(df_out_long['Outbound_Qty'], errors='coerce').fillna(0)
            df_out_long['Month_Date'] = pd.to_datetime(df_out_long['Month_Label'].str.strip(), format='%b-%y', errors='coerce')

            data['outbound_replen'] = df_out_long
            data['outbound_replen_wide'] = df_out_raw
            data['outbound_month_cols'] = month_cols_out
        except Exception as e:
            st.warning(f"Gagal load Expansi Replenishment(Outbound): {e}")
            data['outbound_replen'] = pd.DataFrame()
            data['outbound_replen_wide'] = pd.DataFrame()
            data['outbound_month_cols'] = []

        # ==============================================================================
        # 12. OFFLINE STORE (NEW SHEET) - 2 tabel stack: atas=sales+inventory, bawah=outbound
        # ==============================================================================
        try:
            ws_off = None
            for _cand in ["Offline Store", "Offline_Store", "Offline", "OFFLINE STORE",
                          "Offline Store Sales", "Store Offline", "Offline Store(Sales)"]:
                try:
                    ws_off = spreadsheet.worksheet(_cand)
                    break
                except Exception:
                    continue
            if ws_off is not None:
                _ov = ws_off.get_all_values()
                _hdr = [i for i, r in enumerate(_ov) if r and str(r[0]).strip().lower() == 'store']
                _MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

                def _num_off(x):
                    if isinstance(x, str):
                        return pd.to_numeric(x.replace(',', '').strip(), errors='coerce')
                    return x

                def _parse_off_block(start):
                    header = [str(h).strip() for h in _ov[start]]
                    rows = []
                    for r in _ov[start + 1:]:
                        if not r or str(r[0]).strip() == '':
                            break
                        rows.append((r + [''] * len(header))[:len(header)])
                    dfb = pd.DataFrame(rows, columns=header)
                    dfb = dfb.loc[:, [c for c in dfb.columns if c != '']]
                    dfb = dfb.rename(columns={dfb.columns[0]: 'Store'})
                    dfb['Store'] = dfb['Store'].astype(str).str.strip()
                    return dfb[dfb['Store'] != '']

                if len(_hdr) >= 1:
                    top = _parse_off_block(_hdr[0])
                    mtop = [c for c in top.columns if any(m in c.upper() for m in _MONTHS)]
                    c_onhand = next((c for c in top.columns if 'onhand' in c.lower() or ('stock' in c.lower() and 'cover' not in c.lower())), None)
                    c_avg = next((c for c in top.columns if 'avg' in c.lower() and 'sales' in c.lower()), None)
                    for c in mtop + [x for x in [c_onhand, c_avg] if x]:
                        top[c] = top[c].apply(_num_off).fillna(0)
                    inv = pd.DataFrame({'Store': top['Store'].values})
                    inv['Stock_Onhand'] = top[c_onhand].values if c_onhand else 0
                    inv['Avg_Sales'] = top[c_avg].values if c_avg else (top[mtop].mean(axis=1).values if mtop else 0)
                    inv['Stock_Cover'] = np.where(inv['Avg_Sales'] > 0, inv['Stock_Onhand'] / inv['Avg_Sales'], 0)
                    data['offline_inventory'] = inv
                    if mtop:
                        sl = top.melt(id_vars=['Store'], value_vars=mtop, var_name='Month_Label', value_name='Sales')
                        sl['Sales'] = pd.to_numeric(sl['Sales'], errors='coerce').fillna(0)
                        sl['Month_Date'] = pd.to_datetime(sl['Month_Label'].str.strip(), format='%b-%y', errors='coerce')
                        data['offline_sales_long'] = sl

                if len(_hdr) >= 2:
                    bot = _parse_off_block(_hdr[1])
                    mbot = [c for c in bot.columns if any(m in c.upper() for m in _MONTHS)]
                    for c in mbot:
                        bot[c] = bot[c].apply(_num_off).fillna(0)
                    if mbot:
                        ol = bot.melt(id_vars=['Store'], value_vars=mbot, var_name='Month_Label', value_name='Outbound')
                        ol['Outbound'] = pd.to_numeric(ol['Outbound'], errors='coerce').fillna(0)
                        ol['Month_Date'] = pd.to_datetime(ol['Month_Label'].str.strip(), format='%b-%y', errors='coerce')
                        data['offline_outbound_long'] = ol
        except Exception as e:
            st.warning(f"Gagal load Offline Store: {e}")

        return data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return {}
        

# --- FUNGSI BARU: LOAD DATA RESELLER LENGKAP ---
@st.cache_data(ttl=600, show_spinner=False)
def load_reseller_complete_data(_client):
    """
    Load SEMUA data reseller: forecast, sales, past rofo, past PO
    """
    gsheet_url = st.secrets["gsheet_url"]
    reseller_data = {}
    
    try:
        spreadsheet = _client.open_by_url(gsheet_url)

        # 1. FORECAST 2026 RESELLER
        ws_fcst = spreadsheet.worksheet("Forecast_2026_Reseller")
        df_fcst_raw = pd.DataFrame(ws_fcst.get_all_records())
        df_fcst_raw.columns = [col.strip() for col in df_fcst_raw.columns]
        
        # Identifikasi kolom bulan
        all_month_cols = [c for c in df_fcst_raw.columns if any(m in c.upper() for m in 
                      ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
        
        # Pisahkan 2025 (history) vs 2026+ (forecast)
        hist_cols = []
        fcst_cols = []
        
        for col in all_month_cols:
            col_str = str(col).upper()
            if '25' in col_str or '2025' in col_str:
                hist_cols.append(col)
            else:
                fcst_cols.append(col)  # 2026, 2027, dll
        
        # Convert numeric
        for col in all_month_cols:
            df_fcst_raw[col] = pd.to_numeric(df_fcst_raw[col], errors='coerce').fillna(0)
        
        reseller_data['forecast'] = df_fcst_raw
        reseller_data['forecast_month_cols'] = fcst_cols
        reseller_data['historical_month_cols'] = hist_cols
        
        # 2. SALES RESELLER
        try:
            ws_sales = spreadsheet.worksheet("Sales_Reseller")
            df_sales_raw = pd.DataFrame(ws_sales.get_all_records())
            df_sales_raw.columns = [col.strip() for col in df_sales_raw.columns]
            
            # Transform ke long format
            month_cols_sales = [c for c in df_sales_raw.columns if any(m in c.upper() for m in 
                          ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            
            if month_cols_sales and 'SKU_ID' in df_sales_raw.columns:
                id_cols_sales = ['SKU_ID', 'Brand', 'Product_Name', 'SKU_Tier', 'Floor_Price']
                id_cols_sales = [c for c in id_cols_sales if c in df_sales_raw.columns]
                
                df_sales_long = df_sales_raw.melt(
                    id_vars=id_cols_sales,
                    value_vars=month_cols_sales,
                    var_name='Month_Label',
                    value_name='Sales_Qty'
                )
                df_sales_long['Sales_Qty'] = pd.to_numeric(df_sales_long['Sales_Qty'], errors='coerce').fillna(0)
                df_sales_long['Month'] = df_sales_long['Month_Label'].apply(validate_month_format)
                reseller_data['sales'] = df_sales_long
        except Exception as e:
            st.warning(f"⚠️ Sales_Reseller sheet not accessible: {str(e)}")
        
        # 3. PAST ROFO RESELLER
        try:
            ws_rofo = spreadsheet.worksheet("Past_Rofo_Reseller")
            df_rofo_raw = pd.DataFrame(ws_rofo.get_all_records())
            df_rofo_raw.columns = [col.strip() for col in df_rofo_raw.columns]
            
            month_cols_rofo = [c for c in df_rofo_raw.columns if any(m in c.upper() for m in 
                          ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            
            if month_cols_rofo and 'SKU_ID' in df_rofo_raw.columns:
                id_cols_rofo = ['SKU_ID', 'Brand', 'Product_Name', 'SKU_Tier', 'Floor_Price']
                id_cols_rofo = [c for c in id_cols_rofo if c in df_rofo_raw.columns]
                
                df_rofo_long = df_rofo_raw.melt(
                    id_vars=id_cols_rofo,
                    value_vars=month_cols_rofo,
                    var_name='Month_Label',
                    value_name='Forecast_Qty'
                )
                df_rofo_long['Forecast_Qty'] = pd.to_numeric(df_rofo_long['Forecast_Qty'], errors='coerce').fillna(0)
                df_rofo_long['Month'] = df_rofo_long['Month_Label'].apply(validate_month_format)
                reseller_data['past_rofo'] = df_rofo_long
        except Exception as e:
            st.warning(f"⚠️ Past_Rofo_Reseller sheet not accessible: {str(e)}")
        
        # 4. PAST PO RESELLER
        try:
            ws_po = spreadsheet.worksheet("Past_PO_Reseller")
            df_po_raw = pd.DataFrame(ws_po.get_all_records())
            df_po_raw.columns = [col.strip() for col in df_po_raw.columns]
            
            month_cols_po = [c for c in df_po_raw.columns if any(m in c.upper() for m in 
                          ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'])]
            
            if month_cols_po and 'SKU_ID' in df_po_raw.columns:
                id_cols_po = ['SKU_ID', 'Brand', 'Product_Name', 'SKU_Tier', 'Floor_Price']
                id_cols_po = [c for c in id_cols_po if c in df_po_raw.columns]
                
                df_po_long = df_po_raw.melt(
                    id_vars=id_cols_po,
                    value_vars=month_cols_po,
                    var_name='Month_Label',
                    value_name='PO_Qty'
                )
                df_po_long['PO_Qty'] = pd.to_numeric(df_po_long['PO_Qty'], errors='coerce').fillna(0)
                df_po_long['Month'] = df_po_long['Month_Label'].apply(validate_month_format)
                reseller_data['past_po'] = df_po_long
        except Exception as e:
            st.warning(f"⚠️ Past_PO_Reseller sheet not accessible: {str(e)}")
        
        return reseller_data
        
    except Exception as e:
        st.error(f"❌ Error loading reseller data: {str(e)}")
        return {}

# --- ====================================================== ---
# ---                FINANCIAL FUNCTIONS                    ---
# --- ====================================================== ---

@st.cache_data(ttl=600)
def calculate_financial_metrics_all(df_sales, df_product):
    """Calculate all financial metrics from sales data"""
    
    if df_sales.empty or df_product.empty:
        return pd.DataFrame()
    
    try:
        # Check if price columns exist
        required_price_cols = ['Floor_Price', 'Net_Order_Price']
        price_cols_exist = all(col in df_product.columns for col in required_price_cols)
        
        if not price_cols_exist:
            st.warning("⚠️ Price columns missing in Product Master")
            return pd.DataFrame()
        
        # Ensure sales data has product info with prices
        if 'Floor_Price' not in df_sales.columns or 'Net_Order_Price' not in df_sales.columns:
            df_sales = add_product_info_to_data(df_sales, df_product)
        
        # Fill missing prices
        df_sales['Floor_Price'] = df_sales['Floor_Price'].fillna(0)
        df_sales['Net_Order_Price'] = df_sales['Net_Order_Price'].fillna(0)
        
        # Calculate financial metrics
        df_sales['Revenue'] = df_sales['Sales_Qty'] * df_sales['Floor_Price']
        df_sales['Cost'] = df_sales['Sales_Qty'] * df_sales['Net_Order_Price']
        df_sales['Gross_Margin'] = df_sales['Revenue'] - df_sales['Cost']
        df_sales['Margin_Percentage'] = np.where(
            df_sales['Revenue'] > 0,
            (df_sales['Gross_Margin'] / df_sales['Revenue'] * 100),
            0
        )
        
        # Add additional metrics
        df_sales['Avg_Selling_Price'] = np.where(
            df_sales['Sales_Qty'] > 0,
            df_sales['Revenue'] / df_sales['Sales_Qty'],
            0
        )
        
        return df_sales
        
    except Exception as e:
        st.error(f"Financial metrics calculation error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def calculate_inventory_financial(df_stock, df_product):
    """Calculate inventory financial value"""
    
    if df_stock.empty or df_product.empty:
        return pd.DataFrame()
    
    try:
        # Check price columns
        if 'Floor_Price' not in df_product.columns or 'Net_Order_Price' not in df_product.columns:
            return pd.DataFrame()
        
        # Ensure stock data has prices
        if 'Floor_Price' not in df_stock.columns or 'Net_Order_Price' not in df_stock.columns:
            df_stock = add_product_info_to_data(df_stock, df_product)
        
        # Fill missing prices
        df_stock['Floor_Price'] = df_stock['Floor_Price'].fillna(0)
        df_stock['Net_Order_Price'] = df_stock['Net_Order_Price'].fillna(0)
        
        # Calculate inventory values
        df_stock['Value_at_Cost'] = df_stock['Stock_Qty'] * df_stock['Net_Order_Price']
        df_stock['Value_at_Retail'] = df_stock['Stock_Qty'] * df_stock['Floor_Price']
        df_stock['Potential_Margin'] = df_stock['Value_at_Retail'] - df_stock['Value_at_Cost']
        df_stock['Margin_Percentage'] = np.where(
            df_stock['Value_at_Retail'] > 0,
            (df_stock['Potential_Margin'] / df_stock['Value_at_Retail'] * 100),
            0
        )
        
        return df_stock
        
    except Exception as e:
        st.error(f"Inventory financial calculation error: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def calculate_seasonality(df_financial):
    """Calculate seasonal patterns from financial data"""
    
    if df_financial.empty:
        return pd.DataFrame()
    
    try:
        # Add month and year columns
        df_financial['Year'] = df_financial['Month'].dt.year
        df_financial['Month_Num'] = df_financial['Month'].dt.month
        df_financial['Month_Name'] = df_financial['Month'].dt.strftime('%b')
        
        # Group by month across years
        seasonal_pattern = df_financial.groupby(['Month_Num', 'Month_Name']).agg({
            'Revenue': 'mean',
            'Gross_Margin': 'mean',
            'Sales_Qty': 'mean'
        }).reset_index()
        
        # Calculate seasonal indices
        overall_avg_revenue = seasonal_pattern['Revenue'].mean()
        seasonal_pattern['Seasonal_Index_Revenue'] = seasonal_pattern['Revenue'] / overall_avg_revenue
        
        overall_avg_margin = seasonal_pattern['Gross_Margin'].mean()
        seasonal_pattern['Seasonal_Index_Margin'] = seasonal_pattern['Gross_Margin'] / overall_avg_margin
        
        # Classify seasons
        conditions = [
            seasonal_pattern['Seasonal_Index_Revenue'] >= 1.2,
            (seasonal_pattern['Seasonal_Index_Revenue'] >= 0.9) & (seasonal_pattern['Seasonal_Index_Revenue'] < 1.2),
            seasonal_pattern['Seasonal_Index_Revenue'] < 0.9
        ]
        choices = ['Peak Season', 'Normal Season', 'Low Season']
        
        seasonal_pattern['Season_Type'] = np.select(conditions, choices, default='Normal Season')
        
        return seasonal_pattern.sort_values('Month_Num')
        
    except Exception as e:
        st.error(f"Seasonality calculation error: {str(e)}")
        return pd.DataFrame()

def calculate_eoq(demand, order_cost, holding_cost_per_unit):
    """Calculate Economic Order Quantity"""
    if demand <= 0 or order_cost <= 0 or holding_cost_per_unit <= 0:
        return 0
    
    eoq = math.sqrt((2 * demand * order_cost) / holding_cost_per_unit)
    return round(eoq)

def calculate_forecast_bias(df_forecast, df_po):
    """Calculate aggregate volume-weighted forecast bias"""
    
    if df_forecast.empty or df_po.empty:
        return pd.DataFrame()
    
    try:
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        common_months = sorted(set(forecast_months) & set(po_months))
        
        if not common_months:
            return pd.DataFrame()
        
        bias_results = []
        
        for month in common_months:
            # Tetap hilangkan yang Forecast-nya 0 (Unplanned)
            df_f_month = df_forecast[(df_forecast['Month'] == month) & (df_forecast['Forecast_Qty'] > 0)]
            df_p_month = df_po[df_po['Month'] == month]
            
            df_merged = pd.merge(
                df_f_month[['SKU_ID', 'Forecast_Qty']],
                df_p_month[['SKU_ID', 'PO_Qty']],
                on='SKU_ID', how='inner'
            )
            
            if df_merged.empty:
                continue
                
            df_merged['Bias'] = df_merged['PO_Qty'] - df_merged['Forecast_Qty']
            
            # --- PERBAIKAN: AGGREGATE BIAS ---
            total_month_rofo = df_merged['Forecast_Qty'].sum()
            total_month_po = df_merged['PO_Qty'].sum()
            
            if total_month_rofo > 0:
                aggregate_bias_pct = ((total_month_po - total_month_rofo) / total_month_rofo) * 100
            else:
                aggregate_bias_pct = 0
            
            avg_bias = df_merged['Bias'].mean() # Rata-rata satuan unit (masih oke disimpan)
            
            bias_results.append({
                'Month': month,
                'Avg_Bias': avg_bias,
                'Avg_Bias_Percentage': aggregate_bias_pct, # Sekarang bebas dari outlier ekstrim!
                'Over_Forecast_SKUs': len(df_merged[df_merged['Bias'] < 0]), # PO < Rofo
                'Under_Forecast_SKUs': len(df_merged[df_merged['Bias'] > 0]) # PO > Rofo
            })
        
        return pd.DataFrame(bias_results)
        
    except Exception as e:
        st.error(f"Forecast bias calculation error: {str(e)}")
        return pd.DataFrame()

# --- ====================================================== ---
# ---                ANALYTICS FUNCTIONS                    ---
# --- ====================================================== ---

def calculate_monthly_performance(df_forecast, df_po, df_product):
    """Calculate performance using Hit Rate for Accuracy and WMAPE for Error Rate"""
    
    monthly_performance = {}
    
    if df_forecast.empty or df_po.empty:
        return monthly_performance
    
    try:
        # ADD PRODUCT INFO jika belum ada
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        # Get unique months from both datasets
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        all_months = sorted(set(list(forecast_months) + list(po_months)))
        
        for month in all_months:
            # Get data for this month - FILTER HANYA Forecast_Qty > 0 (Abaikan Unplanned)
            df_forecast_month = df_forecast[
                (df_forecast['Month'] == month) & 
                (df_forecast['Forecast_Qty'] > 0)
            ].copy()
            
            df_po_month = df_po[df_po['Month'] == month].copy()
            
            if df_forecast_month.empty or df_po_month.empty:
                continue
            
            # Merge forecast and PO for this month
            df_merged = pd.merge(
                df_forecast_month, df_po_month,
                on=['SKU_ID'], how='inner', suffixes=('_forecast', '_po')
            )
            
            if not df_merged.empty:
                # Add product info (jika belum ada dari merge)
                if 'Product_Name' not in df_merged.columns or 'Brand' not in df_merged.columns:
                    df_merged = add_product_info_to_data(df_merged, df_product)
                
                # 1. PERHITUNGAN UNTUK HIT RATE (SKU LEVEL)
                df_merged['PO_Rofo_Ratio'] = (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100
                
                # Categorize Status
                conditions = [
                    df_merged['PO_Rofo_Ratio'] < 80,
                    (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
                    df_merged['PO_Rofo_Ratio'] > 120
                ]
                choices = ['Under', 'Accurate', 'Over']
                df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
                
                status_counts = df_merged['Accuracy_Status'].value_counts().to_dict()
                total_records = len(df_merged)
                status_percentages = {k: (v/total_records*100) for k, v in status_counts.items()}
                
                # >>> ACCURACY = HIT RATE (SKU yang Accurate / Total SKU) <<<
                accurate_count = status_counts.get('Accurate', 0)
                monthly_accuracy = (accurate_count / total_records * 100) if total_records > 0 else 0
                
                # 2. PERHITUNGAN UNTUK MAPE (MENGGUNAKAN WMAPE AGAR AMAN)
                total_rofo = df_merged['Forecast_Qty'].sum()
                total_abs_error = abs(df_merged['PO_Qty'] - df_merged['Forecast_Qty']).sum()
                
                # >>> WMAPE = Total Selisih Absolut / Total Rofo <<<
                if total_rofo > 0:
                    wmape = (total_abs_error / total_rofo) * 100
                else:
                    wmape = 0
                
                # Store results
                monthly_performance[month] = {
                    'accuracy': monthly_accuracy, # Hit Rate (Sesuai strategi Bapak)
                    'mape': wmape,                # WMAPE (Aman dari outlier)
                    'status_counts': status_counts,
                    'status_percentages': status_percentages,
                    'total_records': total_records,
                    'data': df_merged,
                    'under_skus': df_merged[df_merged['Accuracy_Status'] == 'Under'].copy(),
                    'over_skus': df_merged[df_merged['Accuracy_Status'] == 'Over'].copy(),
                    'accurate_skus': df_merged[df_merged['Accuracy_Status'] == 'Accurate'].copy()
                }
        
        return monthly_performance
        
    except Exception as e:
        st.error(f"Monthly performance calculation error: {str(e)}")
        return monthly_performance

def get_last_3_months_performance(monthly_performance):
    """Get performance for last 3 months"""
    
    if not monthly_performance:
        return {}
    
    # Get last 3 months
    sorted_months = sorted(monthly_performance.keys())
    if len(sorted_months) >= 3:
        last_3_months = sorted_months[-3:]
    else:
        last_3_months = sorted_months
    
    last_3_data = {}
    for month in last_3_months:
        last_3_data[month] = monthly_performance[month]
    
    return last_3_data

@st.cache_data(ttl=600)
def calculate_inventory_metrics_with_3month_avg(df_stock, df_sales, df_product):
    """Calculate inventory metrics using 3-month average sales (FIXED: AGGREGATE STOCK FIRST)"""
    
    metrics = {}
    
    if df_stock.empty:
        return metrics
    
    try:
        # --- FIX UTAMA: Agregasi Stok dari Level Batch ke Level SKU ---
        # Kita jumlahkan dulu Stock_Qty berdasarkan SKU_ID agar 1 SKU = 1 Baris
        df_stock_agg = df_stock.groupby('SKU_ID').agg({
            'Stock_Qty': 'sum'
        }).reset_index()
        
        # ADD PRODUCT INFO ke data yang sudah di-agregasi
        df_stock_agg = add_product_info_to_data(df_stock_agg, df_product)
        
        # Siapkan Sales Data
        df_sales = add_product_info_to_data(df_sales, df_product)
        
        # Get last 3 months sales data
        if not df_sales.empty:
            sales_months = sorted(df_sales['Month'].unique())
            if len(sales_months) >= 3:
                last_3_sales_months = sales_months[-3:]
                df_sales_last_3 = df_sales[df_sales['Month'].isin(last_3_sales_months)].copy()
            else:
                df_sales_last_3 = df_sales.copy()
        
        # Calculate average monthly sales per SKU
        if not df_sales.empty and not df_sales_last_3.empty:
            avg_monthly_sales = df_sales_last_3.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            avg_monthly_sales.columns = ['SKU_ID', 'Avg_Monthly_Sales_3M']
        else:
            avg_monthly_sales = pd.DataFrame(columns=['SKU_ID', 'Avg_Monthly_Sales_3M'])
        
        # Merge Stock Aggregated dengan Product Info (redundant check but safe)
        df_inventory = pd.merge(
            df_stock_agg,
            df_product[['SKU_ID', 'Product_Name', 'SKU_Tier', 'Brand', 'Status']],
            on='SKU_ID',
            how='left',
            suffixes=('', '_master')
        )
        
        # Bersihkan kolom duplikat jika ada setelah merge
        df_inventory = df_inventory.loc[:,~df_inventory.columns.duplicated()]
        
        # Merge dengan Average Sales
        df_inventory = pd.merge(df_inventory, avg_monthly_sales, on='SKU_ID', how='left')
        df_inventory['Avg_Monthly_Sales_3M'] = df_inventory['Avg_Monthly_Sales_3M'].fillna(0)
        
        # Calculate cover months
        df_inventory['Cover_Months'] = np.where(
            df_inventory['Avg_Monthly_Sales_3M'] > 0,
            df_inventory['Stock_Qty'] / df_inventory['Avg_Monthly_Sales_3M'],
            999  # For SKUs with no sales
        )
        
        # Categorize inventory status
        conditions = [
            df_inventory['Cover_Months'] < 0.8,
            (df_inventory['Cover_Months'] >= 0.8) & (df_inventory['Cover_Months'] <= 2),
            df_inventory['Cover_Months'] > 1.5
        ]
        choices = ['Need Replenishment', 'Ideal/Healthy', 'High Stock']
        df_inventory['Inventory_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Get high/low stock items
        high_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'High Stock'].copy().sort_values('Cover_Months', ascending=False)
        low_stock_df = df_inventory[df_inventory['Inventory_Status'] == 'Need Replenishment'].copy().sort_values('Cover_Months', ascending=True)
        
        # Tier analysis
        if 'SKU_Tier' in df_inventory.columns:
            tier_analysis = df_inventory.groupby('SKU_Tier').agg({
                'SKU_ID': 'count',
                'Stock_Qty': 'sum',
                'Avg_Monthly_Sales_3M': 'sum',
                'Cover_Months': 'mean'
            }).reset_index()
            tier_analysis.columns = ['Tier', 'SKU_Count', 'Total_Stock', 'Total_Sales_3M_Avg', 'Avg_Cover_Months']
            tier_analysis['Turnover'] = tier_analysis['Total_Sales_3M_Avg'] / tier_analysis['Total_Stock']
            metrics['tier_analysis'] = tier_analysis
        
        metrics['inventory_df'] = df_inventory
        metrics['high_stock'] = high_stock_df
        metrics['low_stock'] = low_stock_df
        metrics['total_stock'] = df_inventory['Stock_Qty'].sum()
        metrics['total_skus'] = len(df_inventory)
        metrics['avg_cover'] = df_inventory[df_inventory['Cover_Months'] < 999]['Cover_Months'].mean()
        
        metrics['inventory_value_score'] = (len(df_inventory[df_inventory['Inventory_Status'] == 'Ideal/Healthy']) / 
                                            len(df_inventory) * 100) if len(df_inventory) > 0 else 0
        
        return metrics
        
    except Exception as e:
        st.error(f"Inventory metrics error: {str(e)}")
        return metrics

def calculate_sales_vs_forecast_po(df_sales, df_forecast, df_po, df_product):
    """Calculate sales vs forecast and PO comparison - HANYA ACTIVE SKUS"""
    
    results = {}
    
    if df_sales.empty or df_forecast.empty:
        return results
    
    try:
        # ADD PRODUCT INFO jika belum ada
        df_sales = add_product_info_to_data(df_sales, df_product)
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        # FILTER HANYA ACTIVE SKUS
        if 'Status' in df_product.columns:
            active_skus = df_product[df_product['Status'].str.upper() == 'ACTIVE']['SKU_ID'].tolist()
            
            # Filter semua dataset untuk hanya active SKUs
            df_sales = df_sales[df_sales['SKU_ID'].isin(active_skus)]
            df_forecast = df_forecast[df_forecast['SKU_ID'].isin(active_skus)]
            if not df_po.empty:
                df_po = df_po[df_po['SKU_ID'].isin(active_skus)]
        
        # Get last 3 months for comparison
        sales_months = sorted(df_sales['Month'].unique())
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        
        # Find common months
        common_months = sorted(set(sales_months) & set(forecast_months) & set(po_months))
        
        if not common_months:
            return results
        
        # Use last common month
        last_month = common_months[-1]
        
        # Get data for last month
        df_sales_month = df_sales[df_sales['Month'] == last_month].copy()
        df_forecast_month = df_forecast[df_forecast['Month'] == last_month].copy()
        df_po_month = df_po[df_po['Month'] == last_month].copy()
        
        # Filter hanya SKU dengan Forecast_Qty > 0
        df_forecast_month = df_forecast_month[df_forecast_month['Forecast_Qty'] > 0]
        
        # Merge all data
        df_merged = pd.merge(
            df_sales_month[['SKU_ID', 'Sales_Qty']],
            df_forecast_month[['SKU_ID', 'Forecast_Qty']],
            on='SKU_ID',
            how='inner'
        )
        
        df_merged = pd.merge(
            df_merged,
            df_po_month[['SKU_ID', 'PO_Qty']],
            on='SKU_ID',
            how='left'
        )
        
        # Add product info
        df_merged = add_product_info_to_data(df_merged, df_product)
        
        # Filter out SKU dengan PO_Qty = 0 (tidak ada PO) jika mau
        # df_merged = df_merged[df_merged['PO_Qty'] > 0]
        
        # Calculate ratios
        df_merged['Sales_vs_Forecast_Ratio'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['Sales_Qty'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        df_merged['Sales_vs_PO_Ratio'] = np.where(
            df_merged['PO_Qty'] > 0,
            (df_merged['Sales_Qty'] / df_merged['PO_Qty']) * 100,
            0
        )
        
        # Calculate deviations
        df_merged['Forecast_Deviation'] = abs(df_merged['Sales_vs_Forecast_Ratio'] - 100)
        df_merged['PO_Deviation'] = abs(df_merged['Sales_vs_PO_Ratio'] - 100)
        
        # Identify SKUs with high deviation (> 30%) - HANYA ACTIVE SKUS
        high_deviation_skus = df_merged[
            (df_merged['Forecast_Deviation'] > 30) | 
            (df_merged['PO_Deviation'] > 30)
        ].copy()
        
        high_deviation_skus = high_deviation_skus.sort_values('Forecast_Deviation', ascending=False)
        
        # Calculate overall metrics
        avg_forecast_deviation = df_merged['Forecast_Deviation'].mean()
        avg_po_deviation = df_merged['PO_Deviation'].mean()
        
        results = {
            'last_month': last_month,
            'comparison_data': df_merged,
            'high_deviation_skus': high_deviation_skus,
            'avg_forecast_deviation': avg_forecast_deviation,
            'avg_po_deviation': avg_po_deviation,
            'total_skus_compared': len(df_merged),
            'active_skus_only': True
        }
        
        return results
        
    except Exception as e:
        st.error(f"Sales vs forecast calculation error: {str(e)}")
        return results

def calculate_brand_performance(df_forecast, df_po, df_product):
    """Calculate forecast accuracy performance by brand"""
    
    if df_forecast.empty or df_po.empty or df_product.empty:
        return pd.DataFrame()
    
    try:
        # ADD PRODUCT INFO jika belum ada
        df_forecast = add_product_info_to_data(df_forecast, df_product)
        df_po = add_product_info_to_data(df_po, df_product)
        
        # Get last month data
        forecast_months = sorted(df_forecast['Month'].unique())
        po_months = sorted(df_po['Month'].unique())
        common_months = sorted(set(forecast_months) & set(po_months))
        
        if not common_months:
            return pd.DataFrame()
        
        last_month = common_months[-1]
        
        # Get data for last month
        df_forecast_month = df_forecast[df_forecast['Month'] == last_month].copy()
        df_po_month = df_po[df_po['Month'] == last_month].copy()
        
        # Merge forecast and PO
        df_merged = pd.merge(
            df_forecast_month,
            df_po_month,
            on=['SKU_ID'],
            how='inner'
        )
        
        # Add brand info jika belum ada
        if 'Brand' not in df_merged.columns:
            df_merged = add_product_info_to_data(df_merged, df_product)
        
        if 'Brand' not in df_merged.columns:
            return pd.DataFrame()
        
        # Calculate ratio and accuracy
        df_merged['PO_Rofo_Ratio'] = np.where(
            df_merged['Forecast_Qty'] > 0,
            (df_merged['PO_Qty'] / df_merged['Forecast_Qty']) * 100,
            0
        )
        
        # Categorize
        conditions = [
            df_merged['PO_Rofo_Ratio'] < 80,
            (df_merged['PO_Rofo_Ratio'] >= 80) & (df_merged['PO_Rofo_Ratio'] <= 120),
            df_merged['PO_Rofo_Ratio'] > 120
        ]
        choices = ['Under', 'Accurate', 'Over']
        df_merged['Accuracy_Status'] = np.select(conditions, choices, default='Unknown')
        
        # Calculate brand performance
        brand_performance = df_merged.groupby('Brand').agg({
            'SKU_ID': 'count',
            'Forecast_Qty': 'sum',
            'PO_Qty': 'sum',
            'PO_Rofo_Ratio': lambda x: 100 - abs(x - 100).mean()  # Accuracy
        }).reset_index()
        
        brand_performance.columns = ['Brand', 'SKU_Count', 'Total_Forecast', 'Total_PO', 'Accuracy']
        
        # Calculate additional metrics
        brand_performance['PO_vs_Forecast_Ratio'] = (brand_performance['Total_PO'] / brand_performance['Total_Forecast'] * 100)
        brand_performance['Qty_Difference'] = brand_performance['Total_PO'] - brand_performance['Total_Forecast']
        
        # Get status counts
        status_counts = df_merged.groupby(['Brand', 'Accuracy_Status']).size().unstack(fill_value=0).reset_index()
        
        # Merge with performance data
        brand_performance = pd.merge(brand_performance, status_counts, on='Brand', how='left')
        
        # Fill NaN with 0 for status columns
        for status in ['Under', 'Accurate', 'Over']:
            if status not in brand_performance.columns:
                brand_performance[status] = 0
        
        # Sort by accuracy
        brand_performance = brand_performance.sort_values('Accuracy', ascending=False)
        
        return brand_performance
        
    except Exception as e:
        st.error(f"Brand performance calculation error: {str(e)}")
        return pd.DataFrame()

def identify_profitability_segments(df_financial):
    """Segment SKUs by profitability"""
    
    if df_financial.empty:
        return pd.DataFrame()
    
    try:
        sku_profitability = df_financial.groupby(['SKU_ID', 'Product_Name', 'Brand']).agg({
            'Revenue': 'sum',
            'Gross_Margin': 'sum',
            'Sales_Qty': 'sum'
        }).reset_index()
        
        # Calculate metrics
        sku_profitability['Avg_Margin_Per_SKU'] = sku_profitability['Gross_Margin'] / sku_profitability['Sales_Qty']
        sku_profitability['Margin_Percentage'] = np.where(
            sku_profitability['Revenue'] > 0,
            (sku_profitability['Gross_Margin'] / sku_profitability['Revenue'] * 100),
            0
        )
        
        # Segment by margin percentage
        conditions = [
            (sku_profitability['Margin_Percentage'] >= 40),
            (sku_profitability['Margin_Percentage'] >= 20) & (sku_profitability['Margin_Percentage'] < 40),
            (sku_profitability['Margin_Percentage'] < 20) & (sku_profitability['Margin_Percentage'] > 0),
            (sku_profitability['Margin_Percentage'] <= 0)
        ]
        choices = ['High Margin (>40%)', 'Medium Margin (20-40%)', 'Low Margin (<20%)', 'Negative Margin']
        
        sku_profitability['Margin_Segment'] = np.select(conditions, choices, default='Unknown')
        
        return sku_profitability.sort_values('Gross_Margin', ascending=False)
        
    except Exception as e:
        st.error(f"Profitability segmentation error: {str(e)}")
        return pd.DataFrame()

def validate_data_quality(df, df_name):
    """Comprehensive data quality validation"""
    
    checks = {}
    
    if df.empty:
        checks['Empty Dataset'] = '❌ Dataset kosong'
        return checks
    
    # Basic checks
    checks['Total Rows'] = f"📊 {len(df):,} rows"
    checks['Total Columns'] = f"📋 {len(df.columns)} columns"
    
    # Missing values
    missing_values = df.isnull().sum().sum()
    missing_pct = (missing_values / (len(df) * len(df.columns)) * 100)
    checks['Missing Values'] = f"⚠️ {missing_values:,} ({missing_pct:.1f}%)" if missing_values > 0 else f"✅ {missing_values:,}"
    
    # Duplicates
    duplicates = df.duplicated().sum()
    checks['Duplicate Rows'] = f"⚠️ {duplicates:,}" if duplicates > 0 else f"✅ {duplicates:,}"
    
    # Zero values (for numeric columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        zero_values = (df[numeric_cols] == 0).sum().sum()
        zero_pct = (zero_values / (len(df) * len(numeric_cols)) * 100)
        checks['Zero Values'] = f"📉 {zero_values:,} ({zero_pct:.1f}%)"
    
    # Negative values
    if len(numeric_cols) > 0:
        negative_values = (df[numeric_cols] < 0).sum().sum()
        if negative_values > 0:
            checks['Negative Values'] = f"❌ {negative_values:,}"
    
    # Date range (if Month column exists)
    if 'Month' in df.columns:
        try:
            min_date = df['Month'].min()
            max_date = df['Month'].max()
            checks['Date Range'] = f"📅 {min_date.strftime('%b %Y')} - {max_date.strftime('%b %Y')}"
        except:
            pass
    
    return checks

# --- ====================================================== ---
# ---                DASHBOARD INITIALIZATION               ---
# --- ====================================================== ---

# --- DASHBOARD INITIALIZATION ---
# Initialize connection
client = init_gsheet_connection()

if client is None:
    st.error("❌ Tidak dapat terhubung ke Google Sheets")
    st.stop()

# Load and process data
with st.spinner('🔄 Loading and processing data from Google Sheets...'):
    all_data = load_and_process_data(client)
    
    df_product = all_data.get('product', pd.DataFrame())
    df_product_active = all_data.get('product_active', pd.DataFrame())
    df_sales = all_data.get('sales', pd.DataFrame())
    df_forecast = all_data.get('forecast', pd.DataFrame())
    df_po = all_data.get('po', pd.DataFrame())
    df_stock = all_data.get('stock', pd.DataFrame())
    df_expansi_fulfillment = all_data.get('expansi_fulfillment', pd.DataFrame())
    
    # Ganti rofo_onwards dengan ecomm_forecast (untuk Tab 7)
    df_ecomm_forecast = all_data.get('ecomm_forecast', pd.DataFrame())
    ecomm_forecast_month_cols = all_data.get('ecomm_forecast_month_cols', [])
    
    # Tambah data reseller (untuk Tab 9) - DARI all_data LAMA
    df_reseller_forecast = all_data.get('reseller_forecast', pd.DataFrame())
    reseller_all_month_cols = all_data.get('reseller_all_month_cols', [])
    reseller_historical_cols = all_data.get('reseller_historical_cols', [])
    reseller_forecast_cols = all_data.get('reseller_forecast_cols', [])
    
    # Untuk backward compatibility (jika ada script yang masih pakai nama lama)
    df_rofo_onwards = df_ecomm_forecast  # Alias untuk Tab 7
    rofo_onwards_month_cols = ecomm_forecast_month_cols  # Alias untuk Tab 7
    
    # --- LOAD DATA RESELLER LENGKAP (BARU) ---
    with st.spinner('🔄 Loading Reseller Data...'):
        reseller_complete_data = load_reseller_complete_data(client)
        
        # Data Reseller yang sudah ada (tetap pakai untuk kompatibilitas)
        if df_reseller_forecast.empty and 'forecast' in reseller_complete_data:
            df_reseller_forecast = reseller_complete_data.get('forecast', pd.DataFrame())
        
        if not reseller_forecast_cols and 'forecast_month_cols' in reseller_complete_data:
            reseller_forecast_cols = reseller_complete_data.get('forecast_month_cols', [])
        
        # Data Reseller BARU
        df_sales_reseller = reseller_complete_data.get('sales', pd.DataFrame())
        df_past_rofo_reseller = reseller_complete_data.get('past_rofo', pd.DataFrame())
        df_past_po_reseller = reseller_complete_data.get('past_po', pd.DataFrame())

# Calculate metrics
monthly_performance = calculate_monthly_performance(df_forecast, df_po, df_product)
last_3_months_performance = get_last_3_months_performance(monthly_performance)
inventory_metrics = calculate_inventory_metrics_with_3month_avg(df_stock, df_sales, df_product)
sales_vs_forecast = calculate_sales_vs_forecast_po(df_sales, df_forecast, df_po, df_product)

# Calculate financial metrics
df_financial = calculate_financial_metrics_all(df_sales, df_product)
df_inventory_financial = calculate_inventory_financial(df_stock, df_product)
seasonal_pattern = calculate_seasonality(df_financial) if not df_financial.empty else pd.DataFrame()
forecast_bias = calculate_forecast_bias(df_forecast, df_po)
profitability_segments = identify_profitability_segments(df_financial) if not df_financial.empty else pd.DataFrame()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    with col_sb2:
        if st.button("📊 Show Data Stats", use_container_width=True):
            st.session_state.show_stats = True
            
    # --- TAMBAHAN: TOMBOL CETAK PDF ---
    st.markdown("---")
    import streamlit.components.v1 as components
    
    if st.button("🖨️ Save as PDF", use_container_width=True):
        # Script JavaScript untuk memicu dialog print browser
        components.html(
            """
            <script>
            window.print();
            </script>
            """,
            height=0,
            width=0
        )
    st.caption("Tip: Pilih Destination **'Save as PDF'** & centang **'Background graphics'** di settings print.")
    # ----------------------------------

    st.markdown("---")
    st.markdown("### 📈 Data Overview")
    
    
    if not df_product_active.empty:
        st.metric("Active SKUs", len(df_product_active))
    
    if not df_stock.empty:
        total_stock = df_stock['Stock_Qty'].sum()
        st.metric("Total Stock", f"{total_stock:,.0f}")
    
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        accuracy = monthly_performance[last_month]['accuracy']
        st.metric("Latest Accuracy", f"{accuracy:.1f}%")
    
    # Financial metrics in sidebar
    if not df_financial.empty:
        st.markdown("---")
        st.markdown("### 💰 Financial Overview")
        
        total_revenue = df_financial['Revenue'].sum()
        total_margin = df_financial['Gross_Margin'].sum()
        avg_margin_pct = (total_margin / total_revenue * 100) if total_revenue > 0 else 0
        
        st.metric("Total Revenue", f"Rp {total_revenue:,.0f}")
        st.metric("Total Margin", f"Rp {total_margin:,.0f}")
        st.metric("Avg Margin %", f"{avg_margin_pct:.1f}%")
    
    st.markdown("---")
    
    # Threshold Settings
    st.markdown("### ⚙️ Threshold Settings")
    under_threshold = st.slider("Under Forecast Threshold (%)", 0, 100, 80)
    over_threshold = st.slider("Over Forecast Threshold (%)", 100, 200, 120)
    
    st.markdown("---")
    
    # Inventory Thresholds
    st.markdown("### 📦 Inventory Thresholds")
    low_stock_threshold = st.slider("Low Stock (months)", 0.0, 2.0, 0.8, 0.1)
    high_stock_threshold = st.slider("High Stock (months)", 1.0, 6.0, 2.0, 0.1)
    
    # Financial Thresholds
    st.markdown("---")
    st.markdown("### 💰 Financial Thresholds")
    high_margin_threshold = st.slider("High Margin Threshold (%)", 0, 100, 40)
    low_margin_threshold = st.slider("Low Margin Threshold (%)", 0, 100, 20)
    
    # --- 🎨 THEME SELECTOR (ENHANCED EDITION) ---
    st.markdown("---")
    st.markdown("### 🎨 UI Theme Settings")
    
    theme_choice = st.selectbox(
        "Pilih Tema Dashboard:",
        [
            "⚪ Tema Semula (Enhanced Light)", 
            "⬛ Material Dark (Solid & Colored)"
        ],
        index=0
    )

    theme_css = ""
    
    if theme_choice == "⚪ Tema Semula (Enhanced Light)":
        # Perbaikan: Background utama dibuat soft grey, Chart & Tabel diberi kotak putih & bayangan agar tidak menyatu.
        theme_css = """
        <style>
            /* Background utama dibuat sedikit abu-abu agar chart putih bisa terlihat (pop-out) */
            [data-testid="stAppViewContainer"] { background-color: #F4F6F9 !important; color: #333333 !important; }
            [data-testid="stSidebar"] { background-color: #FFFFFF !important; border-right: 1px solid #EAEAEA !important; }
            
            /* Membungkus Chart Plotly dengan kotak putih & shadow */
            .stPlotlyChart {
                background-color: #FFFFFF !important;
                border-radius: 12px !important;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05) !important;
                padding: 10px !important;
            }
            
            /* Membungkus Dataframe dengan kotak putih */
            .stDataFrame {
                background-color: #FFFFFF !important;
                border-radius: 12px !important;
                box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.05) !important;
                padding: 10px !important;
            }
        </style>
        """
        
    elif theme_choice == "⬛ Material Dark (Solid & Colored)":
        # Perbaikan: Tabel diubah dark mode. Card bewarna dibiarkan tetap bewarna (gradient tidak ditimpa).
        theme_css = """
        <style>
            /* Base Material Dark BG */
            [data-testid="stAppViewContainer"] { background-color: #1A2035 !important; color: #FFFFFF !important; }
            [data-testid="stSidebar"] { background-color: #1F283E !important; border-right: none !important; }
            h1, h2, h3, h4, p, label, .stMarkdown, .stText { color: #FFFFFF !important; }
            hr { border-color: rgba(255,255,255,0.1) !important; }
            
            /* === FIX 1: TABEL (DATAFRAME) MENJADI DARK === */
            [data-testid="stDataFrame"] > div, 
            [data-testid="stDataFrame"] table, 
            [data-testid="stDataFrame"] th, 
            [data-testid="stDataFrame"] td {
                background-color: #1F283E !important;
                color: #FFFFFF !important;
                border-color: rgba(255,255,255,0.05) !important;
            }
            /* Warna Header Tabel sedikit lebih gelap */
            [data-testid="stDataFrame"] th { background-color: #171d30 !important; }
            
            /* === FIX 2: BIARKAN GRADIENT CARD TETAP BERWARNA === */
            /* Pastikan font di dalam card berwarna tetap putih agar terbaca */
            .grad-label, .grad-value, .grad-sub, .tm-title, .tm-main-val, .tm-unit, .tm-sub-row, .fin-title, .fin-val, .fin-sub { 
                color: #FFFFFF !important; 
                text-shadow: 1px 1px 2px rgba(0,0,0,0.3) !important; 
            }
            
            /* Custom HTML Cards yang memang tidak punya warna khusus, kita buat Dark Blue solid */
            .p-card, .sku-header, .metric-highlight {
                background-color: #1F283E !important;
                border: none !important;
                border-radius: 8px !important;
                box-shadow: 0 4px 20px 0 rgba(0,0,0,.14), 0 7px 10px -5px rgba(0,0,0,.4) !important;
                color: #FFFFFF !important;
            }
            .p-val, .p-label { color: #FFFFFF !important; text-shadow: none !important;}
            
            /* Native Streamlit Metrics */
            [data-testid="stMetric"] {
                background-color: #1F283E !important;
                border-radius: 8px !important;
                border: none !important;
                box-shadow: 0 4px 20px 0 rgba(0,0,0,.14) !important;
                padding: 15px !important;
            }
            [data-testid="stMetricValue"] { color: #FFFFFF !important; }
            
            /* Tabs Styling */
            .stTabs [data-baseweb="tab"] { background: #1F283E !important; color: #A9AFBB !important; border: none !important; }
            .stTabs [aria-selected="true"] { background: #9C27B0 !important; color: white !important; box-shadow: 0 4px 20px 0 rgba(0,0,0,.14), 0 7px 10px -5px rgba(156,39,176,.4) !important; }
            
            /* Fix Plotly White Backgrounds to Transparent */
            .js-plotly-plot .plotly .bg, .js-plotly-plot .plotly .paper-bg { fill: transparent !important; }
            .js-plotly-plot .plotly text { fill: #A9AFBB !important; }
            .js-plotly-plot .plotly .gridlayer path { stroke: rgba(255,255,255,0.05) !important; }
        </style>
        """

    if theme_css:
        st.markdown(theme_css, unsafe_allow_html=True)

# Data quality check
if 'show_stats' in st.session_state and st.session_state.show_stats:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Data Quality Check")
    
    for df_name, df in [("Product", df_product), ("Sales", df_sales), 
                       ("Forecast", df_forecast), ("PO", df_po), 
                       ("Stock", df_stock), ("Financial", df_financial)]:
        if not df.empty:
            checks = validate_data_quality(df, df_name)
            with st.sidebar.expander(f"{df_name} Data"):
                for check_name, check_result in checks.items():
                    st.write(f"{check_name}: {check_result}")

# --- MAIN DASHBOARD ---
# =================================================================================
# 📈 PREMIUM FORECAST ACCURACY DASHBOARD SECTION (UPDATED)
# =================================================================================
st.subheader("📈 Forecast Accuracy Performance Trends")

if monthly_performance:
    # 1. Prepare Data
    summary_data = []
    for month, data in sorted(monthly_performance.items()):
        summary_data.append({
            'Month': month,
            'Month_Display': month.strftime('%b %Y'),
            'Accuracy': data['accuracy'],
            'Total_SKUs': data['total_records'],
            'Under': data['status_counts'].get('Under', 0),
            'Over': data['status_counts'].get('Over', 0),
            'Accurate': data['status_counts'].get('Accurate', 0),
            'MAPE': data['mape']
        })
    
    summary_df = pd.DataFrame(summary_data).sort_values('Month')

    if not summary_df.empty:
        # --- A. METRIC CARDS (Gradient Style) ---
        # Calculate Aggregates
        avg_acc = summary_df['Accuracy'].mean()
        last_acc = summary_df['Accuracy'].iloc[-1]
        prev_acc = summary_df['Accuracy'].iloc[-2] if len(summary_df) > 1 else last_acc
        delta_acc = last_acc - prev_acc
        
        best_month = summary_df.loc[summary_df['Accuracy'].idxmax()]
        stability = max(0, 100 - summary_df['Accuracy'].std())

        # CSS khusus untuk Gradient Cards
        st.markdown("""
        <style>
            .grad-card {
                border-radius: 15px;
                padding: 1.5rem;
                color: white;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
                margin-bottom: 1rem;
                position: relative;
                overflow: hidden;
            }
            .grad-card:hover { transform: translateY(-5px); box-shadow: 0 8px 25px rgba(0,0,0,0.15); }
            
            /* Glassmorphism overlay effect */
            .grad-card::before {
                content: "";
                position: absolute;
                top: -50%;
                left: -50%;
                width: 200%;
                height: 200%;
                background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 60%);
                opacity: 0.5;
                pointer-events: none;
            }

            .grad-label { 
                font-size: 0.85rem; 
                font-weight: 600; 
                text-transform: uppercase; 
                letter-spacing: 1px;
                opacity: 0.9;
                margin-bottom: 0.5rem;
                position: relative; 
                z-index: 1;
            }
            .grad-value { 
                font-size: 2.2rem; 
                font-weight: 800; 
                margin-bottom: 0.2rem;
                position: relative; 
                z-index: 1;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .grad-sub { 
                font-size: 0.85rem; 
                font-weight: 500; 
                opacity: 0.9;
                display: flex;
                align-items: center;
                gap: 5px;
                position: relative; 
                z-index: 1;
            }
            .pill {
                background: rgba(255,255,255,0.25);
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 0.75rem;
                backdrop-filter: blur(4px);
            }
        </style>
        """, unsafe_allow_html=True)

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        # Helper Function untuk membuat Gradient Card
        def gradient_card(title, value, sub_text, pill_text, gradient_css):
            return f"""
            <div class="grad-card" style="{gradient_css}">
                <div class="grad-label">{title}</div>
                <div class="grad-value">{value}</div>
                <div class="grad-sub">
                    <span class="pill">{pill_text}</span> {sub_text}
                </div>
            </div>
            """

        with kpi1:
            # Current Accuracy - Blue/Indigo Gradient
            arrow = "▲" if delta_acc >= 0 else "▼"
            grad_css = "background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);"
            st.markdown(gradient_card(
                "Current Accuracy", 
                f"{last_acc:.1f}%", 
                "vs last month", 
                f"{arrow} {abs(delta_acc):.1f}%", 
                grad_css
            ), unsafe_allow_html=True)

        with kpi2:
            # Average YTD - Teal/Cyan Gradient
            avg_diff = avg_acc - 80
            pill_icon = "✅" if avg_diff >= 0 else "⚠️"
            grad_css = "background: linear-gradient(135deg, #0891B2 0%, #22D3EE 100%);"
            st.markdown(gradient_card(
                "Average (YTD)", 
                f"{avg_acc:.1f}%", 
                "vs Target (80%)", 
                f"{pill_icon} {abs(avg_diff):.1f}%", 
                grad_css
            ), unsafe_allow_html=True)

        with kpi3:
            # Best Performance - Emerald/Green Gradient
            grad_css = "background: linear-gradient(135deg, #059669 0%, #10B981 100%);"
            st.markdown(gradient_card(
                "Best Performance", 
                f"{best_month['Accuracy']:.1f}%", 
                "Highest Record", 
                f"📅 {best_month['Month_Display']}", 
                grad_css
            ), unsafe_allow_html=True)

        with kpi4:
            # Stability - Orange/Amber Gradient
            grad_css = "background: linear-gradient(135deg, #EA580C 0%, #F59E0B 100%);"
            st.markdown(gradient_card(
                "Stability Score", 
                f"{stability:.0f}", 
                "Consistency Metric", 
                "📈 0-100", 
                grad_css
            ), unsafe_allow_html=True)

        st.write("") # Spacer

        # --- B. ECHARTS COMBO CHART (Accuracy Trend + SKU Volume) ---
        # Prepare data series
        months_display = summary_df['Month_Display'].tolist()
        accuracy_vals = summary_df['Accuracy'].tolist()
        sku_vals = summary_df['Total_SKUs'].tolist()
        mape_vals = summary_df['MAPE'].tolist()

        # Build accuracy data with per-point color (green/yellow/red)
        acc_series_data = []
        for i, val in enumerate(accuracy_vals):
            if val >= 80:
                color = '#10B981'    # Emerald Green
            elif val >= 70:
                color = '#F59E0B'    # Amber Yellow
            else:
                color = '#EF4444'   # Red
            acc_series_data.append({
                "value": round(val, 1),
                "itemStyle": {
                    "color": color,
                    "borderColor": "#ffffff",
                    "borderWidth": 2
                },
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": f"{val:.1f}%",
                    "fontSize": 11,
                    "fontWeight": "bold",
                    "color": color
                }
            })

        # Build tooltip formatter (show accuracy + mape together)
        tooltip_data = [
            f"Accuracy: {a:.1f}%<br/>MAPE: {m:.1f}%<br/>SKUs: {s}"
            for a, m, s in zip(accuracy_vals, mape_vals, sku_vals)
        ]

        echarts_option = {
            "backgroundColor": "transparent",
            "animation": True,
            "animationDuration": 1200,
            "animationEasing": "cubicOut",
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {
                    "type": "cross",
                    "crossStyle": {"color": "#999"}
                },
                "backgroundColor": "rgba(255,255,255,0.95)",
                "borderColor": "#E5E7EB",
                "borderWidth": 1,
                "textStyle": {"color": "#1F2937", "fontSize": 13}
            },
            "legend": {
                "data": ["Accuracy %", "Total SKUs"],
                "bottom": 0,
                "textStyle": {"fontSize": 12, "color": "#4B5563"}
            },
            "grid": {
                "left": "4%",
                "right": "6%",
                "top": "12%",
                "bottom": "12%",
                "containLabel": True
            },
            "xAxis": {
                "type": "category",
                "data": months_display,
                "axisLabel": {
                    "fontWeight": "bold",
                    "fontSize": 12,
                    "color": "#4B5563"
                },
                "axisLine": {"lineStyle": {"color": "#E5E7EB"}},
                "axisTick": {"show": False}
            },
            "yAxis": [
                {
                    "type": "value",
                    "name": "Accuracy %",
                    "nameTextStyle": {
                        "color": "#6366F1",
                        "fontWeight": "bold",
                        "fontSize": 12
                    },
                    "min": 40,
                    "max": 110,
                    "interval": 10,
                    "axisLabel": {
                        "formatter": "{value}%",
                        "color": "#6366F1",
                        "fontWeight": "bold",
                        "fontSize": 11
                    },
                    "splitLine": {
                        "lineStyle": {"color": "rgba(0,0,0,0.05)", "type": "dashed"}
                    }
                },
                {
                    "type": "value",
                    "name": "SKU Count",
                    "nameTextStyle": {"color": "#9CA3AF", "fontSize": 11},
                    "axisLabel": {"color": "#9CA3AF", "fontSize": 10},
                    "splitLine": {"show": False},
                    "axisLine": {"show": False},
                    "axisTick": {"show": False}
                }
            ],
            "series": [
                {
                    # Background bar — Total SKUs (secondary axis)
                    "name": "Total SKUs",
                    "type": "bar",
                    "yAxisIndex": 1,
                    "barMaxWidth": 35,
                    "data": sku_vals,
                    "itemStyle": {
                        "color": "rgba(99,102,241,0.10)",
                        "borderColor": "rgba(99,102,241,0.25)",
                        "borderWidth": 1,
                        "borderRadius": [4, 4, 0, 0]
                    },
                    "z": 1
                },
                {
                    # Main accuracy line with colored markers
                    "name": "Accuracy %",
                    "type": "line",
                    "yAxisIndex": 0,
                    "data": acc_series_data,
                    "smooth": True,
                    "smoothMonotone": "x",
                    "symbol": "circle",
                    "symbolSize": 14,
                    "lineStyle": {
                        "color": "#6366F1",
                        "width": 3
                    },
                    "areaStyle": {
                        "color": {
                            "type": "linear",
                            "x": 0, "y": 0, "x2": 0, "y2": 1,
                            "colorStops": [
                                {"offset": 0, "color": "rgba(99,102,241,0.15)"},
                                {"offset": 1, "color": "rgba(99,102,241,0.0)"}
                            ]
                        }
                    },
                    "markArea": {
                        "silent": True,
                        "itemStyle": {"color": "rgba(16,185,129,0.06)"},
                        "data": [[{"yAxis": 80}, {"yAxis": 110}]]
                    },
                    "markLine": {
                        "silent": True,
                        "symbol": ["none", "none"],
                        "data": [{"yAxis": 80}],
                        "lineStyle": {
                            "color": "#10B981",
                            "type": "dashed",
                            "width": 1.5
                        },
                        "label": {
                            "show": True,
                            "formatter": "Target 80%",
                            "position": "insideEndTop",
                            "color": "#10B981",
                            "fontWeight": "bold",
                            "fontSize": 11
                        }
                    },
                    "z": 3
                }
            ]
        }

        st_echarts(
            options=echarts_option,
            height="480px",
            key="main_accuracy_trend_chart"
        )

        # --- C. AUTO INSIGHT ---
        # Logic warna insight
        if last_acc >= 80:
            insight_color = "#10B981" # Hijau
            status_text = "EXCELLENT PERFORMANCE 🚀"
        elif last_acc >= 70:
            insight_color = "#F59E0B" # Kuning
            status_text = "MODERATE PERFORMANCE ⚠️"
        else:
            insight_color = "#EF4444" # Merah
            status_text = "CRITICAL ATTENTION NEEDED 🚨"
        
        st.markdown(f"""
        <div style="background-color: white; border-radius: 10px; padding: 1rem; border-left: 6px solid {insight_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.05); color: #4B5563;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <div style="font-size: 1.5rem;">💡</div>
                <div>
                    <div style="font-weight: 800; font-size: 0.8rem; color: {insight_color}; letter-spacing: 1px;">{status_text}</div>
                    <div style="font-size: 0.95rem;">
                        Current accuracy is <strong>{last_acc:.1f}%</strong>. 
                        {'Stable performance.' if abs(delta_acc) < 2 else ('Improved by ' + f'{delta_acc:.1f}%' if delta_acc > 0 else 'Dropped by ' + f'{abs(delta_acc):.1f}%')} from previous month.
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.warning("⚠️ No monthly performance data available to display trend.")

# SECTION 1: LAST 3 MONTHS PERFORMANCE (DIPERBESAR)
st.subheader("🎯 Forecast Performance - 3 Bulan Terakhir")

if last_3_months_performance:
    # Display last 3 months performance
    months_display = []
    
    # Create container untuk 3 bulan
    month_cols = st.columns(3)
    
    for i, (month, data) in enumerate(sorted(last_3_months_performance.items())):
        month_name = month.strftime('%b %Y')
        accuracy = data['accuracy']
        
        with month_cols[i]:
            under_count = data['status_counts'].get('Under', 0)
            accurate_count = data['status_counts'].get('Accurate', 0)
            over_count = data['status_counts'].get('Over', 0)
            total_records = data['total_records']
            
            # Create HTML dengan single line f-string
            html_content = (
                f'<div style="background: white; border-radius: 15px; padding: 1.5rem; margin: 0.5rem 0; box-shadow: 0 6px 20px rgba(0,0,0,0.1); border-top: 5px solid #667eea;">'
                f'<div style="text-align: center; margin-bottom: 1rem;">'
                f'<h3 style="margin: 0; color: #333;">{month_name}</h3>'
                f'<div style="font-size: 2rem; font-weight: 900; color: #667eea;">{accuracy:.1f}%</div>'
                f'<div style="font-size: 0.9rem; color: #666;">Overall Accuracy</div>'
                f'</div>'
                f'<div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 1rem;">'
                f'<div style="text-align: center; padding: 0.5rem; background: #FFEBEE; border-radius: 8px;">'
                f'<div style="font-size: 1.5rem; font-weight: 900; color: #F44336;">{under_count}</div>'
                f'<div style="font-size: 0.8rem; color: #F44336;">Under</div>'
                f'</div>'
                f'<div style="text-align: center; padding: 0.5rem; background: #E8F5E9; border-radius: 8px;">'
                f'<div style="font-size: 1.5rem; font-weight: 900; color: #4CAF50;">{accurate_count}</div>'
                f'<div style="font-size: 0.8rem; color: #4CAF50;">Accurate</div>'
                f'</div>'
                f'<div style="text-align: center; padding: 0.5rem; background: #FFF3E0; border-radius: 8px;">'
                f'<div style="font-size: 1.5rem; font-weight: 900; color: #FF9800;">{over_count}</div>'
                f'<div style="font-size: 0.8rem; color: #FF9800;">Over</div>'
                f'</div>'
                f'</div>'
                f'<div style="text-align: center; font-size: 0.9rem; color: #666;">Total SKUs: {total_records}</div>'
                f'</div>'
            )
            
            st.markdown(html_content, unsafe_allow_html=True)
        
        months_display.append(month_name)
        
    # ==============================================================================
    # 1. TOTAL METRICS - BULAN TERAKHIR (Soft Pastel Gradient Version)
    # ==============================================================================
    st.divider()
    st.subheader("📊 Total Metrics - Rofo Bulan Terakhir")
    
    # Calculate metrics for LAST MONTH ONLY
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        # Count SKUs & Quantities
        under_count = last_month_data[last_month_data['Accuracy_Status'] == 'Under']['SKU_ID'].nunique()
        accurate_count = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['SKU_ID'].nunique()
        over_count = last_month_data[last_month_data['Accuracy_Status'] == 'Over']['SKU_ID'].nunique()
        total_count_last_month = last_month_data['SKU_ID'].nunique()
        
        under_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Under']['Forecast_Qty'].sum()
        accurate_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Accurate']['Forecast_Qty'].sum()
        over_forecast_qty = last_month_data[last_month_data['Accuracy_Status'] == 'Over']['Forecast_Qty'].sum()
        total_forecast_qty = last_month_data['Forecast_Qty'].sum()
        
        # Percentages
        under_pct = (under_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        accurate_pct = (accurate_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        over_pct = (over_count / total_count_last_month * 100) if total_count_last_month > 0 else 0
        
        under_qty_pct = (under_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
        accurate_qty_pct = (accurate_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0
        over_qty_pct = (over_forecast_qty / total_forecast_qty * 100) if total_forecast_qty > 0 else 0

        last_month_accuracy = monthly_performance[last_month]['accuracy']

        # --- CSS STYLE (Fixed Indentation & Soft Colors) ---
        st.markdown("""
<style>
    /* Card Styles */
    .tm-card {
        border-radius: 16px;
        padding: 1.2rem;
        color: white;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    .tm-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); }
    
    .tm-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
    .tm-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.95; }
    .tm-icon { font-size: 1.2rem; opacity: 0.9; background: rgba(255,255,255,0.2); padding: 4px 8px; border-radius: 8px; }
    
    .tm-main-val { font-size: 1.8rem; font-weight: 800; margin-bottom: 0px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
    .tm-unit { font-size: 0.9rem; font-weight: 500; opacity: 0.9; margin-left: 2px; }
    
    .tm-sub-row { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; font-size: 0.8rem; opacity: 0.95; }
    .tm-badge { background: rgba(255,255,255,0.2); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.75rem; }

    /* Process Flow Styles */
    .process-container {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 15px;
        margin-top: 10px;
    }
    .p-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-top: 5px solid #ccc;
        position: relative;
    }
    .p-label { font-size: 0.8rem; font-weight: 700; color: #888; letter-spacing: 1px; margin-bottom: 5px; text-transform: uppercase; }
    .p-val { font-size: 2rem; font-weight: 800; color: #333; margin-bottom: 10px; }
    .p-badge-container { display: flex; gap: 8px; flex-wrap: wrap; }
    .p-badge { 
        font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; 
        display: flex; align-items: center; gap: 4px;
    }
    
    /* Process Flow Colors */
    .border-rofo { border-top-color: #5C6BC0; } /* Soft Indigo */
    .border-po { border-top-color: #FFA726; }   /* Soft Orange */
    .border-sales { border-top-color: #66BB6A; } /* Soft Green */
    
    .text-rofo { color: #3949AB; }
    .text-po { color: #F57C00; }
    .text-sales { color: #2E7D32; }

    .bg-rofo-light { background-color: #E8EAF6; color: #3949AB; }
    .bg-po-light { background-color: #FFF3E0; color: #EF6C00; }
    .bg-sales-light { background-color: #E8F5E9; color: #2E7D32; }
</style>
""", unsafe_allow_html=True)

        # Helper Function untuk render card (TANPA INDENTASI DI HTML)
        def render_soft_card(title, icon, count, qty, qty_pct, bg_gradient):
            # HTML disusun tanpa indentasi agar aman dari bug Markdown
            html = f"""
<div class="tm-card" style="background: {bg_gradient};">
<div class="tm-header">
<span class="tm-title">{title}</span>
<span class="tm-icon">{icon}</span>
</div>
<div>
<span class="tm-main-val">{count}</span><span class="tm-unit">SKUs</span>
</div>
<div class="tm-sub-row">
<span>Qty: {qty:,.0f}</span>
<span class="tm-badge">{qty_pct:.1f}%</span>
</div>
</div>"""
            return html

        # Render Total Metrics
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            # Under - Soft Red
            st.markdown(render_soft_card("Under Forecast", "📉", under_count, under_forecast_qty, under_qty_pct, 
                "linear-gradient(135deg, #ef5350 0%, #e53935 100%)"), unsafe_allow_html=True)
        with c2:
            # Accurate - Soft Green
            st.markdown(render_soft_card("Accurate Forecast", "🎯", accurate_count, accurate_forecast_qty, accurate_qty_pct, 
                "linear-gradient(135deg, #26a69a 0%, #00897b 100%)"), unsafe_allow_html=True)
        with c3:
            # Over - Soft Orange
            st.markdown(render_soft_card("Over Forecast", "📈", over_count, over_forecast_qty, over_qty_pct, 
                "linear-gradient(135deg, #ffa726 0%, #fb8c00 100%)"), unsafe_allow_html=True)
        with c4:
            # Overall - Soft Indigo
            st.markdown(f"""
<div class="tm-card" style="background: linear-gradient(135deg, #5c6bc0 0%, #3949ab 100%);">
<div class="tm-header">
<span class="tm-title">OVERALL SCORE</span>
<span class="tm-icon">🏆</span>
</div>
<div>
<span class="tm-main-val">{last_month_accuracy:.1f}%</span>
</div>
<div class="tm-sub-row">
<span>{last_month.strftime('%B %Y')}</span>
<span class="tm-badge">Total: {total_count_last_month}</span>
</div>
</div>""", unsafe_allow_html=True)

    # ==============================================================================
    # 2. COMPARISON CARDS - PROCESS FLOW STYLE (FIXED RENDERING)
    # ==============================================================================
    if monthly_performance:
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data']
        
        # Calculate Totals
        rofo_tot = last_month_data['Forecast_Qty'].sum()
        po_tot = last_month_data['PO_Qty'].sum()
        
        # Get Sales
        sales_tot = 0
        if not df_sales.empty:
            sales_tot = df_sales[df_sales['Month'] == last_month]['Sales_Qty'].sum()

        # Calculate Ratios
        po_vs_rofo = (po_tot / rofo_tot * 100) if rofo_tot > 0 else 0
        sales_vs_rofo = (sales_tot / rofo_tot * 100) if rofo_tot > 0 else 0
        sales_vs_po = (sales_tot / po_tot * 100) if po_tot > 0 else 0

        st.write("") # Spacer
        st.subheader(f"🔄 Business Flow Performance (Rofo -> PO -> Sales Performance): {last_month.strftime('%B %Y')}")

        # HTML Structure - TANPA INDENTASI SAMA SEKALI
        html_process = f"""
<div class="process-container">
<div class="p-card border-rofo">
<div class="p-label">1. PLANNING (ROFO)</div>
<div class="p-val text-rofo">{rofo_tot:,.0f}</div>
<div class="p-badge-container">
<span class="p-badge bg-rofo-light">🎯 Baseline Target</span>
</div>
</div>
<div class="p-card border-po">
<div class="p-label">2. EXECUTION (PO)</div>
<div class="p-val text-po">{po_tot:,.0f}</div>
<div class="p-badge-container">
<span class="p-badge bg-po-light">{po_vs_rofo:.1f}% vs Rofo</span>
<span class="p-badge" style="background:#f5f5f5; color:#666;">Gap: {po_tot - rofo_tot:+,.0f}</span>
</div>
</div>
<div class="p-card border-sales">
<div class="p-label">3. REALIZATION (SALES)</div>
<div class="p-val text-sales">{sales_tot:,.0f}</div>
<div class="p-badge-container">
<span class="p-badge bg-sales-light">{sales_vs_rofo:.1f}% vs Rofo</span>
<span class="p-badge bg-sales-light">{sales_vs_po:.1f}% vs PO</span>
</div>
</div>
</div>
"""
        st.markdown(html_process, unsafe_allow_html=True)

st.divider()
# SECTION 2: MONTHLY EVALUATION (UNDER & OVER ONLY) DENGAN FILTER BULAN
st.subheader("📋 Evaluasi Rofo per Bulan (Under & Over Forecast)")

if monthly_performance:
    sorted_months = sorted(monthly_performance.keys())
    if sorted_months:
        
        # --- MULAI SCRIPT FILTER BULAN ---
        # 1. Buat list pilihan bulan dalam format string (misal: 'Jan 2025')
        month_options = [m.strftime('%b %Y') for m in sorted_months]
        
        # 2. Buat dropdown selectbox. default index diset ke yang paling akhir (bulan terbaru)
        selected_month_str = st.selectbox(
            "📅 Pilih Bulan Evaluasi:", 
            options=month_options, 
            index=len(month_options) - 1
        )
        
        # 3. Cari kembali key datetime aslinya berdasarkan pilihan user
        selected_month_idx = month_options.index(selected_month_str)
        selected_month_key = sorted_months[selected_month_idx]
        
        # 4. Timpa variabel lama agar script tab HTML di bawahnya tetap berjalan normal tanpa error
        last_month_data = monthly_performance[selected_month_key]
        last_month_name = selected_month_str
        # --- AKHIR SCRIPT FILTER BULAN ---
        
        # Create tabs for Under and Over SKUs (Script di bawah ini tetap sama)
        eval_tab1, eval_tab2 = st.tabs([f"📉 UNDER Forecast ({last_month_name})", f"📈 OVER Forecast ({last_month_name})"])
        
        with eval_tab1:
            under_skus_df = last_month_data['under_skus']
            if not under_skus_df.empty:
                # Add inventory data
                if 'inventory_df' in inventory_metrics:
                    inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
                    under_skus_df = pd.merge(under_skus_df, inventory_data, on='SKU_ID', how='left')
                
                # TAMBAH: Get last 3 months sales data
                sales_cols_last_3 = []
                if not df_sales.empty:
                    # Get last 3 months from sales data
                    sales_months = sorted(df_sales['Month'].unique())
                    if len(sales_months) >= 3:
                        last_3_sales_months = sales_months[-3:]
                        
                        # Create pivot for last 3 months sales
                        try:
                            sales_pivot = df_sales[df_sales['Month'].isin(last_3_sales_months)].pivot_table(
                                index='SKU_ID',
                                columns='Month',
                                values='Sales_Qty',
                                aggfunc='sum',
                                fill_value=0
                            ).reset_index()
                            
                            # Rename columns to month names
                            month_rename = {}
                            for col in sales_pivot.columns:
                                if isinstance(col, datetime):
                                    month_rename[col] = col.strftime('%b-%Y')
                            sales_pivot = sales_pivot.rename(columns=month_rename)
                            
                            # Merge with under_skus_df
                            under_skus_df = pd.merge(
                                under_skus_df,
                                sales_pivot,
                                on='SKU_ID',
                                how='left'
                            )
                            
                            # Get the sales column names
                            sales_cols_last_3 = [col for col in sales_pivot.columns if isinstance(col, str) and '-' in col]
                            sales_cols_last_3 = sorted(sales_cols_last_3[-3:])  # Get last 3 months
                            
                        except Exception as e:
                            st.warning(f"Tidak bisa menambahkan data sales 3 bulan terakhir: {str(e)}")
                
                # Prepare display columns - TAMBAH sales columns
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Accuracy_Status',
                              'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 
                              'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
                
                # Tambah sales columns jika ada
                display_cols.extend(sales_cols_last_3)
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in under_skus_df.columns]
                
                # Pastikan Product_Name selalu ada
                if 'Product_Name' not in available_cols and 'Product_Name' in under_skus_df.columns:
                    available_cols.insert(1, 'Product_Name')
                
                # Format the dataframe
                display_df = under_skus_df[available_cols].copy()
                
                # Add formatted columns
                if 'PO_Rofo_Ratio' in display_df.columns:
                    display_df['PO_Rofo_Ratio'] = display_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
                
                if 'Cover_Months' in display_df.columns:
                    display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
                
                if 'Avg_Monthly_Sales_3M' in display_df.columns:
                    display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
                
                # Format sales columns
                for col in sales_cols_last_3:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                
                # Rename columns for display
                column_names = {
                    'SKU_ID': 'SKU ID',
                    'Product_Name': 'Product Name',
                    'Brand': 'Brand',
                    'SKU_Tier': 'Tier',
                    'Accuracy_Status': 'Status',
                    'Forecast_Qty': 'Forecast Qty',
                    'PO_Qty': 'PO Qty',
                    'PO_Rofo_Ratio': 'PO/Rofo %',
                    'Stock_Qty': 'Stock Available',
                    'Avg_Monthly_Sales_3M': 'Avg Sales (3M)',
                    'Cover_Months': 'Cover (Months)'
                }
                
                # Add sales columns to rename dict
                for col in sales_cols_last_3:
                    column_names[col] = col
                
                display_df = display_df.rename(columns=column_names)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )
                
                # Summary dengan HIGHLIGHT
                total_forecast = under_skus_df['Forecast_Qty'].sum()
                total_po = under_skus_df['PO_Qty'].sum()
                avg_ratio = under_skus_df['PO_Rofo_Ratio'].mean()
                selisih_qty = total_po - total_forecast
                selisih_persen = (selisih_qty / total_forecast * 100) if total_forecast > 0 else 0
                po_rofo_pct = (total_po / total_forecast * 100) if total_forecast > 0 else 0
                
                # Buat HTML content
                html_content = f"""
                <div style="background: #FFEBEE; border-left: 5px solid #F44336; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4 style="color: #C62828; margin-top: 0;">📉 UNDER FORECAST SUMMARY - {last_month_name}</h4>
                    
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 24px; color: #F44336; font-weight: bold; margin-bottom: 5px;">{avg_ratio:.1f}%</div>
                            <div style="font-size: 12px; color: #666;">Avg PO/Rofo</div>
                            <div style="font-size: 10px; color: #999;">Target: 80-120%</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #2E7D32; font-weight: bold; margin-bottom: 5px;">{total_forecast:,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Total Rofo</div>
                            <div style="font-size: 10px; color: #999;">Forecast Qty</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #1565C0; font-weight: bold; margin-bottom: 5px;">{total_po:,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Total PO</div>
                            <div style="font-size: 10px; color: #999;">Purchase Order</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 24px; color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: bold; margin-bottom: 5px;">{selisih_qty:+,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Selisih Qty</div>
                            <div style="font-size: 11px; color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: 600;">({selisih_persen:+.1f}%)</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #FF9800; font-weight: bold; margin-bottom: 5px;">{po_rofo_pct:.1f}%</div>
                            <div style="font-size: 12px; color: #666;">PO/Rofo %</div>
                            <div style="font-size: 10px; color: #999;">Overall Ratio</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(244, 67, 54, 0.3); font-size: 14px; color: #666;">
                        <strong>Total UNDER Forecast SKUs: {len(under_skus_df)}</strong> | 
                        <span style="color: #F44336;">Avg PO/Rofo: {avg_ratio:.1f}%</span> | 
                        <span style="color: #2E7D32;">Rofo: {total_forecast:,.0f}</span> | 
                        <span style="color: #1565C0;">PO: {total_po:,.0f}</span> | 
                        <span style="color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: bold;">Selisih: {selisih_qty:+,.0f} ({selisih_persen:+.1f}%)</span>
                    </div>
                </div>
                """
                
                # Tampilkan dengan st.html()
                st.html(html_content)
            else:
                st.success(f"✅ No SKUs with UNDER forecast in {last_month_name}")
        
        with eval_tab2:
            over_skus_df = last_month_data['over_skus']
            if not over_skus_df.empty:
                # Add inventory data
                if 'inventory_df' in inventory_metrics:
                    inventory_data = inventory_metrics['inventory_df'][['SKU_ID', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']]
                    over_skus_df = pd.merge(over_skus_df, inventory_data, on='SKU_ID', how='left')
                
                # TAMBAH: Get last 3 months sales data
                sales_cols_last_3 = []
                if not df_sales.empty:
                    # Get last 3 months from sales data
                    sales_months = sorted(df_sales['Month'].unique())
                    if len(sales_months) >= 3:
                        last_3_sales_months = sales_months[-3:]
                        
                        # Create pivot for last 3 months sales
                        try:
                            sales_pivot = df_sales[df_sales['Month'].isin(last_3_sales_months)].pivot_table(
                                index='SKU_ID',
                                columns='Month',
                                values='Sales_Qty',
                                aggfunc='sum',
                                fill_value=0
                            ).reset_index()
                            
                            # Rename columns to month names
                            month_rename = {}
                            for col in sales_pivot.columns:
                                if isinstance(col, datetime):
                                    month_rename[col] = col.strftime('%b-%Y')
                            sales_pivot = sales_pivot.rename(columns=month_rename)
                            
                            # Merge with over_skus_df
                            over_skus_df = pd.merge(
                                over_skus_df,
                                sales_pivot,
                                on='SKU_ID',
                                how='left'
                            )
                            
                            # Get the sales column names
                            sales_cols_last_3 = [col for col in sales_pivot.columns if isinstance(col, str) and '-' in col]
                            sales_cols_last_3 = sorted(sales_cols_last_3[-3:])  # Get last 3 months
                            
                        except Exception as e:
                            st.warning(f"Tidak bisa menambahkan data sales 3 bulan terakhir: {str(e)}")
                
                # Prepare display columns - TAMBAH sales columns
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Accuracy_Status',
                              'Forecast_Qty', 'PO_Qty', 'PO_Rofo_Ratio', 
                              'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
                
                # Tambah sales columns jika ada
                display_cols.extend(sales_cols_last_3)
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in over_skus_df.columns]
                
                # Pastikan Product_Name selalu ada
                if 'Product_Name' not in available_cols and 'Product_Name' in over_skus_df.columns:
                    available_cols.insert(1, 'Product_Name')
                
                # Format the dataframe
                display_df = over_skus_df[available_cols].copy()
                
                # Add formatted columns
                if 'PO_Rofo_Ratio' in display_df.columns:
                    display_df['PO_Rofo_Ratio'] = display_df['PO_Rofo_Ratio'].apply(lambda x: f"{x:.1f}%")
                
                if 'Cover_Months' in display_df.columns:
                    display_df['Cover_Months'] = display_df['Cover_Months'].apply(lambda x: f"{x:.1f}" if x < 999 else "N/A")
                
                if 'Avg_Monthly_Sales_3M' in display_df.columns:
                    display_df['Avg_Monthly_Sales_3M'] = display_df['Avg_Monthly_Sales_3M'].apply(lambda x: f"{x:.0f}")
                
                # Format sales columns
                for col in sales_cols_last_3:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.0f}" if pd.notnull(x) else "0")
                
                # Rename columns for display
                column_names = {
                    'SKU_ID': 'SKU ID',
                    'Product_Name': 'Product Name',
                    'Brand': 'Brand',
                    'SKU_Tier': 'Tier',
                    'Accuracy_Status': 'Status',
                    'Forecast_Qty': 'Forecast Qty',
                    'PO_Qty': 'PO Qty',
                    'PO_Rofo_Ratio': 'PO/Rofo %',
                    'Stock_Qty': 'Stock Available',
                    'Avg_Monthly_Sales_3M': 'Avg Sales (3M)',
                    'Cover_Months': 'Cover (Months)'
                }
                
                # Add sales columns to rename dict
                for col in sales_cols_last_3:
                    column_names[col] = col
                
                display_df = display_df.rename(columns=column_names)
                
                st.dataframe(
                    display_df,
                    use_container_width=True,
                    height=500
                )
                
                # Summary dengan HIGHLIGHT
                total_forecast = over_skus_df['Forecast_Qty'].sum()
                total_po = over_skus_df['PO_Qty'].sum()
                avg_ratio = over_skus_df['PO_Rofo_Ratio'].mean()
                selisih_qty = total_po - total_forecast
                selisih_persen = (selisih_qty / total_forecast * 100) if total_forecast > 0 else 0
                po_rofo_pct = (total_po / total_forecast * 100) if total_forecast > 0 else 0
                
                # Buat HTML content untuk OVER
                html_content_over = f"""
                <div style="background: #FFF3E0; border-left: 5px solid #FF9800; padding: 20px; border-radius: 10px; margin: 20px 0;">
                    <h4 style="color: #EF6C00; margin-top: 0;">📈 OVER FORECAST SUMMARY - {last_month_name}</h4>
                    
                    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 24px; color: #FF9800; font-weight: bold; margin-bottom: 5px;">{avg_ratio:.1f}%</div>
                            <div style="font-size: 12px; color: #666;">Avg PO/Rofo</div>
                            <div style="font-size: 10px; color: #999;">Target: 80-120%</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #2E7D32; font-weight: bold; margin-bottom: 5px;">{total_forecast:,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Total Rofo</div>
                            <div style="font-size: 10px; color: #999;">Forecast Qty</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #1565C0; font-weight: bold; margin-bottom: 5px;">{total_po:,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Total PO</div>
                            <div style="font-size: 10px; color: #999;">Purchase Order</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 24px; color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: bold; margin-bottom: 5px;">{selisih_qty:+,.0f}</div>
                            <div style="font-size: 12px; color: #666;">Selisih Qty</div>
                            <div style="font-size: 11px; color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: 600;">({selisih_persen:+.1f}%)</div>
                        </div>
                        
                        <div style="flex: 1; min-width: 150px; background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                            <div style="font-size: 22px; color: #FF9800; font-weight: bold; margin-bottom: 5px;">{po_rofo_pct:.1f}%</div>
                            <div style="font-size: 12px; color: #666;">PO/Rofo %</div>
                            <div style="font-size: 10px; color: #999;">Overall Ratio</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255, 152, 0, 0.3); font-size: 14px; color: #666;">
                        <strong>Total OVER Forecast SKUs: {len(over_skus_df)}</strong> | 
                        <span style="color: #FF9800;">Avg PO/Rofo: {avg_ratio:.1f}%</span> | 
                        <span style="color: #2E7D32;">Rofo: {total_forecast:,.0f}</span> | 
                        <span style="color: #1565C0;">PO: {total_po:,.0f}</span> | 
                        <span style="color: {'#F44336' if selisih_qty < 0 else '#2E7D32'}; font-weight: bold;">Selisih: {selisih_qty:+,.0f} ({selisih_persen:+.1f}%)</span>
                    </div>
                </div>
                """
                
                # Tampilkan dengan st.html()
                st.html(html_content_over)
            else:
                st.success(f"✅ No SKUs with OVER forecast in {last_month_name}")

st.divider()


# --- MAIN TABS ---
tab1, tab2, tab5, tab3, tab4, tab10, tab_exp, tab_off, tab7, tab9, tab8, tab6 = st.tabs([
    "📈 Monthly Performance Details",
    "🏷️ Forecast Performance by Brand & Tier Analysis",
    "📈 Sales & Forecast Analysis",
    "📦 Inventory Analysis",
    "🔍 SKU Evaluation",
    "🚚 Fulfillment Cost Analysis",
    "🏬 Expansi Analytics",
    "🏪 Offline Store",
    "🛒 Ecommerce Forecast",
    "🤝 Reseller Forecast",
    "💰 Profitability Analysis",
    "📋 Data Explorer"
])

        
# --- TAB 1: MONTHLY PERFORMANCE DETAILS (PREMIUM HEATMAP) ---
with tab1:
    st.subheader("📅 Monthly Performance Details")
    
    if monthly_performance:
        # 1. Prepare Data
        summary_data = []
        prev_accuracy = 0
        
        for i, (month, data) in enumerate(sorted(monthly_performance.items())):
            # Tentukan Status & Icon
            acc = data['accuracy']
            if acc >= 90:
                status_icon = "🌟 Excellent"
            elif acc >= 80:
                status_icon = "✅ Good"
            elif acc >= 70:
                status_icon = "⚠️ Fair"
            else:
                status_icon = "🛑 Poor"
            
            # Hitung MoM Change (Delta)
            delta = acc - prev_accuracy if i > 0 else 0
            delta_str = f"{delta:+.1f}%" if i > 0 else "-"
            prev_accuracy = acc

            summary_data.append({
                'Month_Raw': month,
                'Month': month.strftime('%b %Y'),
                'Status': status_icon,
                'Accuracy': acc,
                'MoM': delta_str, # Month over Month Change
                'Under': data['status_counts'].get('Under', 0),
                'Accurate': data['status_counts'].get('Accurate', 0),
                'Over': data['status_counts'].get('Over', 0),
                'Total SKUs': data['total_records'],
                'MAPE': data['mape']
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # 2. Styling dengan Pandas Styler (Soft Pastel Heatmap)
        # Kita gunakan background_gradient untuk menyoroti angka yang tinggi
        
        def highlight_accuracy(val):
            color = '#d1fae5' if val >= 80 else '#fef3c7' if val >= 70 else '#fee2e2'
            return f'background-color: {color}; color: #374151; font-weight: bold;'

        # Create Styler Object
        styler = summary_df.style\
            .background_gradient(subset=['Under'], cmap='Reds', vmin=0, vmax=summary_df['Under'].max()*1.5)\
            .background_gradient(subset=['Accurate'], cmap='Greens', vmin=0, vmax=summary_df['Accurate'].max())\
            .background_gradient(subset=['Over'], cmap='Oranges', vmin=0, vmax=summary_df['Over'].max()*1.5)\
            .map(highlight_accuracy, subset=['Accuracy'])\
            .format({
                'Accuracy': '{:.1f}%',
                'MAPE': '{:.1f}%',
                'Under': '{:,}',
                'Accurate': '{:,}',
                'Over': '{:,}',
                'Total SKUs': '{:,}'
            })

        # 3. Render Dataframe
        st.dataframe(
            styler,
            column_order=['Month', 'Status', 'Accuracy', 'MoM', 'Under', 'Accurate', 'Over', 'Total SKUs', 'MAPE'],
            column_config={
                "Month": st.column_config.TextColumn("Period", help="Month of analysis"),
                "Status": st.column_config.TextColumn("Health Score"),
                "Accuracy": st.column_config.ProgressColumn(
                    "Accuracy %",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                ),
                "MoM": st.column_config.TextColumn("Trend (MoM)", help="Change from previous month"),
                "Under": st.column_config.NumberColumn("📉 Under", help="Count of Under-forecast SKUs"),
                "Accurate": st.column_config.NumberColumn("🎯 Accurate", help="Count of Accurate SKUs"),
                "Over": st.column_config.NumberColumn("📈 Over", help="Count of Over-forecast SKUs"),
                "Total SKUs": st.column_config.NumberColumn("Total Items"),
                "MAPE": st.column_config.NumberColumn("MAPE", help="Mean Absolute Percentage Error")
            },
            use_container_width=True,
            height=500,
            hide_index=True
        )
        
        # Legend Kecil
        st.caption("""
        🎨 **Color Legend:** - **Accuracy:** 🟩 Hijau (>80%), 🟨 Kuning (70-80%), 🟥 Merah (<70%).
        - **Under/Over:** Semakin pekat warnanya, semakin banyak SKU yang bermasalah di kategori tersebut.
        """)

        # Add forecast bias analysis (Existing code)
        if not forecast_bias.empty:
            # ... (kode existing bias analysis tetap ada di sini jika mau ditampilkan di bawah tabel)
            pass
        
        # ==============================================================================
        # 3. PREMIUM FORECAST BIAS ANALYSIS (Diverging Chart Version)
        # ==============================================================================
        if not forecast_bias.empty:
            st.divider()
            st.subheader("🎯 Forecast Bias & Health Analysis")
            
            # 1. Hitung Metrics Utama
            avg_bias_val = forecast_bias['Avg_Bias_Percentage'].mean()
            
            # Tentukan Tendency (Kecenderungan)
            if avg_bias_val > 5:
                tendency = "UNDER-FORECASTING (Demand > Plan)"
                tendency_icon = "📉" # Plan terlalu rendah
                tendency_color = "#3949ab" # Indigo
                risk_msg = "Risk: Potential Lost Sales (Stockout)"
            elif avg_bias_val < -5:
                tendency = "OVER-FORECASTING (Demand < Plan)"
                tendency_icon = "📈" # Plan ketinggian
                tendency_color = "#ef5350" # Red
                risk_msg = "Risk: Excess Stock & Obsolescence"
            else:
                tendency = "BALANCED (Good Accuracy)"
                tendency_icon = "⚖️"
                tendency_color = "#26a69a" # Teal
                risk_msg = "Status: Healthy Forecast"
    
            # Hitung Jumlah Bulan Warning
            critical_months = len(forecast_bias[abs(forecast_bias['Avg_Bias_Percentage']) > 20])
            
            # --- 2. BIAS HEALTH CARDS (CSS) ---
            st.markdown(f"""
            <style>
                .bias-card {{
                    background-color: white;
                    border-radius: 12px;
                    padding: 1.2rem;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                    border-left: 5px solid {tendency_color};
                    height: 100%;
                }}
                .bias-label {{ font-size: 0.8rem; color: #888; font-weight: 600; text-transform: uppercase; }}
                .bias-val {{ font-size: 1.8rem; font-weight: 800; color: #333; margin: 5px 0; }}
                .bias-sub {{ font-size: 0.9rem; color: {tendency_color}; font-weight: 600; }}
                .bias-desc {{ font-size: 0.8rem; color: #666; margin-top: 5px; }}
            </style>
            """, unsafe_allow_html=True)
    
            bc1, bc2, bc3 = st.columns(3)
    
            with bc1:
                st.markdown(f"""
                <div class="bias-card">
                    <div class="bias-label">AVERAGE BIAS (YTD)</div>
                    <div class="bias-val">{avg_bias_val:+.1f}%</div>
                    <div class="bias-sub">{tendency_icon} {tendency}</div>
                </div>
                """, unsafe_allow_html=True)
    
            with bc2:
                st.markdown(f"""
                <div class="bias-card" style="border-left-color: #ffa726;">
                    <div class="bias-label">IMPACT ANALYSIS</div>
                    <div class="bias-val" style="font-size: 1.2rem; margin-top: 15px;">{risk_msg}</div>
                    <div class="bias-desc">Based on average deviation direction</div>
                </div>
                """, unsafe_allow_html=True)
    
            with bc3:
                status_color = "#ef5350" if critical_months > 0 else "#26a69a"
                st.markdown(f"""
                <div class="bias-card" style="border-left-color: {status_color};">
                    <div class="bias-label">VOLATILITY CHECK</div>
                    <div class="bias-val">{critical_months} <span style="font-size:1rem;">Months</span></div>
                    <div class="bias-desc">Months with >20% Deviation (Critical)</div>
                </div>
                """, unsafe_allow_html=True)
    
            # --- 3. DIVERGING BAR CHART (Visualisasi Bias) ---
            st.write("") # Spacer
            
            # Prepare colors based on severity
            # Hijau: -10% s/d 10% (Aman)
            # Kuning: -20% s/d -10% ATAU 10% s/d 20% (Warning)
            # Merah: < -20% ATAU > 20% (Critical)
            
            colors = []
            for val in forecast_bias['Avg_Bias_Percentage']:
                if abs(val) <= 10:
                    colors.append('#4db6ac') # Soft Teal (Aman)
                elif abs(val) <= 20:
                    colors.append('#ffb74d') # Soft Orange (Warning)
                else:
                    colors.append('#ef5350') # Soft Red (Critical)
    
            fig_bias = go.Figure()
    
            fig_bias.add_trace(go.Bar(
                x=forecast_bias['Month'].dt.strftime('%b-%Y'),
                y=forecast_bias['Avg_Bias_Percentage'],
                text=[f"{x:+.1f}%" for x in forecast_bias['Avg_Bias_Percentage']],
                textposition='auto',
                marker_color=colors,
                name='Bias %'
            ))
    
            # Add Reference Lines (Zones)
            fig_bias.add_hrect(y0=-10, y1=10, fillcolor="green", opacity=0.05, line_width=0, annotation_text="Safe Zone", annotation_position="top left")
            fig_bias.add_hrect(y0=-20, y1=20, line_dash="dot", line_color="gray", fillcolor="yellow", opacity=0.05, line_width=1, annotation_text="Warning Limit")
    
            fig_bias.update_layout(
                title="<b>📉 Monthly Forecast Bias Trend</b> (Positive = Under Forecast, Negative = Over Forecast)",
                yaxis_title="Bias Percentage (%)",
                xaxis_title="Period",
                height=400,
                hovermode="x unified",
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black'), # Garis nol dipertegas
                plot_bgcolor='white',
                margin=dict(t=50, b=20, l=20, r=20)
            )
    
            st.plotly_chart(fig_bias, use_container_width=True)
            
            # Footer Note
            st.caption("""
            ℹ️ **Cara Membaca:**
            - **Bar ke Atas (+):** Realisasi (PO) > Forecast. Artinya **Under-Forecast** (Kurang plan, potensi lost sales).
            - **Bar ke Bawah (-):** Realisasi (PO) < Forecast. Artinya **Over-Forecast** (Plan ketinggian, potensi overstock).
            - **Zona Hijau:** Bias ±10% dianggap sehat.
            """)

# --- TAB 2: FORECAST PERFORMANCE BY BRAND & TIER ANALYSIS (FINAL FIXED) ---
with tab2:
    st.subheader("🏷️ Brand & Tier Strategic Analysis")
    st.caption("Portfolio Management: Brand Performance Positioning & Tier Health")

    # ==============================================================================
    # 1. DATA PREPARATION (Last Month vs Last 12 Months)
    # ==============================================================================
    
    # A. Tentukan Periode
    all_months = sorted(monthly_performance.keys()) if monthly_performance else []
    
    if not all_months:
        st.warning("⚠️ Belum ada data performa bulanan.")
        st.stop()

    last_month_date = all_months[-1]
    last_12_months_list = all_months[-12:] if len(all_months) >= 12 else all_months

    # B. Helper untuk Hitung Brand Performance berdasarkan List Bulan
    def get_brand_perf_by_period(months_target, label):
        # Filter Dataframes
        df_f_filtered = df_forecast[df_forecast['Month'].isin(months_target)].copy()
        df_p_filtered = df_po[df_po['Month'].isin(months_target)].copy()
        
        # Merge Basic Info
        df_f_filtered = add_product_info_to_data(df_f_filtered, df_product)
        df_p_filtered = add_product_info_to_data(df_p_filtered, df_product)
        
        # Group by Brand
        f_group = df_f_filtered.groupby('Brand')['Forecast_Qty'].sum().reset_index()
        p_group = df_p_filtered.groupby('Brand')['PO_Qty'].sum().reset_index()
        
        # Merge Forecast & PO
        merged = pd.merge(f_group, p_group, on='Brand', how='outer').fillna(0)
        
        # Hitung SKU Count (ambil dari forecast data active)
        sku_count = df_f_filtered.groupby('Brand')['SKU_ID'].nunique().reset_index(name='SKU_Count')
        merged = pd.merge(merged, sku_count, on='Brand', how='left').fillna(0)
        
        # ---------------------------------------------------------
        # PERBAIKAN LOGIKA AKURASI: HANYA HITUNG JIKA ROFO > 0
        # ---------------------------------------------------------
        sku_level = pd.merge(
            df_f_filtered.groupby(['Brand', 'SKU_ID'])['Forecast_Qty'].sum().reset_index(),
            df_p_filtered.groupby(['Brand', 'SKU_ID'])['PO_Qty'].sum().reset_index(),
            on=['Brand', 'SKU_ID'], how='outer'
        ).fillna(0)
        
        # FILTER: Buang SKU yang Rofo-nya 0 dari perhitungan akurasi
        valid_sku_level = sku_level[sku_level['Forecast_Qty'] > 0].copy()
        
        if not valid_sku_level.empty:
            # Hitung akurasi
            valid_sku_level['Accuracy'] = valid_sku_level.apply(
                lambda x: 100 - abs((x['PO_Qty']/x['Forecast_Qty']*100)-100), axis=1
            )
            # Opsional: Jika PO over sangat jauh (misal 300%), akurasi bisa minus. Kita batasi minimal 0%.
            valid_sku_level['Accuracy'] = valid_sku_level['Accuracy'].clip(lower=0)
            
            # Rata-rata akurasi per Brand (hanya dari SKU yang valid)
            brand_acc = valid_sku_level.groupby('Brand')['Accuracy'].mean().reset_index()
        else:
            brand_acc = pd.DataFrame(columns=['Brand', 'Accuracy'])
        
        # Gabungkan hasil akurasi ke dataframe utama
        final_df = pd.merge(merged, brand_acc, on='Brand', how='left')
        
        # Jika ada brand yang isinya Rofo 0 semua, akurasinya di-set 0 (atau bisa di-set None)
        final_df['Accuracy'] = final_df['Accuracy'].fillna(0)
        
        return final_df

    # C. UI Selector Period
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        view_period = st.radio(
            "📅 Pilih Periode Analisis:",
            ["Bulan Terakhir", "1 Tahun Terakhir (L12M)"],
            horizontal=False
        )

    # D. Generate Data sesuai Pilihan
    if view_period == "Bulan Terakhir":
        active_df = get_brand_perf_by_period([last_month_date], "Last Month")
        period_label = last_month_date.strftime('%B %Y')
    else:
        active_df = get_brand_perf_by_period(last_12_months_list, "Last 12M")
        period_label = f"Last 12 Months ({len(last_12_months_list)} periods)"

    st.markdown(f"#### 📊 Analisis Periode: {period_label}")

    if not active_df.empty:
        # ==============================================================================
        # 2. BRAND KPI CARDS (Soft Gradient - Fixed HTML)
        # ==============================================================================
        
        best_brand = active_df.loc[active_df['Accuracy'].idxmax()]
        high_vol_brand = active_df.loc[active_df['Forecast_Qty'].idxmax()]
        most_sku_brand = active_df.loc[active_df['SKU_Count'].idxmax()]
        
        # Calculate Weighted Accuracy Portfolio
        total_vol = active_df['Forecast_Qty'].sum()
        # Weighted avg accuracy
        weighted_acc = (active_df['Accuracy'] * active_df['Forecast_Qty']).sum() / total_vol if total_vol > 0 else 0

        # CSS TANPA INDENTASI
        st.markdown("""
<style>
.b-card {
border-radius: 12px;
padding: 1.2rem;
color: white;
box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
position: relative;
overflow: hidden;
transition: transform 0.3s ease;
}
.b-card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1); }
.b-label { font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9; margin-bottom: 5px; }
.b-val { font-size: 1.4rem; font-weight: 800; margin-bottom: 5px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.b-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; display: flex; align-items: center; gap: 5px; }
.b-badge { background: rgba(255,255,255,0.25); padding: 2px 8px; border-radius: 6px; font-size: 0.75rem; font-weight: 600; backdrop-filter: blur(4px); }
</style>
""", unsafe_allow_html=True)

        def render_brand_card_fixed(label, brand_name, metric_val, metric_label, gradient):
            return f"""
<div class="b-card" style="background: {gradient};">
<div class="b-label">{label}</div>
<div class="b-val">{brand_name}</div>
<div class="b-sub">
<span class="b-badge">{metric_val}</span> {metric_label}
</div>
</div>
"""

        bc1, bc2, bc3, bc4 = st.columns(4)
        
        with bc1:
            st.markdown(render_brand_card_fixed(
                "🏆 Best Accuracy", best_brand['Brand'], f"{best_brand['Accuracy']:.1f}%", "Avg Accuracy",
                "linear-gradient(135deg, #10B981 0%, #059669 100%)"
            ), unsafe_allow_html=True)
            
        with bc2:
            st.markdown(render_brand_card_fixed(
                "📦 Highest Volume", high_vol_brand['Brand'], f"{high_vol_brand['Forecast_Qty']:,.0f}", "Units Fcst",
                "linear-gradient(135deg, #6366F1 0%, #4338CA 100%)"
            ), unsafe_allow_html=True)
            
        with bc3:
            st.markdown(render_brand_card_fixed(
                "🗂️ Most SKUs", most_sku_brand['Brand'], f"{most_sku_brand['SKU_Count']}", "Active Items",
                "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)"
            ), unsafe_allow_html=True)
            
        with bc4:
            st.markdown(render_brand_card_fixed(
                "⚖️ Portfolio Health", "All Brands", f"{weighted_acc:.1f}%", "Vol. Wgt. Acc",
                "linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)"
            ), unsafe_allow_html=True)

        # ==============================================================================
        # 3. STRATEGIC MAGIC QUADRANT (SCATTER PLOT) - PREMIUM EDITION
        # ==============================================================================
        st.write("")
        st.subheader("🎯 Strategic Brand Positioning (S&OP Portfolio Matrix)")
        st.caption("Membagi brand ke dalam 4 kuadran strategis berdasarkan kontribusi Volume dan keandalan Akurasi.")
        
        # Hapus col_quad1, col_quad2, kita buat grafiknya full width (lebar penuh)
        scatter_data = active_df.copy()
        # Hindari error log-scale jika ada volume 0
        scatter_data = scatter_data[scatter_data['Forecast_Qty'] > 0]
        
        median_vol = scatter_data['Forecast_Qty'].median()
        target_acc = 80
        
        fig_quad = px.scatter(
            scatter_data,
            x='Forecast_Qty',
            y='Accuracy',
            size='SKU_Count',
            color='Accuracy',
            text='Brand',
            color_continuous_scale='RdYlGn',
            size_max=50,
            custom_data=['Brand', 'SKU_Count', 'Forecast_Qty', 'Accuracy']
        )
        
        # Quadrant Zones Background
        max_vol = scatter_data['Forecast_Qty'].max() * 1.5
        fig_quad.add_shape(type="rect",
            x0=median_vol, y0=0, x1=max_vol, y1=target_acc,
            fillcolor="rgba(239, 68, 68, 0.08)", line_width=0, layer="below" # Red tint
        )
        fig_quad.add_shape(type="rect",
            x0=median_vol, y0=target_acc, x1=max_vol, y1=110,
            fillcolor="rgba(16, 185, 129, 0.08)", line_width=0, layer="below" # Green tint
        )
        
        # Garis Pembatas (Crosshair)
        fig_quad.add_hline(y=target_acc, line_dash="dash", line_color="gray", annotation_text="Target Accuracy (80%)")
        fig_quad.add_vline(x=median_vol, line_dash="dash", line_color="gray", annotation_text="Median Volume")

        # --- FITUR PRO: WATERMARK LABEL LANGSUNG DI DALAM GRAFIK ---
        fig_quad.add_annotation(x=0.95, y=0.95, xref="paper", yref="paper", text="🌟 STARS<br><span style='font-size:12px'>High Vol, High Acc</span>", showarrow=False, font=dict(size=18, color="#10B981", weight="bold"), opacity=0.3, align="right")
        fig_quad.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text="🚨 RISK (PRIORITY FIX)<br><span style='font-size:12px'>High Vol, Low Acc</span>", showarrow=False, font=dict(size=18, color="#EF4444", weight="bold"), opacity=0.3, align="right")
        fig_quad.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text="❓ NICHE / QUESTION<br><span style='font-size:12px'>Low Vol, High Acc</span>", showarrow=False, font=dict(size=18, color="#F59E0B", weight="bold"), opacity=0.3, align="left")
        fig_quad.add_annotation(x=0.05, y=0.05, xref="paper", yref="paper", text="💤 SLEEPERS<br><span style='font-size:12px'>Low Vol, Low Acc</span>", showarrow=False, font=dict(size=18, color="#6B7280", weight="bold"), opacity=0.3, align="left")

        fig_quad.update_traces(
            textposition='top center',
            textfont=dict(weight='bold', color='#1F2937'),
            hovertemplate="<b>%{customdata[0]}</b><br>Accuracy: %{y:.1f}%<br>Volume: %{x:,.0f}<br>SKUs: %{marker.size}<extra></extra>"
        )
        
        fig_quad.update_layout(
            height=550, # Dibuat lebih tinggi agar lapang
            xaxis_title="Forecast Volume (Log Scale) ➡️",
            yaxis_title="Accuracy (%) ➡️",
            xaxis_type="log", 
            yaxis_range=[max(0, scatter_data['Accuracy'].min()-10), min(105, scatter_data['Accuracy'].max()+10)], # Dinamis
            plot_bgcolor="white",
            coloraxis_showscale=False, # Sembunyikan colorbar legend karena sudah ada zona warna
            margin=dict(t=20, l=20, r=20, b=20)
        )
        st.plotly_chart(fig_quad, use_container_width=True)
        
        # ==============================================================================
        # 4. BRAND PERFORMANCE COMBO CHART (BAR + LINE) - DYNAMIC COLORS
        # ==============================================================================
        st.divider()
        st.subheader("📊 Brand Detail: Volume vs Accuracy")
        st.caption("Membandingkan volume Forecast (Plan) vs PO (Eksekusi) beserta persentase akurasinya.")
        
        # Sort by Volume (Berdasarkan Forecast_Qty)
        chart_df = active_df.sort_values('Forecast_Qty', ascending=False)

        fig_combo = go.Figure()

        # Bar 1: Forecast Volume (Plan)
        fig_combo.add_trace(go.Bar(
            x=chart_df['Brand'],
            y=chart_df['Forecast_Qty'],
            name='Forecast Volume (Plan)',
            marker_color='rgba(99, 102, 241, 0.7)', # Soft Indigo
            marker_line_color='rgba(99, 102, 241, 1.0)',
            marker_line_width=1.5,
            yaxis='y1'
        ))

        # Bar 2: PO Volume (Execution)
        fig_combo.add_trace(go.Bar(
            x=chart_df['Brand'],
            y=chart_df['PO_Qty'],
            name='PO Volume (Execution)',
            marker_color='rgba(245, 158, 11, 0.7)', # Soft Amber/Orange
            marker_line_color='rgba(245, 158, 11, 1.0)',
            marker_line_width=1.5,
            yaxis='y1'
        ))

        # Dynamic Marker Colors untuk Line Chart
        line_colors = ['#10B981' if acc >= 80 else '#F59E0B' if acc >= 70 else '#EF4444' for acc in chart_df['Accuracy']]

        # Line: Accuracy
        fig_combo.add_trace(go.Scatter(
            x=chart_df['Brand'],
            y=chart_df['Accuracy'],
            name='Accuracy %',
            mode='lines+markers+text',
            text=[f"{acc:.1f}%" for acc in chart_df['Accuracy']],
            textposition="top center",
            textfont=dict(color='#1F2937', size=10, weight='bold'),
            line=dict(color='#4B5563', width=2), # Garis abu-abu gelap
            marker=dict(size=12, color=line_colors, line=dict(width=2, color='white')), # Titik warna-warni
            yaxis='y2'
        ))

        fig_combo.update_layout(
            height=480,
            barmode='group', # Menjadikan Bar Forecast & PO bersebelahan
            xaxis_title="Brand (Sorted by Forecast Volume)",
            yaxis=dict(
                title="Volume (Units)",
                showgrid=False
            ),
            yaxis2=dict(
                title="Accuracy (%)",
                overlaying='y',
                side='right',
                range=[0, 130], # Ruang ekstra agar label teks persentase tidak terpotong di atas
                showgrid=True,
                gridcolor='rgba(0,0,0,0.05)'
            ),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
            plot_bgcolor='white',
            margin=dict(t=50, b=0)
        )
        
        # Tambah garis target akurasi
        fig_combo.add_hline(y=80, line_dash="dot", line_color="#EF4444", yref="y2", annotation_text="Target 80%")

        st.plotly_chart(fig_combo, use_container_width=True)

        # ==============================================================================
        # 5. TIER ANALYSIS REPLACEMENT (STACKED BAR) - NEW!
        # ==============================================================================
        st.divider()
        c_tier1, c_tier2 = st.columns(2)
        
        with c_tier1:
            st.subheader("🧬 Tier & Brand Composition")
            st.caption("Komposisi Jumlah SKU berdasarkan Tier")
            
            # Sunburst tetap dipertahankan karena bagus untuk hierarki
            if not df_product.empty and 'SKU_Tier' in df_product.columns:
                sunburst_data = df_product[df_product['Status'].str.upper() == 'ACTIVE'].groupby(['SKU_Tier', 'Brand']).size().reset_index(name='Count')
                fig_sun = px.sunburst(
                    sunburst_data,
                    path=['SKU_Tier', 'Brand'],
                    values='Count',
                    color='SKU_Tier',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_sun.update_layout(height=400, margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(fig_sun, use_container_width=True)

        with c_tier2:
            st.subheader("🏆 Tier Performance (Bar Chart)")
            st.caption(f"Rata-rata Akurasi per Tier ({period_label})")
            
            # Hitung data per Tier untuk periode terpilih
            # Kita perlu re-calculate karena active_df tadi per Brand
            
            # 1. Filter raw data lagi berdasarkan periode
            df_f_tier = df_forecast[df_forecast['Month'].isin(last_12_months_list if view_period != "Bulan Terakhir" else [last_month_date])].copy()
            df_p_tier = df_po[df_po['Month'].isin(last_12_months_list if view_period != "Bulan Terakhir" else [last_month_date])].copy()
            
            # 2. Add Tier Info
            df_f_tier = add_product_info_to_data(df_f_tier, df_product)
            df_p_tier = add_product_info_to_data(df_p_tier, df_product)
            
            # 3. Group by Tier
            tier_stats = pd.merge(
                df_f_tier.groupby(['SKU_Tier', 'SKU_ID'])['Forecast_Qty'].sum().reset_index(),
                df_p_tier.groupby(['SKU_Tier', 'SKU_ID'])['PO_Qty'].sum().reset_index(),
                on=['SKU_Tier', 'SKU_ID'], how='outer'
            ).fillna(0)
            
            # Hitung akurasi per SKU
            tier_stats['Accuracy'] = tier_stats.apply(
                lambda x: 100 - abs((x['PO_Qty']/x['Forecast_Qty']*100)-100) if x['Forecast_Qty'] > 0 else 0, axis=1
            )
            
            # Average per Tier
            tier_summary = tier_stats.groupby('SKU_Tier')['Accuracy'].mean().reset_index()
            tier_summary = tier_summary.sort_values('Accuracy', ascending=False)
            
            # Bar Chart Horizontal
            fig_bar_tier = go.Figure()
            fig_bar_tier.add_trace(go.Bar(
                y=tier_summary['SKU_Tier'],
                x=tier_summary['Accuracy'],
                orientation='h',
                marker_color='#10B981', # Emerald
                text=[f"{x:.1f}%" for x in tier_summary['Accuracy']],
                textposition='auto'
            ))
            
            # Add Target Line
            fig_bar_tier.add_vline(x=80, line_dash="dash", line_color="red", annotation_text="Target 80%")
            
            fig_bar_tier.update_layout(
                height=400,
                title="Average Accuracy by Tier",
                xaxis_title="Accuracy %",
                yaxis_title="Tier",
                xaxis_range=[0, 110],
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig_bar_tier, use_container_width=True)

    else:
        st.info("📊 Data tidak tersedia untuk analisis brand.")

# --- TAB 3: INVENTORY HEALTH, COVERAGE & AGING (FIXED VALUE) ---
with tab3:
    st.subheader("📦 Inventory Health & Optimization Dashboard")
    st.caption("Comprehensive Stock Analysis: Value, Coverage, Warehouse Capacity, and Aging Profile")

    # ==============================================================================
    # 0. SIDEBAR INPUT FOR THIS TAB
    # ==============================================================================
    with st.expander("⚙️ Warehouse Settings", expanded=False):
        WH_CAPACITY = st.number_input(
            "🏢 Total Warehouse Capacity (pcs)",
            min_value=1000, max_value=10000000, value=250000, step=10000,
            help="Kapasitas maksimal gudang dalam satuan pcs/unit"
        )

    # ==============================================================================
    # 1. DATA PREPARATION & CLEANING (ROBUST FIX)
    # ==============================================================================
    if not df_stock.empty:
        df_batch = df_stock.copy()
        
        # 1.1. Standardize Category Column
        col_cat = 'Stock_Category'
        if col_cat not in df_batch.columns:
            candidates = [c for c in df_batch.columns if 'cat' in c.lower() or 'kategori' in c.lower()]
            col_cat = candidates[0] if candidates else None
        
        if col_cat:
            df_batch = df_batch.rename(columns={col_cat: 'Stock_Category'})
            
        # Clean Category & Qty
        if 'Stock_Category' in df_batch.columns:
            df_batch['Stock_Category'] = df_batch['Stock_Category'].astype(str).str.strip()
        else:
            df_batch['Stock_Category'] = 'Uncategorized'
            
        df_batch['Stock_Qty'] = pd.to_numeric(df_batch['Stock_Qty'], errors='coerce').fillna(0)
        df_batch = df_batch[df_batch['Stock_Qty'] > 0] # Filter stok > 0

        # 1.2. MERGE PRODUCT INFO (THE FIX IS HERE)
        # Hapus dulu kolom info produk yang mungkin menempel di df_stock (biar tidak double _x _y)
        cols_to_drop = ['Product_Name', 'Brand', 'Status', 'Floor_Price', 'SKU_Tier', 'Net_Order_Price']
        df_batch = df_batch.drop(columns=[c for c in cols_to_drop if c in df_batch.columns], errors='ignore')

        # Merge fresh dari Product Master
        if not df_product.empty:
            # Pastikan kolom-kolom ini ada di df_product
            master_cols = ['SKU_ID'] + [c for c in cols_to_drop if c in df_product.columns]
            
            # Merge
            df_batch = pd.merge(df_batch, df_product[master_cols], on='SKU_ID', how='left')
            
            # Fill missing text
            for txt_col in ['Status', 'Product_Name', 'Brand']:
                if txt_col in df_batch.columns:
                    df_batch[txt_col] = df_batch[txt_col].fillna('Unknown')

            # 1.3. CALCULATE VALUE
            if 'Floor_Price' in df_batch.columns:
                df_batch['Floor_Price'] = pd.to_numeric(df_batch['Floor_Price'], errors='coerce').fillna(0)
                df_batch['Total_Value'] = df_batch['Stock_Qty'] * df_batch['Floor_Price']
            else:
                df_batch['Total_Value'] = 0
                st.warning("⚠️ Kolom 'Floor_Price' tidak ditemukan di Product Master. Nilai Aset = 0.")

        # 1.4. EXPIRY LOGIC
        def get_expiry_desc(row):
            expiry_cols = [c for c in row.index if 'expir' in c.lower() or 'ed' in c.lower()]
            if not expiry_cols: return 'Not Defined'
            val = row[expiry_cols[0]]
            if pd.isna(val) or str(val).strip() in ['', '-', 'nan']: return 'Not Defined'
            try:
                exp_date = pd.to_datetime(val, dayfirst=True, errors='coerce')
                if pd.isna(exp_date): return 'Not Defined'
                days = (exp_date - pd.Timestamp.now()).days
                if days < 0: return '❌ EXPIRED'
                elif days <= 30: return '🚨 Critical (<30 Days)'
                elif days <= 90: return '⚠️ NED (1-3 Months)'
                elif days <= 180: return '📅 NED (3-6 Months)'
                elif days <= 365: return '✅ Safe (6-12 Months)'
                else: return '🌟 Fresh (>1 Year)'
            except: return 'Not Defined'

        df_batch['Expiry_Category'] = df_batch.apply(get_expiry_desc, axis=1)

        # 1.5. COVERAGE LOGIC
        df_stock_agg = df_batch.groupby('SKU_ID')['Stock_Qty'].sum().reset_index()
        
        # Get Sales Data
        df_avg_sales = pd.DataFrame()
        if not df_sales.empty:
            months = sorted(df_sales['Month'].unique())
            last_3 = months[-3:] if len(months) >= 3 else months
            df_sales_3m = df_sales[df_sales['Month'].isin(last_3)]
            df_avg_sales = df_sales_3m.groupby('SKU_ID')['Sales_Qty'].mean().reset_index()
            df_avg_sales.rename(columns={'Sales_Qty': 'Avg_Sales'}, inplace=True)
        
        df_cover = pd.merge(df_stock_agg, df_avg_sales, on='SKU_ID', how='left')
        df_cover['Avg_Sales'] = df_cover['Avg_Sales'].fillna(0)
        
        df_cover['Cover_Months'] = np.where(
            df_cover['Avg_Sales'] > 0, 
            df_cover['Stock_Qty'] / df_cover['Avg_Sales'], 
            999
        )
        
        total_global_stock = df_cover['Stock_Qty'].sum()
        total_global_avg_sales = df_cover['Avg_Sales'].sum()
        
        global_cover_months = (total_global_stock / total_global_avg_sales) if total_global_avg_sales > 0 else 0
        
        current_occupancy = df_batch['Stock_Qty'].sum()
        occupancy_pct = (current_occupancy / WH_CAPACITY * 100)

        # ==============================================================================
        # 2. EXECUTIVE KPI CARDS (PASTEL & SMART VALUE)
        # ==============================================================================
        total_val = df_batch['Total_Value'].sum()
        total_sku = df_batch['SKU_ID'].nunique()
        
        risk_mask = df_batch['Expiry_Category'].isin(['❌ EXPIRED', '🚨 Critical (<30 Days)'])
        risk_val = df_batch[risk_mask]['Total_Value'].sum()
        risk_pct = (risk_val / total_val * 100) if total_val > 0 else 0

        # Helper: Format Uang Pintar
        def format_currency_smart(value):
            if value >= 1_000_000_000: return f"Rp {value/1e9:,.1f} M"
            elif value >= 1_000_000: return f"Rp {value/1e6:,.1f} Jt"
            else: return f"Rp {value:,.0f}"

        # Gunakan Master Function dari Sidebar
        val_display = format_currency_smart(total_val)
        risk_display = format_currency_smart(risk_val)

        # CSS Styles
        st.markdown("""
        <style>
            .inv-card {
                border-radius: 12px; padding: 1.2rem; color: white;
                box-shadow: 0 4px 10px rgba(0,0,0,0.05); transition: transform 0.3s;
                position: relative; overflow: hidden;
            }
            .inv-card:hover { transform: translateY(-3px); }
            .inv-label { font-size: 0.8rem; font-weight: 700; opacity: 0.9; text-transform: uppercase; margin-bottom: 5px; }
            .inv-val { font-size: 1.6rem; font-weight: 800; margin-bottom: 5px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
            .inv-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; }
        </style>
        """, unsafe_allow_html=True)

        def render_inv_card(title, val, sub, bg):
            return f"""
            <div class="inv-card" style="background: {bg};">
                <div class="inv-label">{title}</div>
                <div class="inv-val">{val}</div>
                <div class="inv-sub">{sub}</div>
            </div>
            """

        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            # Soft Indigo
            st.markdown(render_inv_card("Total Asset Value", val_display, f"{total_sku:,} Items", 
                "linear-gradient(135deg, #7986cb 0%, #5c6bc0 100%)"), unsafe_allow_html=True)
        with c2:
            # Soft Teal
            st.markdown(render_inv_card("Total Quantity", f"{current_occupancy:,.0f}", f"{occupancy_pct:.1f}% Capacity", 
                "linear-gradient(135deg, #4db6ac 0%, #26a69a 100%)"), unsafe_allow_html=True)
        with c3:
            # Soft Orange
            st.markdown(render_inv_card("Global Stock Cover", f"{global_cover_months:.1f} Mo", "Total Stock / Total Sales", 
                "linear-gradient(135deg, #ffb74d 0%, #ffa726 100%)"), unsafe_allow_html=True)
        with c4:
            # Soft Red or Green depending on risk
            risk_bg = "linear-gradient(135deg, #ef5350 0%, #e53935 100%)" if risk_pct > 5 else "linear-gradient(135deg, #66bb6a 0%, #43a047 100%)"
            st.markdown(render_inv_card("Expiry Risk Value", risk_display, f"{risk_pct:.1f}% of Total", 
                risk_bg), unsafe_allow_html=True)

        # --- BARIS KEDUA: ADVANCED INVENTORY METRICS ---
        st.write("") # Spacer kecil agar tidak terlalu menempel
        
        # Ekstrak list SKU yang HANYA berstatus 'SKU Regular' (TANPA FILTER ACTIVE)
        regular_skus_only = df_batch[
            df_batch['Stock_Category'].str.contains('Regular', case=False, na=False)
        ]['SKU_ID'].unique()

        # Filter df_cover khusus untuk perhitungan KPI
        df_cover_filtered = df_cover[df_cover['SKU_ID'].isin(regular_skus_only)]
        
        # Kalkulasi Metrik Baru
        inv_turnover = (12 / global_cover_months) if global_cover_months > 0 else 0
        
        # 🔥 UBAH ANGKA 0.3 DI SINI MENJADI 0.1
        stockout_skus = len(df_cover_filtered[df_cover_filtered['Cover_Months'] < 0.2]) 
        
        total_skus_cover = len(df_cover_filtered)
        stockout_rate = (stockout_skus / total_skus_cover * 100) if total_skus_cover > 0 else 0
        replenish_skus = len(df_cover_filtered[df_cover_filtered['Cover_Months'] < 0.8])

        c5, c6, c7 = st.columns(3)
        
        with c5:
            # Soft Purple untuk Turnover
            st.markdown(render_inv_card("Inventory Turnover", f"{inv_turnover:.1f}x", "Annualized Ratio", 
                "linear-gradient(135deg, #ab47bc 0%, #8e24aa 100%)"), unsafe_allow_html=True)
        with c6:
            # Merah Tua jika Stockout Rate > 5%, Hijau jika aman
            so_bg = "linear-gradient(135deg, #e53935 0%, #c62828 100%)" if stockout_rate > 5 else "linear-gradient(135deg, #43a047 0%, #2e7d32 100%)"
            st.markdown(render_inv_card("Stock Out Rate", f"{stockout_rate:.1f}%", f"{stockout_skus} SKUs (< 0.2 Mo)", 
                so_bg), unsafe_allow_html=True)
        with c7:
            # Amber/Orange Tua untuk Need Replenishment
            st.markdown(render_inv_card("Need Replenishment", f"{replenish_skus} SKUs", "SKUs (< 0.8 Mo Cover)", 
                "linear-gradient(135deg, #fb8c00 0%, #ef6c00 100%)"), unsafe_allow_html=True)

        # ==============================================================================
        # 3. STOCK COVER & OCCUPANCY DASHBOARD (GAUGE DENGAN 1 DESIMAL)
        # ==============================================================================
        st.write("")
        st.subheader("⚡ Inventory Health & Warehouse Utilization")
        
        col_speed1, col_speed2 = st.columns(2)
        
        with col_speed1:
            # Gauge: Global Coverage menggunakan ECharts
            if ECHARTS_AVAILABLE:
                # Format value dengan 1 desimal
                cover_value = round(global_cover_months, 1)
                
                # Dynamic Color Logic untuk Stock Coverage
                if cover_value < 0.8:
                    c_color = '#EF4444' # Merah (Need Replenishment)
                elif cover_value <= 2.0:
                    c_color = '#10B981' # Hijau (Ideal/Aman)
                else:
                    c_color = '#F59E0B' # Kuning/Orange (Overstock)
                
                option_cover = {
                    "series": [{
                        "type": 'gauge',
                        "center": ['50%', '55%'], # Digeser sedikit agar proporsional
                        "radius": '85%',
                        "startAngle": 210,
                        "endAngle": -30,
                        "min": 0,
                        "max": 6,
                        "splitNumber": 6,
                        "progress": {
                            "show": True,
                            "width": 18,
                            "roundCap": True,
                            "itemStyle": {"color": c_color}
                        },
                        "axisLine": {
                            "lineStyle": {
                                "width": 18,
                                "color": [
                                    [0.133, 'rgba(239,68,68,0.2)'],
                                    [0.333, 'rgba(16,185,129,0.2)'],
                                    [1, 'rgba(245,158,11,0.2)']
                                ]
                            }
                        },
                        "pointer": {
                            "itemStyle": {"color": c_color} # FIX: Warna jarum solid mengikuti status
                        },
                        "axisTick": {"show": False},
                        "splitLine": {"show": False},
                        "axisLabel": {
                            "show": True, 
                            "distance": -35,
                            "fontSize": 11, 
                            "fontWeight": 'bold', 
                            "color": '#6B7280',
                            "formatter": '{value}'  
                        },
                        "title": {
                            "show": True,
                            "offsetCenter": [0, '50%'], # Judul masuk ke dalam lingkaran
                            "fontSize": 13,
                            "fontWeight": 'bold',
                            "color": '#6B7280'
                        },
                        "detail": {
                            "show": True,
                            "valueAnimation": True,
                            "fontSize": 38,
                            "fontWeight": '900',
                            "offsetCenter": [0, '15%'], # Angka masuk ke dalam, pas di bawah jarum
                            "color": '#1F2937',
                            "formatter": '{value} Mo'
                        },
                        "data": [{
                            "value": cover_value,
                            "name": "Stock Coverage"
                        }]
                    }]
                }
                
                try:
                    st_echarts(options=option_cover, height="350px", key="gauge_cover_tab3")
                except Exception as e:
                    st.warning(f"⚠️ Gagal render gauge: {str(e)}")
                    st.metric("Global Stock Cover", f"{global_cover_months:.1f} Mo")
            else:
                st.metric("Global Stock Cover", f"{global_cover_months:.1f} Mo")
            
            st.caption(f"📊 Current: **{global_cover_months:.1f} Bulan** | Target: **0.8 - 2.0 Bulan**")
        
        with col_speed2:
            # Gauge: WH Occupancy
            if ECHARTS_AVAILABLE:
                occ_value = round(occupancy_pct, 1)
                
                # Logic warna: <80% Hijau, 80-90% Kuning, >90% Merah
                occ_color = "#10B981" if occ_value < 80 else "#F59E0B" if occ_value < 90 else "#EF4444"
                
                option_occ = {
                    "series": [{
                        "type": 'gauge',
                        "center": ['50%', '55%'],
                        "radius": '85%',
                        "startAngle": 210,
                        "endAngle": -30,
                        "min": 0,
                        "max": 100,
                        "splitNumber": 5,
                        "progress": {
                            "show": True,
                            "width": 18,
                            "roundCap": True,
                            "itemStyle": {"color": occ_color}
                        },
                        "axisLine": {
                            "lineStyle": {
                                "width": 18,
                                "color": [
                                    [0.8, 'rgba(16,185,129,0.2)'], 
                                    [0.9, 'rgba(245,158,11,0.2)'], 
                                    [1, 'rgba(239,68,68,0.2)']
                                ]
                            }
                        },
                        "pointer": {
                            "itemStyle": {"color": occ_color} # FIX: Warna jarum solid mengikuti status
                        },
                        "axisTick": {"show": False},
                        "splitLine": {"show": False},
                        "axisLabel": {
                            "show": True, 
                            "distance": -35,
                            "fontSize": 11, 
                            "fontWeight": 'bold', 
                            "color": '#6B7280',
                            "formatter": '{value}%'
                        },
                        "title": {
                            "show": True,
                            "offsetCenter": [0, '50%'], # Judul masuk ke dalam lingkaran
                            "fontSize": 13,
                            "fontWeight": 'bold',
                            "color": '#6B7280'
                        },
                        "detail": {
                            "show": True,
                            "valueAnimation": True,
                            "fontSize": 38,
                            "fontWeight": '900',
                            "offsetCenter": [0, '15%'], # Angka masuk ke dalam, pas di bawah jarum
                            "color": '#1F2937',
                            "formatter": '{value}%'
                        },
                        "data": [{
                            "value": occ_value,
                            "name": "Occupancy"
                        }]
                    }]
                }
                
                try:
                    st_echarts(options=option_occ, height="350px", key="gauge_occ_tab3")
                except Exception as e:
                    st.warning(f"⚠️ Gagal render gauge: {str(e)}")
                    st.metric("Warehouse Occupancy", f"{occupancy_pct:.1f}%")
            else:
                st.metric("Warehouse Occupancy", f"{occupancy_pct:.1f}%")
            
            st.caption(f"📦 Capacity Used: **{current_occupancy:,.0f}** / **{WH_CAPACITY:,.0f}** pcs ({occ_value:.1f}%)")

        # ==============================================================================
        # 3.5 ACTIONABLE INVENTORY ALERTS (ACTIVE & REGULAR SKU ONLY)
        # ==============================================================================
        st.divider()
        st.subheader("🚨 Actionable Inventory Alerts (Active & Regular SKU Only)")
        st.caption("Daftar *SKU Regular* berstatus **Active** yang membutuhkan tindakan operasional segera berdasarkan sales 3 bulan terakhir.")
        
        # Ekstrak list SKU yang HANYA berstatus 'SKU Regular' DAN 'Active'
        active_regular_skus = df_batch[
            (df_batch['Stock_Category'].str.contains('Regular', case=False, na=False)) &
            (df_batch['Status'].str.upper() == 'ACTIVE')
        ]['SKU_ID'].unique()
        
        if 'high_stock' in inventory_metrics and 'low_stock' in inventory_metrics:
            df_low = inventory_metrics['low_stock'].copy()
            df_high = inventory_metrics['high_stock'].copy()
            
            # Terapkan Filter Lapis Ganda & SORTING (Urutkan dari yang terparah)
            df_low = df_low[df_low['SKU_ID'].isin(active_regular_skus)].sort_values('Cover_Months', ascending=True) # Stock paling kritis (0.0) di atas
            df_high = df_high[df_high['SKU_ID'].isin(active_regular_skus)].sort_values('Cover_Months', ascending=False) # Dead stock terlama (999) di atas
            
            cols_to_show = ['SKU_ID', 'Product_Name', 'Stock_Qty', 'Avg_Monthly_Sales_3M', 'Cover_Months']
            
            col_alert1, col_alert2 = st.columns(2)
            
            with col_alert1:
                st.markdown(f"**📉 Need Replenishment (< 0.8 Bulan): <span style='color:#EF4444;'>{len(df_low)} SKUs</span>**", unsafe_allow_html=True)
                if not df_low.empty:
                    disp_low = df_low[cols_to_show].rename(columns={'Avg_Monthly_Sales_3M':'Sales/Mo', 'Cover_Months':'Cover'})
                    disp_low['Stock_Qty'] = disp_low['Stock_Qty'].apply(lambda x: f"{x:,.0f}")
                    disp_low['Sales/Mo'] = disp_low['Sales/Mo'].apply(lambda x: f"{x:,.0f}")
                    disp_low['Cover'] = disp_low['Cover'].apply(lambda x: f"{x:.1f}")
                    st.dataframe(disp_low, use_container_width=True, height=250, hide_index=True)
                else:
                    st.success("✅ Tidak ada SKU kritis. Semua aman!")
                    
            with col_alert2:
                st.markdown(f"**📦 Overstock / Dead Stock Alert (> 2 Bulan): <span style='color:#F59E0B;'>{len(df_high)} SKUs</span>**", unsafe_allow_html=True)
                if not df_high.empty:
                    disp_high = df_high[cols_to_show].rename(columns={'Avg_Monthly_Sales_3M':'Sales/Mo', 'Cover_Months':'Cover'})
                    disp_high['Stock_Qty'] = disp_high['Stock_Qty'].apply(lambda x: f"{x:,.0f}")
                    disp_high['Sales/Mo'] = disp_high['Sales/Mo'].apply(lambda x: f"{x:,.0f}")
                    disp_high['Cover'] = disp_high['Cover'].apply(lambda x: "No Sales" if x > 900 else f"{x:.1f}")
                    st.dataframe(disp_high, use_container_width=True, height=250, hide_index=True)
                else:
                    st.success("✅ Tidak ada SKU Overstock. Gudang efisien!")

        # ==============================================================================
        # 4. ABC ANALYSIS & FINANCIAL AGING EXPOSURE (KHUSUS SKU REGULAR)
        # ==============================================================================
        st.divider()
        st.subheader("🧬 Strategic Inventory Classification & Risk Exposure (SKU Regular)")
        st.caption("Memetakan prioritas *SKU Regular* berdasarkan nilai aset (ABC Analysis) dan risiko kedaluwarsa secara finansial.")
        
        # Buat dataframe khusus SKU Regular untuk perhitungan ABC & Aging
        df_batch_regular = df_batch[df_batch['Stock_Category'].str.contains('Regular', case=False, na=False)].copy()
        
        col_abc, col_age = st.columns([1, 1.2])
        
        with col_abc:
            # --- KONSEP ABC ANALYSIS ---
            st.markdown("**📊 ABC Inventory Classification (By Value)**")
            
            if not df_batch_regular.empty:
                df_abc = df_batch_regular.groupby(['SKU_ID', 'Product_Name'])['Total_Value'].sum().reset_index()
                df_abc = df_abc.sort_values('Total_Value', ascending=False)
                df_abc['Cum_Value'] = df_abc['Total_Value'].cumsum()
                df_abc['Cum_Pct'] = df_abc['Cum_Value'] / df_abc['Total_Value'].sum() * 100
                
                def classify_abc(pct):
                    if pct <= 80: return 'A (Top 80% Value)'
                    elif pct <= 95: return 'B (Next 15% Value)'
                    else: return 'C (Bottom 5% Value)'
                    
                df_abc['ABC_Class'] = df_abc['Cum_Pct'].apply(classify_abc)
                abc_summary = df_abc.groupby('ABC_Class').agg(
                    SKU_Count=('SKU_ID', 'count'),
                    Total_Value=('Total_Value', 'sum')
                ).reset_index()
                
                fig_abc = px.pie(abc_summary, values='Total_Value', names='ABC_Class', hole=0.5,
                                 color='ABC_Class',
                                 color_discrete_map={
                                     'A (Top 80% Value)': '#10B981', # Emerald
                                     'B (Next 15% Value)': '#3B82F6', # Blue
                                     'C (Bottom 5% Value)': '#9CA3AF'  # Gray
                                 },
                                 custom_data=['SKU_Count'])
                
                fig_abc.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate="<b>%{label}</b><br>Value: Rp %{value:,.0f}<br>Total SKUs: %{customdata[0]}<extra></extra>"
                )
                fig_abc.update_layout(height=380, showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
                
                fig_abc.add_annotation(text=f"<b>{len(df_abc)}</b><br>Regular SKUs", x=0.5, y=0.5, font_size=18, showarrow=False)
                st.plotly_chart(fig_abc, use_container_width=True)
            else:
                st.info("Tidak ada data SKU Regular.")

        with col_age:
            # --- FINANCIAL AGING PROFILE ---
            st.markdown("**⏳ Financial Exposure by Expiry Status**")
            
            if not df_batch_regular.empty:
                age_dist = df_batch_regular.groupby('Expiry_Category').agg({'Total_Value': 'sum', 'Stock_Qty': 'sum'}).reset_index()
                order_list = ['❌ EXPIRED', '🚨 Critical (<30 Days)', '⚠️ NED (1-3 Months)', '📅 NED (3-6 Months)', '✅ Safe (6-12 Months)', '🌟 Fresh (>1 Year)', 'Not Defined']
                age_dist['Expiry_Category'] = pd.Categorical(age_dist['Expiry_Category'], categories=order_list, ordered=True)
                age_dist = age_dist.sort_values('Expiry_Category')
                
                color_map = {
                    '❌ EXPIRED': '#EF4444', '🚨 Critical (<30 Days)': '#F87171',
                    '⚠️ NED (1-3 Months)': '#F59E0B', '📅 NED (3-6 Months)': '#FBBF24',
                    '✅ Safe (6-12 Months)': '#34D399', '🌟 Fresh (>1 Year)': '#10B981', 'Not Defined': '#9CA3AF'
                }
                
                fig_age = px.bar(
                    age_dist, x='Total_Value', y='Expiry_Category', orientation='h',
                    color='Expiry_Category', color_discrete_map=color_map,
                    text=age_dist['Total_Value'].apply(lambda x: f"Rp {x/1e6:,.0f} Jt" if x > 0 else "")
                )
                
                fig_age.update_traces(textposition='outside', textfont=dict(weight='bold', color='#4B5563'))
                fig_age.update_layout(
                    height=380, showlegend=False, plot_bgcolor='white',
                    xaxis=dict(title="Trapped Capital (Rupiah)", showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                    yaxis=dict(title="", autorange="reversed"),
                    margin=dict(t=10, l=10, r=40, b=10)
                )
                st.plotly_chart(fig_age, use_container_width=True)

        # ==============================================================================
        # 5. INVENTORY MATRIX (CATEGORY VS EXPIRY) - PREMIUM STYLING
        # ==============================================================================
        st.divider()
        st.subheader("🗓️ Inventory Matrix: Risk Detection")
        st.caption("Peta persebaran kuantitas stok berdasarkan Kategori dan Status Kedaluwarsa.")
        
        pivot = pd.pivot_table(df_batch, values='Stock_Qty', index='Stock_Category', columns='Expiry_Category', aggfunc='sum', fill_value=0)
        existing_cols = [c for c in order_list if c in pivot.columns]
        pivot = pivot[existing_cols]
        pivot['TOTAL (Qty)'] = pivot.sum(axis=1)
        pivot = pivot.sort_values('TOTAL (Qty)', ascending=False)
        
        # Styling dengan warna background yang lebih halus (Soft Reds untuk kolom bahaya)
        risk_cols = [c for c in existing_cols if 'EXPIRED' in c or 'Critical' in c or 'NED' in c]
        safe_cols = [c for c in existing_cols if 'Safe' in c or 'Fresh' in c]
        
        styler = pivot.style.format("{:,.0f}")
        if risk_cols:
            styler = styler.background_gradient(cmap='Reds', subset=risk_cols, vmin=0)
        if safe_cols:
            styler = styler.background_gradient(cmap='Greens', subset=safe_cols, vmin=0)
            
        st.dataframe(styler, use_container_width=True, height=350)

        # ==============================================================================
        # 6. DRILL-DOWN ANALYSIS
        # ==============================================================================
        st.divider()
        with st.expander("🔍 Drill-Down & Search Data", expanded=True):
            f1, f2, f3 = st.columns([1, 1, 2])
            with f1: sel_status = st.multiselect("Status:", df_batch['Status'].unique(), default=['Active'])
            with f2: sel_expiry = st.multiselect("Expiry:", df_batch['Expiry_Category'].unique())
            with f3: search_sku = st.text_input("Search (SKU/Name):", placeholder="Type here...")
            
            df_drill = df_batch.copy()
            if sel_status: df_drill = df_drill[df_drill['Status'].isin(sel_status)]
            if sel_expiry: df_drill = df_drill[df_drill['Expiry_Category'].isin(sel_expiry)]
            if search_sku: 
                df_drill = df_drill[df_drill['SKU_ID'].str.contains(search_sku, case=False) | df_drill['Product_Name'].str.contains(search_sku, case=False)]
            
            # Display
            cols = ['SKU_ID', 'Product_Name', 'Status', 'Stock_Category', 'Expiry_Category', 'Stock_Qty', 'Floor_Price', 'Total_Value']
            if 'Expiry_Date' in df_batch.columns: cols.insert(5, 'Expiry_Date')
            
            final_cols = [c for c in cols if c in df_drill.columns]
            
            st.dataframe(
                df_drill[final_cols].sort_values('Total_Value', ascending=False), 
                column_config={
                    "Total_Value": st.column_config.NumberColumn("Value", format="Rp %d"),
                    "Floor_Price": st.column_config.NumberColumn("Price", format="Rp %d"),
                    "Stock_Qty": st.column_config.NumberColumn("Qty")
                },
                use_container_width=True
            )

    else:
        st.warning("⚠️ Data Stok Kosong.")

# --- TAB 4: SKU EVALUATION (SKU 360 INSIGHT DECK) ---
with tab4:
    st.subheader("🔍 SKU 360° Deep Dive Analysis")
    st.caption("Micro-level analysis for individual Product Performance & Health")

    # 1. SKU SELECTOR & DATA PREP
    if monthly_performance and not df_sales.empty:
        # Get last month for evaluation
        last_month = sorted(monthly_performance.keys())[-1]
        last_month_data = monthly_performance[last_month]['data'].copy()
        
        # Prepare list for dropdown
        available_skus = []
        if not last_month_data.empty:
                # Sort by highest forecast volume (Pareto) to show important items first
                sorted_skus = last_month_data.sort_values('Forecast_Qty', ascending=False)
                
                # 🔥 HAPUS .head(200) AGAR SEMUA SKU MASUK KE LIST
                for _, row in sorted_skus.iterrows(): 
                    sku_label = f"{row['SKU_ID']} - {row.get('Product_Name', 'N/A')}"
                    available_skus.append(sku_label)


        # UI Selectbox
        col_sel1, col_sel2 = st.columns([2, 1])
        with col_sel1:
            # 🔥 UBAH LABEL TEXT-NYA
            selected_sku_display = st.selectbox(
                "📋 Select SKU to Analyze (All SKUs)", 
                options=available_skus
            )
        
        if selected_sku_display:
            selected_sku = selected_sku_display.split(" - ")[0]
            
            # Get SKU Details
            sku_details = last_month_data[last_month_data['SKU_ID'] == selected_sku].iloc[0]
            
            # Add Inventory & Sales Stats
            stock_qty = 0
            avg_sales_3m = 0
            cover_months = 0
            
            if 'inventory_df' in inventory_metrics:
                inv_row = inventory_metrics['inventory_df'][inventory_metrics['inventory_df']['SKU_ID'] == selected_sku]
                if not inv_row.empty:
                    stock_qty = inv_row.iloc[0]['Stock_Qty']
                    avg_sales_3m = inv_row.iloc[0].get('Avg_Monthly_Sales_3M', 0)
                    cover_months = inv_row.iloc[0].get('Cover_Months', 0)

            # ==============================================================================
            # 2. SKU PROFILE HEADER (HTML/CSS)
            # ==============================================================================
            st.markdown("""
            <style>
                .sku-header {
                    background-color: white;
                    border-radius: 12px;
                    padding: 1.5rem;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
                    border-left: 6px solid #6366F1;
                    margin-bottom: 1.5rem;
                }
                .sku-title { font-size: 1.4rem; font-weight: 800; color: #1F2937; margin-bottom: 0.5rem; }
                .sku-badges { display: flex; gap: 10px; flex-wrap: wrap; }
                .badge { 
                    padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; 
                    display: flex; align-items: center; gap: 5px;
                }
                .badge-blue { background: #E0E7FF; color: #4338CA; }
                .badge-purple { background: #F3E8FF; color: #7E22CE; }
                .badge-gray { background: #F3F4F6; color: #4B5563; }
                .price-tag { font-size: 1.1rem; font-weight: 700; color: #059669; }
            </style>
            """, unsafe_allow_html=True)

            product_name = sku_details.get('Product_Name', 'Unknown')
            brand = sku_details.get('Brand', 'Unknown')
            tier = sku_details.get('SKU_Tier', 'Standard')
            price = sku_details.get('Floor_Price', 0)
            
            st.markdown(f"""
            <div class="sku-header">
                <div class="sku-title">{product_name} <span style="font-weight:400; font-size:1rem; color:#6B7280;">({selected_sku})</span></div>
                <div class="sku-badges">
                    <span class="badge badge-blue">🏷️ {brand}</span>
                    <span class="badge badge-purple">💎 {tier} Tier</span>
                    <span class="badge badge-gray" style="margin-left:auto;">Floor Price: <span class="price-tag">Rp {price:,.0f}</span></span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ==============================================================================
            # 3. KEY METRICS GRID (Pastel Gradient Cards)
            # ==============================================================================
            # Prepare Colors based on Logic
            cover_color = "linear-gradient(135deg, #10B981 0%, #059669 100%)" # Green (Ideal)
            if cover_months < 0.8: cover_color = "linear-gradient(135deg, #EF4444 0%, #B91C1C 100%)" # Red
            elif cover_months > 2.0: cover_color = "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)" # Orange

            # Helper for Card
            def render_sku_card(label, val, sub, bg):
                return f"""
                <div style="background: {bg}; border-radius: 12px; padding: 1.2rem; color: white; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <div style="font-size: 0.8rem; font-weight: 600; opacity: 0.9; text-transform: uppercase;">{label}</div>
                    <div style="font-size: 1.6rem; font-weight: 800; margin: 5px 0;">{val}</div>
                    <div style="font-size: 0.85rem; opacity: 0.95; font-weight: 500;">{sub}</div>
                </div>
                """

            k1, k2, k3, k4 = st.columns(4)
            
            with k1:
                st.markdown(render_sku_card("Current Stock", f"{stock_qty:,.0f}", "Units Available", 
                    "linear-gradient(135deg, #6366F1 0%, #4338CA 100%)"), unsafe_allow_html=True) # Indigo
            with k2:
                st.markdown(render_sku_card("Stock Cover", f"{cover_months:.1f} Mo", "Inventory Health", 
                    cover_color), unsafe_allow_html=True) # Dynamic
            with k3:
                st.markdown(render_sku_card("Avg Sales (3M)", f"{avg_sales_3m:,.0f}", "Monthly Velocity", 
                    "linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%)"), unsafe_allow_html=True) # Sky Blue
            with k4:
                # Calculate Accuracy if data exists
                acc_val = "N/A"
                if sku_details['Forecast_Qty'] > 0:
                    acc = (sku_details['PO_Qty'] / sku_details['Forecast_Qty'] * 100)
                    acc_val = f"{acc:.1f}%"
                
                st.markdown(render_sku_card("PO vs Rofo", acc_val, f"Last Month: {last_month.strftime('%b')}", 
                    "linear-gradient(135deg, #EC4899 0%, #DB2777 100%)"), unsafe_allow_html=True) # Pink

            # ==============================================================================
            # 4. SUPPLY CHAIN PULSE CHART (Combo Chart)
            # ==============================================================================
            st.write("")
            st.subheader("📈 Supply Chain Pulse (Trend Analysis)")
            
            # Prepare Historical Data
            hist_data = []
            if not df_sales.empty:
                sales_months = sorted(df_sales['Month'].unique())
                # Tampilkan seluruh horizon historis data yang tersedia
                target_months = sales_months
                
                for m in target_months:
                    s_qty = df_sales[(df_sales['Month'] == m) & (df_sales['SKU_ID'] == selected_sku)]['Sales_Qty'].sum()
                    f_qty = df_forecast[(df_forecast['Month'] == m) & (df_forecast['SKU_ID'] == selected_sku)]['Forecast_Qty'].sum() if not df_forecast.empty else 0
                    p_qty = df_po[(df_po['Month'] == m) & (df_po['SKU_ID'] == selected_sku)]['PO_Qty'].sum() if not df_po.empty else 0
                    
                    hist_data.append({
                        'Month': m,
                        'Month_Txt': m.strftime('%b-%y'),
                        'Sales': s_qty,
                        'Forecast': f_qty,
                        'PO': p_qty
                    })
            
            if hist_data:
                df_hist = pd.DataFrame(hist_data)
                
                fig = go.Figure()

                # Area: Sales (Realization)
                fig.add_trace(go.Scatter(
                    x=df_hist['Month_Txt'], y=df_hist['Sales'],
                    name='Sales (Real)',
                    mode='lines',
                    fill='tozeroy',
                    line=dict(color='#10B981', width=3), # Emerald Green
                    fillcolor='rgba(16, 185, 129, 0.1)'
                ))

                # Line: Forecast (Plan)
                fig.add_trace(go.Scatter(
                    x=df_hist['Month_Txt'], y=df_hist['Forecast'],
                    name='Forecast (Plan)',
                    mode='lines+markers',
                    line=dict(color='#6366F1', width=3, dash='dash'), # Indigo Dashed
                    marker=dict(size=6, color='#6366F1')
                ))

                # Bar: PO (Execution)
                fig.add_trace(go.Bar(
                    x=df_hist['Month_Txt'], y=df_hist['PO'],
                    name='PO (Order)',
                    marker_color='rgba(245, 158, 11, 0.4)', # Orange Transparent
                    marker_line_color='#F59E0B',
                    marker_line_width=1.5
                ))

                fig.update_layout(
                    height=450,
                    title=dict(text="<b>Plan vs Execution vs Reality</b>", font=dict(size=16)),
                    hovermode="x unified",
                    plot_bgcolor="white",
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', title="Units"),
                    margin=dict(t=50, l=20, r=20, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # ==============================================================================
            # 5. SMART DIAGNOSTICS & RECOMMENDATION
            # ==============================================================================
            st.write("")
            c_diag1, c_diag2 = st.columns([1.5, 1])

            with c_diag1:
                st.subheader("🩺 Smart Diagnostics")
                
                # Logic Diagnosa
                diagnoses = []
                
                # Cek Stock Cover
                if cover_months < 0.8:
                    diagnoses.append(("🔴", "High Stockout Risk", f"Stock cover is only {cover_months:.1f} months. Urgent replenishment needed."))
                elif cover_months > 2.0:
                    diagnoses.append(("🟡", "Overstock Alert", f"Stock cover is {cover_months:.1f} months. Consider holding PO or running promo."))
                else:
                    diagnoses.append(("🟢", "Healthy Inventory", "Stock levels are optimal (0.8 - 2.0 months)."))

                # Cek Tren Sales (Growth)
                if len(df_hist) >= 3:
                    last_3_sales = df_hist['Sales'].tail(3).mean()
                    prev_3_sales = df_hist['Sales'].iloc[-6:-3].mean() if len(df_hist) >= 6 else last_3_sales
                    
                    if prev_3_sales > 0:
                        growth = (last_3_sales - prev_3_sales) / prev_3_sales * 100
                        if growth > 20:
                            diagnoses.append(("🚀", "Surging Demand", f"Sales trend is up +{growth:.1f}% vs prev period. Adjust forecast up."))
                        elif growth < -20:
                            diagnoses.append(("📉", "Declining Sales", f"Sales trend is down {growth:.1f}%. Review forecast down."))

                # Render Diagnosa
                for icon, title, desc in diagnoses:
                    bg_col = "#F0FDF4" if icon == "🟢" else "#FEF2F2" if icon == "🔴" else "#FFFBEB"
                    border_col = "#22C55E" if icon == "🟢" else "#EF4444" if icon == "🔴" else "#F59E0B"
                    
                    st.markdown(f"""
                    <div style="background:{bg_col}; border-left:4px solid {border_col}; padding:12px; border-radius:8px; margin-bottom:10px;">
                        <div style="font-weight:700; color:#374151; display:flex; align-items:center; gap:8px;">
                            <span style="font-size:1.2rem;">{icon}</span> {title}
                        </div>
                        <div style="font-size:0.9rem; color:#4B5563; margin-left:32px;">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with c_diag2:
                # Mini Stats Table
                st.subheader("📋 Quick Stats")
                
                # Data Variance
                last_fcst = df_hist.iloc[-1]['Forecast'] if hist_data else 0
                last_po = df_hist.iloc[-1]['PO'] if hist_data else 0
                last_sales = df_hist.iloc[-1]['Sales'] if hist_data else 0
                
                stats_data = {
                    "Metric": ["Forecast (Last Mo)", "PO (Last Mo)", "Sales (Last Mo)", "Bias (Last Mo)"],
                    "Value": [
                        f"{last_fcst:,.0f}", 
                        f"{last_po:,.0f}", 
                        f"{last_sales:,.0f}",
                        f"{last_po - last_fcst:+,.0f}"
                    ]
                }
                st.table(pd.DataFrame(stats_data))

    else:
        st.info("👋 Please ensure Sales and Monthly Performance data are loaded to view SKU insights.")

# --- TAB 5: SALES & FORECAST ANALYSIS (EASY TO UNDERSTAND VERSION) ---
with tab5:
    st.subheader("📈 Realization & Gap Analysis")
    st.caption("Membandingkan Perencanaan (Rofo), Eksekusi (PO), dan Hasil Akhir (Sales) secara dinamis.")

    if not df_sales.empty and not df_forecast.empty:
        # ==============================================================================
        # 1. DATA PREPARATION & DYNAMIC FILTERS (YEAR & BRAND)
        # ==============================================================================
        
        # Get all unique months & brands from the datasets
        all_months = sorted(list(set(df_sales['Month'].unique()) | set(df_forecast['Month'].unique()) | set(df_po['Month'].unique())))
        available_years = sorted(list(set([m.year for m in all_months if pd.notnull(m)])))
        
        # Mengambil daftar brand yang tersedia (hapus nilai kosong)
        all_brands = set()
        if 'Brand' in df_forecast.columns: all_brands.update(df_forecast['Brand'].dropna().unique())
        if 'Brand' in df_sales.columns: all_brands.update(df_sales['Brand'].dropna().unique())
        available_brands = sorted(list(all_brands))

        # UI Filter Multi-Select untuk Tahun & Brand (Bersebelahan)
        col_fil1, col_fil2 = st.columns(2)
        with col_fil1:
            selected_years = st.multiselect(
                "📅 Filter Tahun:",
                options=available_years,
                default=available_years,
                help="Pilih tahun untuk menganalisis performa pada periode tertentu."
            )
        with col_fil2:
            selected_brands = st.multiselect(
                "🏷️ Filter Brand:",
                options=available_brands,
                default=available_brands, # Default: tampilkan semua brand
                help="Pilih spesifik brand untuk membedah tren performanya."
            )
        
        # Filter list bulan berdasarkan tahun yang dipilih
        filtered_months = [m for m in all_months if m.year in selected_years]

        # Cegah error jika user menghapus semua pilihan
        if not filtered_months or not selected_brands:
            st.warning("⚠️ Silakan pilih minimal 1 Tahun dan 1 Brand untuk menampilkan data.")
        else:
            # --- Buat Dataframe Terfilter Global (Bulan & Brand) untuk Tab 5 ---
            df_f_filtered = df_forecast[(df_forecast['Month'].isin(filtered_months)) & (df_forecast['Brand'].isin(selected_brands))]
            df_p_filtered = df_po[(df_po['Month'].isin(filtered_months)) & (df_po['Brand'].isin(selected_brands))]
            df_s_filtered = df_sales[(df_sales['Month'].isin(filtered_months)) & (df_sales['Brand'].isin(selected_brands))]
            
            monthly_data = []
            for month in filtered_months:
                # Sales, Forecast, PO per bulan (sudah terfilter by brand)
                s_qty = df_s_filtered[df_s_filtered['Month'] == month]['Sales_Qty'].sum()
                f_qty = df_f_filtered[df_f_filtered['Month'] == month]['Forecast_Qty'].sum()
                p_qty = df_p_filtered[df_p_filtered['Month'] == month]['PO_Qty'].sum()
                
                # Hanya masukkan ke grafik jika bulan tersebut ada datanya (tidak nol semua)
                if s_qty > 0 or f_qty > 0 or p_qty > 0:
                    monthly_data.append({
                        'Month': month,
                        'Month_Txt': month.strftime('%b-%y'),
                        'Rofo': f_qty,
                        'Sales': s_qty,
                        'PO': p_qty,
                        'Gap_Sales_Rofo': s_qty - f_qty
                    })
                
            df_trend = pd.DataFrame(monthly_data)
            
            # Totals for KPI (Menggunakan data terfilter)
            total_rofo = df_trend['Rofo'].sum() if not df_trend.empty else 0
            total_sales = df_trend['Sales'].sum() if not df_trend.empty else 0
            total_po = df_trend['PO'].sum() if not df_trend.empty else 0
            
            # ==============================================================================
            # 2. KPI CARDS (PASTEL GRADIENT)
            # ==============================================================================
            st.markdown("""
            <style>
                .kpi-box {
                    border-radius: 12px; padding: 1.2rem; color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.05); position: relative;
                    transition: transform 0.3s;
                }
                .kpi-box:hover { transform: translateY(-3px); }
                .kpi-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; margin-bottom: 5px; }
                .kpi-num { font-size: 1.8rem; font-weight: 800; margin-bottom: 0px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
                .kpi-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; }
            </style>
            """, unsafe_allow_html=True)

            def render_kpi(title, val, sub, gradient):
                return f"""
                <div class="kpi-box" style="background: {gradient};">
                    <div class="kpi-title">{title}</div>
                    <div class="kpi-num">{val}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>
                """

            c1, c2, c3, c4 = st.columns(4)
            
            with c1:
                st.markdown(render_kpi("1. PLAN (ROFO)", f"{total_rofo:,.0f}", "Total Forecast", 
                    "linear-gradient(135deg, #7986cb 0%, #5c6bc0 100%)"), unsafe_allow_html=True)
            with c2:
                po_vs_rofo = (total_po / total_rofo * 100) if total_rofo > 0 else 0
                st.markdown(render_kpi("2. EXECUTION (PO)", f"{total_po:,.0f}", f"{po_vs_rofo:.1f}% of Plan", 
                    "linear-gradient(135deg, #ffb74d 0%, #ffa726 100%)"), unsafe_allow_html=True)
            with c3:
                sales_vs_rofo = (total_sales / total_rofo * 100) if total_rofo > 0 else 0
                st.markdown(render_kpi("3. RESULT (SALES)", f"{total_sales:,.0f}", f"{sales_vs_rofo:.1f}% Achievement", 
                    "linear-gradient(135deg, #4db6ac 0%, #26a69a 100%)"), unsafe_allow_html=True)
            with c4:
                gap = total_sales - total_rofo
                gap_col = "linear-gradient(135deg, #ef5350 0%, #e53935 100%)" if gap < 0 else "linear-gradient(135deg, #66bb6a 0%, #43a047 100%)"
                st.markdown(render_kpi("GAP (SALES vs PLAN)", f"{gap:+,.0f}", "Units Variance", gap_col), unsafe_allow_html=True)

            # ==============================================================================
            # 3. MAIN COMPARISON CHART (GROUPED BAR)
            # ==============================================================================
            st.divider()
            st.subheader("📊 Performance Triad: Plan(Rofo) vs Exec(PO) vs Result(Sales)")
            st.caption("Grafik membandingkan posisi Rencana (Rofo), Pembelian (PO), dan Penjualan (Sales). **Angka persentase pada batang hijau (Sales) menunjukkan pencapaian terhadap PO (Sell-Through).**")

            fig_main = go.Figure()

            # Rofo (Plan) - Garis Putus-putus
            fig_main.add_trace(go.Scatter(
                x=df_trend['Month_Txt'], y=df_trend['Rofo'],
                name='Plan (Rofo)',
                mode='lines+markers',
                line=dict(color='#3949AB', width=3, dash='dash'), 
                marker=dict(size=8, color='#3949AB')
            ))

            # PO (Execution)
            fig_main.add_trace(go.Bar(
                x=df_trend['Month_Txt'], y=df_trend['PO'],
                name='Execution (PO)',
                marker_color='#FFB74D', 
                text=[f"{x:,.0f}" for x in df_trend['PO']],
                textposition='auto',
                textfont=dict(size=11)
            ))

            # Sales (Result) - DITAMBAHKAN PERSENTASE VS PO
            sales_text = []
            for s, p in zip(df_trend['Sales'], df_trend['PO']):
                if p > 0:
                    pct = (s / p) * 100
                    sales_text.append(f"{s:,.0f}<br>({pct:.1f}%)") # Memunculkan Qty dan % di bawahnya
                elif s > 0:
                    sales_text.append(f"{s:,.0f}<br>(No PO)")
                else:
                    sales_text.append("0")

            fig_main.add_trace(go.Bar(
                x=df_trend['Month_Txt'], y=df_trend['Sales'],
                name='Result (Sales)',
                marker_color='#4DB6AC', 
                text=sales_text,
                textposition='auto',
                textfont=dict(size=11, weight='bold') # Dipertegas agar mudah dibaca
            ))

            fig_main.update_layout(
                height=480, # Sedikit ditinggikan agar label teks bersusun (2 baris) tidak terpotong
                xaxis_title="Month",
                yaxis_title="Quantity (Units)",
                barmode='group', 
                hovermode="x unified",
                legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                plot_bgcolor='white',
                margin=dict(t=50, b=20, l=20, r=20)
            )
            st.plotly_chart(fig_main, use_container_width=True)

            # ==============================================================================
            # 3.5 [NEW] BRAND BREAKDOWN ANALYSIS (BULLET CHART & PROGRESS MATRIX)
            # ==============================================================================
            st.divider()
            st.subheader("🏢 Performance Breakdown by Brand")
            st.caption("Membandingkan langsung Hasil (Sales) dengan Target (Rofo) dan Eksekusi (PO). Garis biru adalah batas Target.")

            # Group data by Brand dari dataframe yang sudah terfilter tahunnya
            brand_f = df_f_filtered.groupby('Brand')['Forecast_Qty'].sum().reset_index()
            brand_p = df_p_filtered.groupby('Brand')['PO_Qty'].sum().reset_index()
            brand_s = df_s_filtered.groupby('Brand')['Sales_Qty'].sum().reset_index()
            
            # Merge All
            df_brand = pd.merge(brand_f, brand_p, on='Brand', how='outer')
            df_brand = pd.merge(df_brand, brand_s, on='Brand', how='outer').fillna(0)
            
            # Hitung Persentase Pencapaian untuk tabel & warna
            df_brand['Achievement (%)'] = np.where(
                df_brand['Forecast_Qty'] > 0, 
                (df_brand['Sales_Qty'] / df_brand['Forecast_Qty'] * 100), 
                0
            )

            # Buat 2 Tab: Chart View & Table View agar rapi
            tab_brand_chart, tab_brand_table = st.tabs(["📊 Bullet Chart View", "📋 Progress Matrix View"])

            with tab_brand_chart:
                # Urutkan Ascending karena Plotly horizontal bar menggambar dari bawah ke atas
                df_brand_chart = df_brand.sort_values('Forecast_Qty', ascending=True)

                # Bullet Chart Style (Bar + Marker)
                fig_brand = go.Figure()
                
                # 1. Bar utama untuk Sales (Result)
                fig_brand.add_trace(go.Bar(
                    y=df_brand_chart['Brand'], 
                    x=df_brand_chart['Sales_Qty'],
                    name='Result (Sales)', 
                    orientation='h',
                    marker_color='#4DB6AC', # Soft Teal
                    opacity=0.85,
                    hovertemplate="<b>%{y}</b><br>Sales: %{x:,.0f} units<extra></extra>"
                ))
                
                # 2. Garis Target untuk Rofo (Plan)
                fig_brand.add_trace(go.Scatter(
                    y=df_brand_chart['Brand'], 
                    x=df_brand_chart['Forecast_Qty'],
                    name='Plan (Rofo Target)', 
                    mode='markers',
                    marker=dict(symbol='line-ns-open', size=20, color='#3949AB', line=dict(width=4, color='#3949AB')),
                    hovertemplate="<b>%{y}</b><br>Target Rofo: %{x:,.0f} units<extra></extra>"
                ))
                
                # 3. Titik (Dot) untuk PO (Execution)
                fig_brand.add_trace(go.Scatter(
                    y=df_brand_chart['Brand'], 
                    x=df_brand_chart['PO_Qty'],
                    name='Execution (PO)', 
                    mode='markers',
                    marker=dict(symbol='circle', size=10, color='#FFB74D', line=dict(width=1, color='white')),
                    hovertemplate="<b>%{y}</b><br>PO: %{x:,.0f} units<extra></extra>"
                ))
                
                fig_brand.update_layout(
                    height=500,
                    barmode='overlay', # Agar marker menimpa bar
                    xaxis_title="Quantity (Units)",
                    yaxis_title="",
                    hovermode="closest",
                    legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
                    plot_bgcolor='white',
                    margin=dict(t=30, b=20, l=20, r=20)
                )
                fig_brand.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
                
                st.plotly_chart(fig_brand, use_container_width=True)
                st.caption("💡 **Cara baca:** Batang hijau adalah Sales. Jika batang menyentuh/melewati garis vertikal biru (Rofo), artinya capai target. Titik oranye adalah PO yang diturunkan.")

            with tab_brand_table:
                # Urutkan Descending untuk tabel (Top Brand di atas)
                df_brand_table = df_brand.sort_values('Forecast_Qty', ascending=False)
                
                # Gunakan st.dataframe dengan column_config untuk visualisasi in-line
                st.dataframe(
                    df_brand_table,
                    column_order=("Brand", "Forecast_Qty", "PO_Qty", "Sales_Qty", "Achievement (%)"),
                    column_config={
                        "Brand": st.column_config.TextColumn("Brand Name", width="medium"),
                        "Forecast_Qty": st.column_config.NumberColumn("Plan (Rofo)", format="%d"),
                        "PO_Qty": st.column_config.NumberColumn("Exec (PO)", format="%d"),
                        "Sales_Qty": st.column_config.NumberColumn("Result (Sales)", format="%d"),
                        "Achievement (%)": st.column_config.ProgressColumn(
                            "Achievement %",
                            help="Sales vs Forecast Percentage",
                            format="%.1f%%",
                            min_value=0,
                            max_value=120, # Cap max visual progress di 120%
                        ),
                    },
                    use_container_width=True,
                    hide_index=True,
                    height=450
                )

            # ==============================================================================
            # 4. TOP GAP ANALYSIS (SKU LEVEL)
            # ==============================================================================
            st.divider()
            st.subheader("🚨 Top Gap Analysis (SKU Level)")
            st.caption(f"Daftar barang dengan selisih terbesar antara Forecast vs Realisasi Sales untuk Tahun {', '.join(map(str, selected_years))}.")

            # Data processing untuk Gap per SKU (menggunakan data terfilter)
            df_f_sku = df_f_filtered.groupby(['SKU_ID', 'Product_Name'])['Forecast_Qty'].sum().reset_index()
            df_s_sku = df_s_filtered.groupby(['SKU_ID', 'Product_Name'])['Sales_Qty'].sum().reset_index()
            
            df_gap = pd.merge(df_f_sku, df_s_sku, on=['SKU_ID', 'Product_Name'], how='outer').fillna(0)
            df_gap['Gap'] = df_gap['Sales_Qty'] - df_gap['Forecast_Qty']
            
            # Pisahkan menjadi dua kelompok
            top_spikes = df_gap[df_gap['Gap'] > 0].sort_values('Gap', ascending=False).head(10)
            top_drops = df_gap[df_gap['Gap'] < 0].sort_values('Gap', ascending=True).head(10)

            c_gap1, c_gap2 = st.columns(2)

            with c_gap1:
                st.markdown("##### 🚀 Top Unexpected Demand (Sales > Rofo)")
                st.caption("Barang ini **LAKU KERAS** melebihi prediksi. Cek stok, Hati hati OOS!")
                
                fig_spike = go.Figure()
                fig_spike.add_trace(go.Bar(
                    y=top_spikes['Product_Name'].str[:20], # Truncate nama biar rapi
                    x=top_spikes['Gap'],
                    orientation='h',
                    marker_color='#66BB6A', # Green
                    text=[f"+{x:,.0f}" for x in top_spikes['Gap']],
                    textposition='auto',
                    name='Extra Sales'
                ))
                fig_spike.update_layout(
                    height=400,
                    xaxis_title="Extra Units Sold vs Plan",
                    yaxis=dict(autorange="reversed"), # Urutan dari atas ke bawah
                    plot_bgcolor='white',
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_spike, use_container_width=True)

            with c_gap2:
                st.markdown("##### 🐌 Top Slow Moving vs Plan (Sales < Rofo)")
                st.caption("Barang ini **Over Estimated** dibanding prediksi. Hati-hati overstock!!")
                
                fig_drop = go.Figure()
                fig_drop.add_trace(go.Bar(
                    y=top_drops['Product_Name'].str[:20],
                    x=top_drops['Gap'], # Nilai negatif
                    orientation='h',
                    marker_color='#EF5350', # Red
                    text=[f"{x:,.0f}" for x in top_drops['Gap']],
                    textposition='auto',
                    name='Missed Sales'
                ))
                fig_drop.update_layout(
                    height=400,
                    xaxis_title="Missed Units vs Plan",
                    yaxis=dict(autorange="reversed", side="right"), # Label di kanan biar tidak tabrakan
                    plot_bgcolor='white',
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                st.plotly_chart(fig_drop, use_container_width=True)

    else:
        st.info("ℹ️ Membutuhkan data Sales dan Forecast untuk menampilkan analisis.")

# --- TAB 6: DATA EXPLORER ---
with tab6:
    st.subheader("📋 Raw Data Explorer")
    
    dataset_options = {
        "Product Master": df_product,
        "Active Products": df_product_active,
        "Sales Data": df_sales,
        "Forecast Data": df_forecast,
        "PO Data": df_po,
        "Stock Data": df_stock,
        "Financial Data": df_financial,
        "Inventory Financial": df_inventory_financial
    }
    
    selected_dataset = st.selectbox("Select Dataset", list(dataset_options.keys()))
    df_selected = dataset_options[selected_dataset]
    
    if not df_selected.empty:
        # Ensure Product_Name is shown alongside SKU_ID if available
        if 'SKU_ID' in df_selected.columns and 'Product_Name' in df_selected.columns:
            # Reorder columns to show SKU_ID and Product_Name first
            cols = list(df_selected.columns)
            if 'Product_Name' in cols:
                cols.remove('Product_Name')
                cols.insert(1, 'Product_Name')
            df_selected = df_selected[cols]
        
        # Data info
        st.write(f"**Rows:** {df_selected.shape[0]:,} | **Columns:** {df_selected.shape[1]}")
        
        # Column selector
        if st.checkbox("Select Columns", False):
            all_columns = df_selected.columns.tolist()
            selected_columns = st.multiselect("Choose columns:", all_columns, default=all_columns[:10])
            df_display = df_selected[selected_columns]
        else:
            df_display = df_selected
        
        # Data preview
        st.dataframe(
            df_display,
            use_container_width=True,
            height=500
        )
        
        # Download option
        csv = df_selected.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{selected_dataset.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("No data available for selected dataset")

# --- TAB 7: ECOMMERCE FORECAST INTELLIGENCE (COMPLETE WITH QUARTERLY & EXPLORER) ---
with tab7:
    st.subheader("🔮 Ecommerce Forecast Intelligence")
    st.caption("Future Planning: Seasonality Analysis, Quarterly Strategy, Scenario Testing & Data Explorer")

    # ==============================================================================
    # 1. DATA PREPARATION (ROBUST MONTH PARSING)
    # ==============================================================================
    if not df_ecomm_forecast.empty:
        # Coba deteksi kolom bulan forecast
        id_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Status', 'Floor_Price', 'Net_Order_Price']
        forecast_cols = [c for c in df_ecomm_forecast.columns if c not in id_cols]
        
        # Validasi: Kolom harus mengandung angka
        valid_fcst_cols = []
        for c in forecast_cols:
            try:
                if df_ecomm_forecast[c].dtype == object:
                    sample = df_ecomm_forecast[c].iloc[0]
                    if isinstance(sample, str) and any(i.isdigit() for i in sample):
                        valid_fcst_cols.append(c)
                elif np.issubdtype(df_ecomm_forecast[c].dtype, np.number):
                    valid_fcst_cols.append(c)
            except:
                pass
        
        # Mapping tanggal
        col_date_map = []
        if valid_fcst_cols:
            for c in valid_fcst_cols:
                clean_c = str(c).strip()
                for fmt in ['%b-%y', '%b %y', '%b-%Y', '%B %Y', '%Y-%m', '%b_%y']:
                    try:
                        dt = datetime.strptime(clean_c, fmt)
                        col_date_map.append({'col': c, 'date': dt})
                        break
                    except:
                        continue
        
        if not col_date_map:
            st.warning("⚠️ Tidak dapat membaca kolom bulan secara otomatis.")
            st.stop()
        else:
            # Sort kolom secara kronologis
            col_date_map.sort(key=lambda x: x['date'])
            sorted_fcst_cols = [x['col'] for x in col_date_map]
            
            # Update dataframe active
            df_fcst_active = df_ecomm_forecast.copy()
            for c in sorted_fcst_cols:
                df_fcst_active[c] = pd.to_numeric(df_fcst_active[c], errors='coerce').fillna(0)

            # Merge product info if missing
            if 'Floor_Price' not in df_fcst_active.columns:
                df_fcst_active = add_product_info_to_data(df_fcst_active, df_product)

            # ==============================================================================
            # 2. FORECAST HEALTH KPI
            # ==============================================================================
            total_fcst_qty = df_fcst_active[sorted_fcst_cols].sum().sum()
            
            # Value Calculation
            total_fcst_val = 0
            has_price = False
            if 'Floor_Price' in df_fcst_active.columns:
                df_fcst_active['Floor_Price'] = pd.to_numeric(df_fcst_active['Floor_Price'], errors='coerce').fillna(0)
                row_sums = df_fcst_active[sorted_fcst_cols].sum(axis=1)
                total_fcst_val = (row_sums * df_fcst_active['Floor_Price']).sum()
                has_price = True

            # CSS
            st.markdown("""
            <style>
                .fcst-card {
                    border-radius: 12px; padding: 1.2rem; color: white;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.05); transition: transform 0.3s;
                }
                .fcst-card:hover { transform: translateY(-3px); }
                .fcst-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; margin-bottom: 5px; }
                .fcst-val { font-size: 1.8rem; font-weight: 800; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
                .fcst-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; }
            </style>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="fcst-card" style="background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%);"><div class="fcst-title">Total Forecast Volume</div><div class="fcst-val">{total_fcst_qty:,.0f}</div><div class="fcst-sub">Units</div></div>""", unsafe_allow_html=True)
            with c2:
                val_text = f"Rp {total_fcst_val/1e9:,.1f} M" if has_price else "N/A"
                st.markdown(f"""<div class="fcst-card" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);"><div class="fcst-title">Total Forecast Value</div><div class="fcst-val">{val_text}</div><div class="fcst-sub">Gross Revenue</div></div>""", unsafe_allow_html=True)
            with c3:
                peak_month = df_fcst_active[sorted_fcst_cols].sum().idxmax()
                st.markdown(f"""<div class="fcst-card" style="background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);"><div class="fcst-title">Peak Season</div><div class="fcst-val">{str(peak_month).split('.')[0]}</div><div class="fcst-sub">Highest Volume Month</div></div>""", unsafe_allow_html=True)

            # ==============================================================================
            # 3. SEASONALITY CHART
            # ==============================================================================
            st.divider()
            st.subheader("📈 Monthly Forecast Trend")
            monthly_agg = df_fcst_active[sorted_fcst_cols].sum().reset_index()
            monthly_agg.columns = ['Month', 'Qty']
            
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Bar(x=monthly_agg['Month'], y=monthly_agg['Qty'], name='Monthly Forecast', marker_color='#818CF8'))
            fig_trend.add_trace(go.Scatter(x=monthly_agg['Month'], y=[monthly_agg['Qty'].mean()]*len(monthly_agg), name='Average', mode='lines', line=dict(color='#F59E0B', width=2, dash='dash')))
            fig_trend.update_layout(height=350, hovermode="x unified", plot_bgcolor='white', margin=dict(t=20, b=20))
            st.plotly_chart(fig_trend, use_container_width=True)

            # ==============================================================================
            # 4. QUARTERLY BRAND ANALYSIS (NEW FEATURE)
            # ==============================================================================
            st.divider()
            st.subheader("📅 Quarterly Brand Analysis")
            st.caption("Aggregated performance by Quarter (Q1-Q4) to identify strategic periods.")

            # Logic Quarter
            q_map = {'Q1': ['jan', 'feb', 'mar'], 'Q2': ['apr', 'may', 'jun'], 
                     'Q3': ['jul', 'aug', 'sep'], 'Q4': ['oct', 'nov', 'dec']}
            
            quarter_cols_map = {'Q1': [], 'Q2': [], 'Q3': [], 'Q4': []}
            
            for col in sorted_fcst_cols:
                m_str = str(col).lower()[:3]
                for q, months in q_map.items():
                    if m_str in months:
                        quarter_cols_map[q].append(col)
            
            active_quarters = [q for q, cols in quarter_cols_map.items() if len(cols) > 0]

            if active_quarters and 'Brand' in df_fcst_active.columns:
                q_tab1, q_tab2 = st.tabs(["📦 By Quantity (Heatmap)", "💰 By Value (Heatmap)"])
                
                # --- QTY HEATMAP ---
                with q_tab1:
                    q_brand_qty = []
                    for brand in df_fcst_active['Brand'].unique():
                        brand_data = df_fcst_active[df_fcst_active['Brand'] == brand]
                        row = {'Brand': brand}
                        total_row = 0
                        for q in active_quarters:
                            q_val = brand_data[quarter_cols_map[q]].sum().sum()
                            row[q] = q_val
                            total_row += q_val
                        row['Total'] = total_row
                        q_brand_qty.append(row)
                    
                    df_all_qty = pd.DataFrame(q_brand_qty)
                    df_q_qty = df_all_qty.sort_values('Total', ascending=False).head(15) # Top 15 Brands
                    
                    # Hitung Grand Total (dari SEMUA brand, bukan hanya top 15)
                    grand_total_qty = {'Brand': 'TOTAL (ALL BRANDS)'}
                    for col in active_quarters + ['Total']:
                        grand_total_qty[col] = df_all_qty[col].sum()
                    
                    # Gabungkan baris Total ke dalam dataframe display
                    df_q_qty = pd.concat([df_q_qty, pd.DataFrame([grand_total_qty])], ignore_index=True)
                    
                    # Tambahkan 'Total' ke kolom yang akan didisplay di Heatmap
                    display_cols = active_quarters + ['Total']
                    
                    fig_heat_qty = go.Figure(data=go.Heatmap(
                        z=df_q_qty[display_cols].values,
                        x=display_cols,
                        y=df_q_qty['Brand'],
                        colorscale='Blues',
                        text=df_q_qty[display_cols].values,
                        texttemplate="%{text:,.0f}"
                    ))
                    
                    fig_heat_qty.update_layout(
                        height=550, 
                        title="Top 15 Brands - Quarterly Volume",
                        yaxis=dict(autorange="reversed") # Balik Y-axis agar Top 1 di atas & Total di bawah
                    )
                    st.plotly_chart(fig_heat_qty, use_container_width=True)

                # --- VALUE HEATMAP ---
                with q_tab2:
                    if has_price:
                        q_brand_val = []
                        # Pre-calc temp price column
                        df_fcst_active['Temp_Price'] = df_fcst_active['Floor_Price'].fillna(0)
                        
                        for brand in df_fcst_active['Brand'].unique():
                            brand_data = df_fcst_active[df_fcst_active['Brand'] == brand]
                            row = {'Brand': brand}
                            total_row = 0
                            for q in active_quarters:
                                cols = quarter_cols_map[q]
                                # Vectorized multiplication
                                q_val = 0
                                for c in cols:
                                    q_val += (brand_data[c] * brand_data['Temp_Price']).sum()
                                row[q] = q_val
                                total_row += q_val
                            row['Total'] = total_row
                            q_brand_val.append(row)
                        
                        df_all_val = pd.DataFrame(q_brand_val)
                        df_q_val = df_all_val.sort_values('Total', ascending=False).head(15)
                        
                        # Hitung Grand Total Value (dari SEMUA brand)
                        grand_total_val = {'Brand': 'TOTAL (ALL BRANDS)'}
                        for col in active_quarters + ['Total']:
                            grand_total_val[col] = df_all_val[col].sum()
                            
                        # Gabungkan baris Total ke dalam dataframe display
                        df_q_val = pd.concat([df_q_val, pd.DataFrame([grand_total_val])], ignore_index=True)
                        
                        # Tambahkan 'Total' ke kolom yang akan didisplay di Heatmap
                        display_cols = active_quarters + ['Total']
                        
                        fig_heat_val = go.Figure(data=go.Heatmap(
                            z=df_q_val[display_cols].values,
                            x=display_cols,
                            y=df_q_val['Brand'],
                            colorscale='Greens',
                            text=df_q_val[display_cols].values,
                            texttemplate="Rp %{text:,.0f}" # Full number format
                        ))
                        
                        fig_heat_val.update_layout(
                            height=550, 
                            title="Top 15 Brands - Quarterly Revenue Projection",
                            yaxis=dict(autorange="reversed") # Balik Y-axis agar Top 1 di atas & Total di bawah
                        )
                        st.plotly_chart(fig_heat_val, use_container_width=True)
                    else:
                        st.warning("⚠️ Data Harga (Floor_Price) tidak ditemukan.")

            # ==============================================================================
            # 5. SCENARIO PLANNER & ANOMALY (KEPT FROM PREV)
            # ==============================================================================
            st.divider()
            c_scen, c_anom = st.columns(2)
            
            with c_scen:
                st.subheader("🎮 Quick Scenario")
                growth_pct = st.slider("Growth Adjustment (%)", -50, 50, 0, 5)
                
                # Simple Scenario Calc
                monthly_base = df_fcst_active[sorted_fcst_cols].sum()
                monthly_scen = monthly_base * (1 + growth_pct/100)
                
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(x=sorted_fcst_cols, y=monthly_base, name="Baseline", fill='tozeroy'))
                fig_s.add_trace(go.Scatter(x=sorted_fcst_cols, y=monthly_scen, name=f"Scenario {growth_pct:+}%", line=dict(dash='dot')))
                fig_s.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                st.plotly_chart(fig_s, use_container_width=True)
            
            with c_anom:
                st.subheader("🚨 Spike Detection")
                # Top 5 Extreme Spikes
                anomalies = []
                for idx, row in df_fcst_active.sort_values(sorted_fcst_cols[-1], ascending=False).head(200).iterrows():
                    vals = row[sorted_fcst_cols].values
                    if np.mean(vals) > 10 and np.max(vals) > 3 * np.mean(vals):
                        anomalies.append({
                            'Product': row.get('Product_Name', row['SKU_ID']),
                            'Spike_Month': sorted_fcst_cols[np.argmax(vals)],
                            'Spike_Qty': np.max(vals)
                        })
                if anomalies:
                    st.dataframe(pd.DataFrame(anomalies).head(5), height=300, use_container_width=True)
                else:
                    st.success("✅ No extreme spikes detected.")

            # ==============================================================================
            # 6. DATA EXPLORER (NEW FEATURE)
            # ==============================================================================
            st.divider()
            st.subheader("📋 Forecast Data Explorer")
            st.caption("Drill-down ke level SKU per bulan.")

            with st.container():
                # Filter UI
                col_f1, col_f2 = st.columns([1, 2])
                
                with col_f1:
                    # Filter Brand
                    all_brands = df_fcst_active['Brand'].unique().tolist() if 'Brand' in df_fcst_active.columns else []
                    sel_brands = st.multiselect("Filter Brands:", options=all_brands, default=[])
                    
                    # Filter Months to Show
                    months_to_show = st.slider("Jumlah Bulan Ditampilkan:", min_value=3, max_value=len(sorted_fcst_cols), value=6)
                
                with col_f2:
                    # Search SKU
                    search_txt = st.text_input("Search SKU Name / ID:", placeholder="Ketik nama produk...")

                # Apply Filter
                df_exp = df_fcst_active.copy()
                if sel_brands:
                    df_exp = df_exp[df_exp['Brand'].isin(sel_brands)]
                if search_txt:
                    df_exp = df_exp[
                        df_exp['SKU_ID'].astype(str).str.contains(search_txt, case=False) | 
                        df_exp['Product_Name'].astype(str).str.contains(search_txt, case=False)
                    ]
                
                # Column Selection
                # Basic info cols + selected months
                cols_display = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier']
                month_cols_display = sorted_fcst_cols[:months_to_show] # Show first X months or last? Usually first few months are priority.
                
                final_cols = [c for c in cols_display if c in df_exp.columns] + month_cols_display
                
                # Display Dataframe
                st.dataframe(
                    df_exp[final_cols], 
                    use_container_width=True, 
                    height=500,
                    column_config={
                        c: st.column_config.NumberColumn(format="%.0f") for c in month_cols_display
                    }
                )
                
                # Download Button
                csv = df_exp.to_csv(index=False)
                st.download_button(
                    label="📥 Download Filtered Data (CSV)",
                    data=csv,
                    file_name="forecast_data_filtered.csv",
                    mime="text/csv"
                )

    else:
        st.info("ℹ️ Silakan upload data forecast di sheet 'Forecast_2026_Ecomm' terlebih dahulu.")

# --- TAB 8: PROFITABILITY & MARGIN ANALYSIS (WITH TIER ANALYSIS) ---
with tab8:
    st.subheader("💰 Profitability & Margin Intelligence")
    st.caption("Financial Projection 2026: Revenue, Cost of Goods Sold (COGS), and Gross Margin Analysis")

    # ==============================================================================
    # 1. DATA MERGING & PREPARATION (ECOMM + RESELLER)
    # ==============================================================================
    combined_data = []
    
    # A. Process Ecommerce
    if not df_ecomm_forecast.empty:
        # Detect numeric columns (months)
        fcst_cols = [c for c in df_ecomm_forecast.columns if any(char.isdigit() for char in str(c))]
        if fcst_cols:
            df_e = df_ecomm_forecast.melt(id_vars=['SKU_ID'], value_vars=fcst_cols, var_name='Month_Label', value_name='Qty')
            df_e['Channel'] = 'Ecommerce'
            combined_data.append(df_e)

    # B. Process Reseller
    if not df_reseller_forecast.empty:
        fcst_cols_res = [c for c in df_reseller_forecast.columns if any(char.isdigit() for char in str(c))]
        if fcst_cols_res:
            df_r = df_reseller_forecast.melt(id_vars=['SKU_ID'], value_vars=fcst_cols_res, var_name='Month_Label', value_name='Qty')
            df_r['Channel'] = 'Reseller'
            combined_data.append(df_r)

    if combined_data:
        # Combine
        df_fin = pd.concat(combined_data, ignore_index=True)
        df_fin['Qty'] = pd.to_numeric(df_fin['Qty'], errors='coerce').fillna(0)
        df_fin = df_fin[df_fin['Qty'] > 0] # Filter non-zero

        # Merge with Product Master for Prices
        # Need: Floor_Price (Revenue) and Net_Order_Price (COGS/HPP)
        if not df_product.empty:
            cols_price = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Floor_Price', 'Net_Order_Price']
            existing_cols = [c for c in cols_price if c in df_product.columns]
            df_fin = pd.merge(df_fin, df_product[existing_cols], on='SKU_ID', how='left')
            
            # Fill NaNs
            if 'Floor_Price' in df_fin.columns: df_fin['Floor_Price'] = pd.to_numeric(df_fin['Floor_Price'], errors='coerce').fillna(0)
            if 'Net_Order_Price' in df_fin.columns: df_fin['Net_Order_Price'] = pd.to_numeric(df_fin['Net_Order_Price'], errors='coerce').fillna(0)
            
            # CALCULATE FINANCIALS
            df_fin['Revenue'] = df_fin['Qty'] * df_fin['Floor_Price']
            df_fin['COGS'] = df_fin['Qty'] * df_fin['Net_Order_Price']
            df_fin['Gross_Margin'] = df_fin['Revenue'] - df_fin['COGS']
            
            # ==============================================================================
            # 2. FINANCIAL HEALTH CARDS (PASTEL STYLE)
            # ==============================================================================
            total_rev = df_fin['Revenue'].sum()
            total_cogs = df_fin['COGS'].sum()
            total_margin = df_fin['Gross_Margin'].sum()
            margin_pct = (total_margin / total_rev * 100) if total_rev > 0 else 0
            
            # CSS
            st.markdown("""
            <style>
                .fin-card {
                    border-radius: 12px; padding: 1.2rem; color: white;
                    box-shadow: 0 4px 10px rgba(0,0,0,0.05); transition: transform 0.3s;
                }
                .fin-card:hover { transform: translateY(-3px); }
                .fin-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; opacity: 0.9; margin-bottom: 5px; }
                .fin-val { font-size: 1.5rem; font-weight: 800; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
                .fin-sub { font-size: 0.85rem; font-weight: 500; opacity: 0.95; }
            </style>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            
            # Helper Format
            def fmt_money(x): 
                if x >= 1e9: return f"Rp {x/1e9:,.1f} M"
                elif x >= 1e6: return f"Rp {x/1e6:,.1f} Jt"
                return f"Rp {x:,.0f}"

            with c1:
                st.markdown(f"""<div class="fin-card" style="background: linear-gradient(135deg, #6366F1 0%, #4338CA 100%);"><div class="fin-title">Total Revenue</div><div class="fin-val">{fmt_money(total_rev)}</div><div class="fin-sub">Gross Sales</div></div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="fin-card" style="background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);"><div class="fin-title">Total COGS (HPP)</div><div class="fin-val">{fmt_money(total_cogs)}</div><div class="fin-sub">Cost of Goods</div></div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="fin-card" style="background: linear-gradient(135deg, #10B981 0%, #059669 100%);"><div class="fin-title">Gross Margin (Cuan)</div><div class="fin-val">{fmt_money(total_margin)}</div><div class="fin-sub">Net Profit (Gross)</div></div>""", unsafe_allow_html=True)
            with c4:
                color_m = "#10B981" if margin_pct > 30 else "#EF4444"
                st.markdown(f"""<div class="fin-card" style="background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);"><div class="fin-title">Blended Margin %</div><div class="fin-val">{margin_pct:.1f}%</div><div class="fin-sub">Profitability Ratio</div></div>""", unsafe_allow_html=True)

            # ==============================================================================
            # 3. WATERFALL & CHANNEL MIX
            # ==============================================================================
            st.divider()
            c_water, c_mix = st.columns([2, 1])
            
            with c_water:
                st.subheader("🌊 Financial Waterfall")
                # Waterfall Chart
                fig_water = go.Figure(go.Waterfall(
                    name = "20", orientation = "v",
                    measure = ["relative", "relative", "total"],
                    x = ["Total Revenue", "COGS (Cost)", "Gross Margin"],
                    textposition = "outside",
                    text = [fmt_money(total_rev), fmt_money(-total_cogs), fmt_money(total_margin)],
                    y = [total_rev, -total_cogs, total_margin],
                    connector = {"line":{"color":"rgb(63, 63, 63)"}},
                    increasing = {"marker":{"color":"#6366F1"}}, # Revenue
                    decreasing = {"marker":{"color":"#F59E0B"}}, # Cost
                    totals = {"marker":{"color":"#10B981"}}      # Margin
                ))
                fig_water.update_layout(height=400, title="Profitability Flow: Revenue to Margin", showlegend=False)
                st.plotly_chart(fig_water, use_container_width=True)
                
            with c_mix:
                st.subheader("🏢 Margin by Channel")
                channel_data = df_fin.groupby('Channel')['Gross_Margin'].sum().reset_index()
                fig_don = px.pie(channel_data, values='Gross_Margin', names='Channel', hole=0.4, 
                                 color_discrete_sequence=['#10B981', '#3B82F6'], title="Profit Contribution")
                fig_don.update_layout(height=400, showlegend=True, legend=dict(orientation="h", y=-0.1))
                st.plotly_chart(fig_don, use_container_width=True)

            # ==============================================================================
            # 4. PROFITABILITY BY SKU TIER (NEW SECTION)
            # ==============================================================================
            if 'SKU_Tier' in df_fin.columns:
                st.divider()
                st.subheader("💎 Profitability by SKU Tier")
                st.caption("Analisis kontribusi profit berdasarkan segmen (Tier). Tier dengan volume besar belum tentu margin % nya besar.")

                # Calculate Tier Metrics
                tier_fin = df_fin.groupby('SKU_Tier').agg({
                    'Revenue': 'sum',
                    'Gross_Margin': 'sum',
                    'Qty': 'sum'
                }).reset_index()
                
                tier_fin['Margin_Pct'] = (tier_fin['Gross_Margin'] / tier_fin['Revenue'] * 100).fillna(0)
                tier_fin = tier_fin.sort_values('Gross_Margin', ascending=False)

                t1, t2 = st.columns(2)

                with t1:
                    # Bar Chart: Total Gross Margin (Rupiah)
                    fig_tier_val = go.Figure()
                    fig_tier_val.add_trace(go.Bar(
                        y=tier_fin['SKU_Tier'],
                        x=tier_fin['Gross_Margin'],
                        orientation='h',
                        text=[fmt_money(x) for x in tier_fin['Gross_Margin']],
                        textposition='auto',
                        marker_color='#6366F1', # Soft Indigo
                        name='Gross Margin (Rp)'
                    ))
                    fig_tier_val.update_layout(
                        height=400, 
                        title="💰 Total Gross Margin Contribution by Tier",
                        xaxis_title="Gross Margin (Rp)",
                        yaxis_title="Tier",
                        plot_bgcolor='white',
                        yaxis=dict(autorange="reversed") # Tier tertinggi di atas
                    )
                    st.plotly_chart(fig_tier_val, use_container_width=True)

                with t2:
                    # Bar Chart: Margin % (Efisiensi)
                    # Color logic: Green high margin, Red low margin
                    colors = []
                    for val in tier_fin['Margin_Pct']:
                        if val >= 40: colors.append('#10B981') # Green
                        elif val >= 20: colors.append('#F59E0B') # Orange
                        else: colors.append('#EF4444') # Red

                    fig_tier_pct = go.Figure()
                    fig_tier_pct.add_trace(go.Bar(
                        y=tier_fin['SKU_Tier'],
                        x=tier_fin['Margin_Pct'],
                        orientation='h',
                        text=[f"{x:.1f}%" for x in tier_fin['Margin_Pct']],
                        textposition='auto',
                        marker_color=colors,
                        name='Margin %'
                    ))
                    
                    # Add avg line
                    avg_margin_tier = tier_fin['Margin_Pct'].mean()
                    fig_tier_pct.add_vline(x=avg_margin_tier, line_dash="dash", line_color="gray", annotation_text="Avg")

                    fig_tier_pct.update_layout(
                        height=400, 
                        title="📊 Efficiency: Margin % by Tier",
                        xaxis_title="Margin %",
                        yaxis_title="Tier",
                        plot_bgcolor='white',
                        yaxis=dict(autorange="reversed")
                    )
                    st.plotly_chart(fig_tier_pct, use_container_width=True)

            # ==============================================================================
            # 5. PROFITABILITY MATRIX (SCATTER PLOT)
            # ==============================================================================
            st.divider()
            st.subheader("🎯 Profitability Matrix (SKU Level)")
            st.caption("Analisis posisi SKU berdasarkan **Revenue (Volume)** vs **Margin % (Quality)**.")
            
            # Group by SKU
            sku_fin = df_fin.groupby(['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier']).agg({
                'Revenue': 'sum', 'Gross_Margin': 'sum'
            }).reset_index()
            sku_fin['Margin_Pct'] = (sku_fin['Gross_Margin'] / sku_fin['Revenue'] * 100).fillna(0)
            
            # Scatter Plot
            fig_scat = px.scatter(
                sku_fin, x='Revenue', y='Margin_Pct', size='Gross_Margin', color='Brand',
                hover_name='Product_Name', 
                hover_data=['SKU_Tier', 'Gross_Margin'],
                labels={'Revenue': 'Revenue (Rp)', 'Margin_Pct': 'Margin %'},
                size_max=50, title="Matrix: Revenue vs Margin % (Size = Total Margin Rp)"
            )
            
            # Add Average Lines
            avg_rev = sku_fin['Revenue'].mean()
            avg_mar = sku_fin['Margin_Pct'].mean()
            
            fig_scat.add_hline(y=avg_mar, line_dash="dash", line_color="gray", annotation_text="Avg Margin")
            fig_scat.add_vline(x=avg_rev, line_dash="dash", line_color="gray", annotation_text="Avg Revenue")
            
            # Quadrant Backgrounds
            max_x = sku_fin['Revenue'].max() * 1.1
            max_y = sku_fin['Margin_Pct'].max() * 1.1
            
            # High Rev / High Margin (Stars)
            fig_scat.add_shape(type="rect", x0=avg_rev, y0=avg_mar, x1=max_x, y1=max_y, fillcolor="rgba(16, 185, 129, 0.1)", layer="below", line_width=0)
            # High Rev / Low Margin (Cash Cows)
            fig_scat.add_shape(type="rect", x0=avg_rev, y0=0, x1=max_x, y1=avg_mar, fillcolor="rgba(245, 158, 11, 0.1)", layer="below", line_width=0)
            
            fig_scat.update_layout(height=500, plot_bgcolor='white', xaxis_type="log") 
            st.plotly_chart(fig_scat, use_container_width=True)
            
            st.info("""
            **Cara Baca Matrix:**
            - 🟩 **Kanan Atas (Stars):** Revenue Tinggi + Margin Tinggi. **Pertahankan Stok!**
            - 🟨 **Kanan Bawah (Cash Cows):** Revenue Tinggi + Margin Rendah. **Volume Maker**, hati-hati cost.
            - 🟦 **Kiri Atas (Niche):** Revenue Rendah + Margin Tinggi. **Produk Premium**.
            - ⬜ **Kiri Bawah (Dogs):** Revenue Rendah + Margin Rendah. **Evaluasi/Discontinue**.
            """)

            # ==============================================================================
            # 6. PARETO CUAN (80/20 RULE)
            # ==============================================================================
            st.divider()
            st.subheader("📉 Pareto Cuan: Top Profit Contributors")
            
            # Sort by Margin desc
            sku_pareto = sku_fin.sort_values('Gross_Margin', ascending=False)
            sku_pareto['Cum_Margin'] = sku_pareto['Gross_Margin'].cumsum()
            sku_pareto['Cum_Pct'] = sku_pareto['Cum_Margin'] / total_margin * 100
            
            # Visual Pareto
            top_30 = sku_pareto.head(30)
            
            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(x=top_30['Product_Name'].str[:20], y=top_30['Gross_Margin'], name='Gross Margin', marker_color='#10B981'))
            fig_par.add_trace(go.Scatter(x=top_30['Product_Name'].str[:20], y=top_30['Cum_Pct'], name='Cumulative %', yaxis='y2', mode='lines+markers', line=dict(color='#F59E0B')))
            
            fig_par.update_layout(
                height=450, title="Top 30 SKUs by Gross Margin",
                yaxis=dict(title="Gross Margin (Rp)"),
                yaxis2=dict(title="Cumulative %", overlaying='y', side='right', range=[0, 110], showgrid=False),
                xaxis=dict(tickangle=-45), hovermode="x unified",
                plot_bgcolor='white'
            )
            fig_par.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="80% Threshold", yref="y2")
            st.plotly_chart(fig_par, use_container_width=True)

            # Detail Table
            with st.expander("📋 View Financial Detail Table"):
                disp_fin = df_fin.groupby(['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier']).agg({
                    'Qty': 'sum', 'Revenue': 'sum', 'Gross_Margin': 'sum'
                }).reset_index().sort_values('Gross_Margin', ascending=False)
                
                disp_fin['Margin %'] = (disp_fin['Gross_Margin'] / disp_fin['Revenue'] * 100).fillna(0)
                
                # Formatting
                disp_fin['Revenue'] = disp_fin['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                disp_fin['Gross_Margin'] = disp_fin['Gross_Margin'].apply(lambda x: f"Rp {x:,.0f}")
                disp_fin['Margin %'] = disp_fin['Margin %'].apply(lambda x: f"{x:.1f}%")
                
                st.dataframe(disp_fin, use_container_width=True)

        else:
            st.warning("⚠️ Data Harga ('Floor_Price', 'Net_Order_Price') tidak ditemukan di Product Master. Tidak bisa menghitung profitabilitas.")
            st.info("Pastikan kolom 'Floor_Price' (Harga Jual) dan 'Net_Order_Price' (HPP) ada di file Product Master.")
    else:
        st.info("ℹ️ Tidak ada data forecast Ecommerce atau Reseller untuk dianalisis.")

# --- TAB 9: RESELLER PERFORMANCE DASHBOARD ---
with tab9:
    st.subheader("🤝 Reseller Performance Dashboard")
    st.markdown("**Comprehensive Reseller Analytics: Forecast Accuracy, Sales Performance & Inventory Planning**")
    
    # ================ 1. RESELLER PERFORMANCE TABS ================
    tab_res1, tab_res2, tab_res3, tab_res4 = st.tabs([
        "📈 Performance Overview",
        "🎯 Forecast Accuracy",
        "💰 Financial Analysis", 
        "📊 Data Explorer"
    ])
    
    # --- TAB 1: PERFORMANCE OVERVIEW ---
    with tab_res1:
        st.subheader("📊 Reseller Performance Overview")
        
        # Container untuk metrik utama
        metric_container = st.container()
        
        with metric_container:
            col1, col2, col3, col4 = st.columns(4)
            
            # Data untuk bulan Jan 26
            jan_26_data = {}
            
            # 1. Rofo Jan 26
            rofo_jan26 = 0
            if not df_past_rofo_reseller.empty:
                rofo_jan26 = df_past_rofo_reseller[
                    df_past_rofo_reseller['Month_Label'].str.contains('Jan 26', na=False)
                ]['Forecast_Qty'].sum()
            
            # 2. Sales Jan 26
            sales_jan26 = 0
            if not df_sales_reseller.empty:
                sales_jan26 = df_sales_reseller[
                    df_sales_reseller['Month_Label'].str.contains('Jan 26', na=False)
                ]['Sales_Qty'].sum()
            
            # 3. PO Jan 26
            po_jan26 = 0
            if not df_past_po_reseller.empty:
                po_jan26 = df_past_po_reseller[
                    df_past_po_reseller['Month_Label'].str.contains('Jan 26', na=False)
                ]['PO_Qty'].sum()
            
            # 4. Active SKUs - jumlah SKU unik di forecast 2026
            active_skus = len(df_reseller_forecast) if not df_reseller_forecast.empty else 0
            
            # 5. Accuracy Jan 26
            accuracy_jan26 = 0
            if rofo_jan26 > 0:
                accuracy_jan26 = 100 - abs((po_jan26 / rofo_jan26 * 100) - 100)
            
            with col1:
                st.metric("Rofo Jan 26", f"{rofo_jan26:,.0f}")
            
            with col2:
                st.metric("Sales Jan 26", f"{sales_jan26:,.0f}")
            
            with col3:
                st.metric("PO Jan 26", f"{po_jan26:,.0f}")
            
            with col4:
                st.metric("Active SKUs", f"{active_skus:,}")
            
            # Baris kedua untuk accuracy
            col5, col6 = st.columns(2)
            
            with col5:
                st.metric("Jan 26 Accuracy", f"{accuracy_jan26:.1f}%")
            
            with col6:
                # Calculate average sales per active SKU
                avg_sales_per_sku = sales_jan26 / active_skus if active_skus > 0 else 0
                st.metric("Avg Sales/SKU", f"{avg_sales_per_sku:.1f}")
        
        # ROW 2: Triple Comparison Chart - FIXED MONTH ORDER
        st.divider()
        st.subheader("📈 Triple Comparison: Forecast vs PO vs Sales")
        
        if not df_sales_reseller.empty and not df_past_rofo_reseller.empty and not df_past_po_reseller.empty:
            # Aggregate monthly data
            monthly_comparison = []
            
            # Gabungkan semua bulan unik
            all_months_set = set()
            
            # Add months from sales
            if 'Month_Label' in df_sales_reseller.columns:
                all_months_set.update(df_sales_reseller['Month_Label'].unique())
            
            # Add months from rofo
            if 'Month_Label' in df_past_rofo_reseller.columns:
                all_months_set.update(df_past_rofo_reseller['Month_Label'].unique())
            
            # Add months from po
            if 'Month_Label' in df_past_po_reseller.columns:
                all_months_set.update(df_past_po_reseller['Month_Label'].unique())
            
            # Parse bulan untuk sorting
            month_data = []
            for month_label in all_months_set:
                try:
                    # Convert month label to datetime for sorting
                    month_str = str(month_label).strip()
                    if ' ' in month_str:
                        month_part, year_part = month_str.split(' ')
                        month_date = datetime.strptime(f"{month_part[:3]}-{year_part}", "%b-%y")
                    elif '-' in month_str:
                        month_part, year_part = month_str.split('-')
                        month_date = datetime.strptime(f"{month_part[:3]}-{year_part}", "%b-%y")
                    else:
                        continue
                    
                    month_data.append({
                        'label': month_label,
                        'date': month_date,
                        'display': month_date.strftime('%b-%y')
                    })
                except:
                    continue
            
            # Sort by date
            month_data.sort(key=lambda x: x['date'])
            
            # Collect data for sorted months
            for month_info in month_data:
                month_label = month_info['label']
                month_display = month_info['display']
                
                # Sales
                sales_qty = df_sales_reseller[df_sales_reseller['Month_Label'] == month_label]['Sales_Qty'].sum()
                
                # Rofo
                rofo_qty = df_past_rofo_reseller[df_past_rofo_reseller['Month_Label'] == month_label]['Forecast_Qty'].sum()
                
                # PO
                po_qty = df_past_po_reseller[df_past_po_reseller['Month_Label'] == month_label]['PO_Qty'].sum()
                
                # Skip jika semua 0
                if sales_qty == 0 and rofo_qty == 0 and po_qty == 0:
                    continue
                
                monthly_comparison.append({
                    'Month': month_display,
                    'Month_Date': month_info['date'],
                    'Sales': sales_qty,
                    'Rofo': rofo_qty,
                    'PO': po_qty
                })
            
            if monthly_comparison:
                comp_df = pd.DataFrame(monthly_comparison)
                comp_df = comp_df.sort_values('Month_Date')
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=comp_df['Month'],
                    y=comp_df['Rofo'],
                    name='Rofo',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                fig.add_trace(go.Bar(
                    x=comp_df['Month'],
                    y=comp_df['PO'],
                    name='PO',
                    marker_color='#FF9800',
                    opacity=0.7
                ))
                
                fig.add_trace(go.Bar(
                    x=comp_df['Month'],
                    y=comp_df['Sales'],
                    name='Sales',
                    marker_color='#4CAF50',
                    opacity=0.7
                ))
                
                fig.update_layout(
                    height=400,
                    title='Reseller Performance: Rofo vs PO vs Sales',
                    xaxis_title='Month',
                    yaxis_title='Quantity',
                    barmode='group',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📊 No comparison data available for the selected period")
        else:
            st.info("ℹ️ Need sales, rofo, and PO data for comparison analysis")
        
        # ROW 3: Brand Performance Matrix - SIMPLE BAR CHART
        st.divider()
        st.subheader("🏷️ Top Performing Brands (Reseller)")
        
        if not df_reseller_forecast.empty and 'Brand' in df_reseller_forecast.columns:
            # Aggregate brand performance
            brand_performance = []
            brands = df_reseller_forecast['Brand'].unique()
            
            for brand in brands:
                brand_data = df_reseller_forecast[df_reseller_forecast['Brand'] == brand]
                
                # Forecast 2026
                forecast_2026 = brand_data[reseller_forecast_cols].sum().sum() if reseller_forecast_cols else 0
                
                # Sales Jan 26 (jika ada)
                sales_jan26 = 0
                if not df_sales_reseller.empty and 'Brand' in df_sales_reseller.columns:
                    brand_sales = df_sales_reseller[
                        (df_sales_reseller['Brand'] == brand) & 
                        (df_sales_reseller['Month_Label'].str.contains('Jan 26'))
                    ]
                    sales_jan26 = brand_sales['Sales_Qty'].sum()
                
                brand_performance.append({
                    'Brand': brand,
                    'Forecast_2026': forecast_2026,
                    'Sales_Jan26': sales_jan26,
                    'SKU_Count': len(brand_data)
                })
            
            if brand_performance:
                brand_df = pd.DataFrame(brand_performance)
                brand_df = brand_df.sort_values('Forecast_2026', ascending=False).head(10)
                
                # Simple Bar Chart (tidak kombinasi line)
                fig_brand = go.Figure()
                
                # Bar: Forecast 2026
                fig_brand.add_trace(go.Bar(
                    x=brand_df['Brand'],
                    y=brand_df['Forecast_2026'],
                    name='Forecast 2026',
                    marker_color='#667eea',
                    text=[f"{x:,.0f}" for x in brand_df['Forecast_2026']],
                    textposition='auto'
                ))
                
                fig_brand.update_layout(
                    height=400,
                    title='Top 10 Brands by Forecast 2026',
                    xaxis_title='Brand',
                    yaxis_title='Forecast Quantity',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_brand, use_container_width=True)
    
    # --- TAB 2: FORECAST ACCURACY ---
    with tab_res2:
        st.subheader("🎯 Reseller Forecast Accuracy Analysis")
        
        if not df_past_rofo_reseller.empty and not df_past_po_reseller.empty:
            # Hitung accuracy per SKU untuk Jan 26
            accuracy_data = []
            
            # Cari SKU yang ada di Jan 26
            rofo_jan26 = df_past_rofo_reseller[df_past_rofo_reseller['Month_Label'].str.contains('Jan 26', na=False)]
            po_jan26 = df_past_po_reseller[df_past_po_reseller['Month_Label'].str.contains('Jan 26', na=False)]
            
            # Gabungkan SKU yang ada di kedua dataset
            common_skus = set(rofo_jan26['SKU_ID']).intersection(set(po_jan26['SKU_ID']))
            
            for sku in common_skus:
                rofo_qty = rofo_jan26[rofo_jan26['SKU_ID'] == sku]['Forecast_Qty'].sum()
                po_qty = po_jan26[po_jan26['SKU_ID'] == sku]['PO_Qty'].sum()
                
                if rofo_qty > 0:
                    accuracy = (min(rofo_qty, po_qty) / max(rofo_qty, po_qty) * 100)
                    status = 'Accurate' if accuracy >= 80 else 'Under' if po_qty < rofo_qty else 'Over'
                    
                    # Get brand, product info, dan sales
                    brand = ''
                    product = ''
                    sales_qty = 0
                    
                    # Cari dari rofo data
                    sku_rofo_data = rofo_jan26[rofo_jan26['SKU_ID'] == sku]
                    if not sku_rofo_data.empty:
                        brand = sku_rofo_data.iloc[0].get('Brand', '')
                        product = sku_rofo_data.iloc[0].get('Product_Name', '')
                    
                    # Cari sales untuk SKU ini di Jan 26
                    if not df_sales_reseller.empty:
                        sales_data = df_sales_reseller[
                            (df_sales_reseller['SKU_ID'] == sku) & 
                            (df_sales_reseller['Month_Label'].str.contains('Jan 26', na=False))
                        ]
                        sales_qty = sales_data['Sales_Qty'].sum() if not sales_data.empty else 0
                    
                    accuracy_data.append({
                        'SKU_ID': sku,
                        'Brand': brand,
                        'Product_Name': product,
                        'Rofo_Qty': rofo_qty,
                        'PO_Qty': po_qty,
                        'Sales_Qty': sales_qty,  # TAMBAHKAN INI
                        'Accuracy': accuracy,
                        'Status': status,
                        'Variance': po_qty - rofo_qty,
                        'Variance_Pct': ((po_qty - rofo_qty) / rofo_qty * 100) if rofo_qty > 0 else 0
                    })
            
            if accuracy_data:
                accuracy_df = pd.DataFrame(accuracy_data)
                
                # Summary Metrics
                col_acc1, col_acc2, col_acc3, col_acc4 = st.columns(4)
                
                with col_acc1:
                    avg_accuracy = accuracy_df['Accuracy'].mean()
                    st.metric("Avg Accuracy", f"{avg_accuracy:.1f}%")
                
                with col_acc2:
                    accurate_count = len(accuracy_df[accuracy_df['Accuracy'] >= 80])
                    total_count = len(accuracy_df)
                    st.metric("Accurate SKUs", f"{accurate_count}/{total_count}")
                
                with col_acc3:
                    under_count = len(accuracy_df[accuracy_df['Status'] == 'Under'])
                    st.metric("Under Forecast", f"{under_count}")
                
                with col_acc4:
                    over_count = len(accuracy_df[accuracy_df['Status'] == 'Over'])
                    st.metric("Over Forecast", f"{over_count}")
                
                # Accuracy Distribution Chart
                st.divider()
                st.subheader("📊 Accuracy Distribution")
                
                fig_dist = go.Figure()
                
                # Histogram accuracy
                fig_dist.add_trace(go.Histogram(
                    x=accuracy_df['Accuracy'],
                    nbinsx=20,
                    name='Accuracy Distribution',
                    marker_color='#667eea',
                    opacity=0.7
                ))
                
                fig_dist.update_layout(
                    height=300,
                    title='Forecast Accuracy Distribution',
                    xaxis_title='Accuracy %',
                    yaxis_title='Number of SKUs',
                    bargap=0.1
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Detail Table dengan Sales_Qty
                st.divider()
                st.subheader("📋 SKU-Level Accuracy Details")
                
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'Rofo_Qty', 'PO_Qty', 
                              'Sales_Qty', 'Accuracy', 'Status', 'Variance', 'Variance_Pct']
                
                available_cols = [col for col in display_cols if col in accuracy_df.columns]
                
                detail_df = accuracy_df[available_cols].copy()
                detail_df['Accuracy'] = detail_df['Accuracy'].apply(lambda x: f"{x:.1f}%")
                detail_df['Variance_Pct'] = detail_df['Variance_Pct'].apply(lambda x: f"{x:+.1f}%")
                
                st.dataframe(
                    detail_df.sort_values('Accuracy'),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("📊 No accuracy data available for Jan 26")
        else:
            st.info("ℹ️ Need past rofo and PO data for accuracy analysis")
    
    # --- TAB 3: FINANCIAL ANALYSIS ---
    with tab_res3:
        st.subheader("💰 Reseller Financial Analysis")
        
        # Cek apakah ada data harga
        has_price_data = 'Floor_Price' in df_reseller_forecast.columns
        
        if has_price_data and reseller_forecast_cols:
            # Calculate financial projections - PERBAIKAN: hitung per SKU dengan price masing-masing
            df_financial = df_reseller_forecast.copy()
            
            # Ensure price is numeric
            df_financial['Floor_Price'] = pd.to_numeric(df_financial['Floor_Price'], errors='coerce').fillna(0)
            
            # Calculate monthly revenue projections - FIXED: Hitung revenue per SKU lalu sum
            monthly_revenue = {}
            total_revenue_2026 = 0
            
            # Debug: tampilkan sample data
            st.caption(f"📊 Data sample: {len(df_financial)} SKUs, {len(reseller_forecast_cols)} bulan")
            
            for month_col in reseller_forecast_cols:
                # Hitung revenue untuk bulan ini: SUM(quantity * floor_price per SKU)
                month_revenue = 0
                for idx, row in df_financial.iterrows():
                    qty = pd.to_numeric(row[month_col], errors='coerce')
                    price = row['Floor_Price']
                    if pd.notna(qty) and pd.notna(price):
                        month_revenue += qty * price
                
                monthly_revenue[month_col] = month_revenue
                total_revenue_2026 += month_revenue
            
            # Financial Metrics
            col_fin1, col_fin2, col_fin3 = st.columns(3)
            
            with col_fin1:
                st.metric("Total Revenue 2026", f"Rp {total_revenue_2026:,.0f}")
            
            with col_fin2:
                avg_monthly_rev = total_revenue_2026 / len(reseller_forecast_cols) if reseller_forecast_cols else 0
                st.metric("Avg Monthly Revenue", f"Rp {avg_monthly_rev:,.0f}")
            
            with col_fin3:
                if monthly_revenue:
                    peak_month = max(monthly_revenue, key=monthly_revenue.get)
                    peak_rev = monthly_revenue.get(peak_month, 0)
                    st.metric("Peak Revenue Month", f"Rp {peak_rev:,.0f}", delta=peak_month)
                else:
                    st.metric("Peak Revenue Month", "Rp 0")
            
            # Revenue Trend Chart - FIXED ORDER
            st.divider()
            st.subheader("📈 Monthly Revenue Projection (Feb 26 - Jan 27)")
            
            if monthly_revenue:
                # Sort months chronologically
                revenue_list = []
                for month_col, revenue in monthly_revenue.items():
                    try:
                        month_str = str(month_col).strip().upper()
                        
                        # Parse berbagai format bulan
                        if '_' in month_str:
                            month_part, year_part = month_str.split('_')
                            month_name = month_part[:3]
                            year_num = int(year_part) if len(year_part) == 2 else 2026
                            year_full = 2000 + year_num if year_num < 100 else year_num
                        elif ' ' in month_str:
                            month_part, year_part = month_str.split(' ')
                            month_name = month_part[:3]
                            year_num = int(year_part) if year_part.isdigit() else 2026
                            year_full = 2000 + year_num if year_num < 100 else year_num
                        elif '-' in month_str:
                            month_part, year_part = month_str.split('-')
                            month_name = month_part[:3]
                            year_num = int(year_part) if year_part.isdigit() else 2026
                            year_full = 2000 + year_num if year_num < 100 else year_num
                        else:
                            month_name = month_str[:3]
                            year_full = 2026
                        
                        # Map nama bulan ke angka
                        month_map = {
                            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                            'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                        }
                        
                        month_num = month_map.get(month_name, 1)
                        month_date = datetime(year_full, month_num, 1)
                        display_name = f"{month_name}-{str(year_full)[-2:]}"
                        
                        revenue_list.append({
                            'Month': month_col,
                            'Month_Date': month_date,
                            'Revenue': revenue,
                            'Display': display_name
                        })
                    except Exception as e:
                        st.write(f"⚠️ Error parsing {month_col}: {str(e)}")
                        continue
                
                if revenue_list:
                    revenue_df = pd.DataFrame(revenue_list)
                    revenue_df = revenue_df.sort_values('Month_Date')
                    
                    # Filter Feb 26 - Jan 27
                    start_date = datetime(2026, 2, 1)
                    end_date = datetime(2027, 2, 1)  # Termasuk Jan 27
                    
                    revenue_filtered = revenue_df[
                        (revenue_df['Month_Date'] >= start_date) & 
                        (revenue_df['Month_Date'] < end_date)
                    ].copy()
                    
                    # Debug info
                    st.caption(f"📅 Menampilkan {len(revenue_filtered)} bulan (Feb 26 - Jan 27)")
                    
                    if not revenue_filtered.empty:
                        # Urutkan display name sesuai urutan kronologis
                        revenue_filtered = revenue_filtered.sort_values('Month_Date')
                        
                        fig_rev = go.Figure()
                        
                        fig_rev.add_trace(go.Bar(
                            x=revenue_filtered['Display'],
                            y=revenue_filtered['Revenue'],
                            name='Projected Revenue',
                            marker_color='#4CAF50',
                            text=[f"Rp {x:,.0f}" for x in revenue_filtered['Revenue']],
                            textposition='auto'
                        ))
                        
                        fig_rev.update_layout(
                            height=400,
                            title='Reseller Revenue Projection (Feb 26 - Jan 27)',
                            xaxis_title='Month',
                            yaxis_title='Revenue (Rp)',
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_rev, use_container_width=True)
                        
                        # Tampilkan data tabel
                        with st.expander("📋 View Revenue Data"):
                            display_df = revenue_filtered[['Month', 'Display', 'Revenue']].copy()
                            display_df['Revenue'] = display_df['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                            st.dataframe(display_df)
                    else:
                        st.warning("⚠️ Tidak ada data untuk periode Feb 26 - Jan 27")
                        # Tampilkan semua data yang ada
                        with st.expander("📋 Lihat Semua Data Revenue"):
                            all_df = revenue_df[['Month', 'Display', 'Month_Date', 'Revenue']].copy()
                            all_df['Revenue'] = all_df['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                            all_df['Month_Date'] = all_df['Month_Date'].dt.strftime('%Y-%m')
                            st.dataframe(all_df)
                else:
                    st.warning("⚠️ Tidak ada data revenue yang bisa di-parse")
            else:
                st.warning("⚠️ Tidak ada data revenue")
                
            # PERBAIKAN: Revenue by Brand - hitung dengan benar per SKU
            st.divider()
            st.subheader("🏷️ Revenue Contribution by Brand")
            
            if 'Brand' in df_financial.columns:
                # Hitung revenue per brand
                brand_revenue_dict = {}
                
                for brand in df_financial['Brand'].unique():
                    brand_data = df_financial[df_financial['Brand'] == brand]
                    brand_rev = 0
                    
                    # Hitung revenue untuk semua bulan
                    for month_col in reseller_forecast_cols:
                        for idx, row in brand_data.iterrows():
                            qty = pd.to_numeric(row[month_col], errors='coerce')
                            price = row['Floor_Price']
                            if pd.notna(qty) and pd.notna(price):
                                brand_rev += qty * price
                    
                    brand_revenue_dict[brand] = {
                        'Revenue': brand_rev,
                        'SKU_Count': len(brand_data),
                        'Avg_Price': brand_data['Floor_Price'].mean() if not brand_data['Floor_Price'].isna().all() else 0
                    }
                
                if brand_revenue_dict:
                    # Convert to dataframe
                    brand_revenue_list = []
                    for brand, data in brand_revenue_dict.items():
                        brand_revenue_list.append({
                            'Brand': brand,
                            'Revenue': data['Revenue'],
                            'SKU_Count': data['SKU_Count'],
                            'Avg_Price': data['Avg_Price']
                        })
                    
                    brand_rev_df = pd.DataFrame(brand_revenue_list).sort_values('Revenue', ascending=False)
                    
                    fig_brand_rev = go.Figure()
                    
                    fig_brand_rev.add_trace(go.Bar(
                        x=brand_rev_df['Brand'],
                        y=brand_rev_df['Revenue'],
                        name='Revenue',
                        marker_color='#9C27B0',
                        text=[f"Rp {x:,.0f}" for x in brand_rev_df['Revenue']],
                        textposition='auto'
                    ))
                    
                    fig_brand_rev.update_layout(
                        height=400,
                        title='Brand Revenue Contribution 2026',
                        xaxis_title='Brand',
                        yaxis_title='Revenue (Rp)'
                    )
                    
                    st.plotly_chart(fig_brand_rev, use_container_width=True)
                    
                    # Tampilkan tabel ringkasan
                    with st.expander("📋 Brand Revenue Summary"):
                        summary_df = brand_rev_df.copy()
                        summary_df['Revenue'] = summary_df['Revenue'].apply(lambda x: f"Rp {x:,.0f}")
                        summary_df['Avg_Price'] = summary_df['Avg_Price'].apply(lambda x: f"Rp {x:,.0f}")
                        summary_df['Revenue_Share'] = (brand_rev_df['Revenue'] / total_revenue_2026 * 100).apply(lambda x: f"{x:.1f}%")
                        st.dataframe(summary_df[['Brand', 'SKU_Count', 'Revenue', 'Revenue_Share', 'Avg_Price']])
        
        else:
            if not has_price_data:
                st.info("ℹ️ Add 'Floor_Price' column to Reseller forecast data for financial analysis")
            else:
                st.info("ℹ️ No forecast columns available for financial analysis")
    
    # --- TAB 4: DATA EXPLORER ---
    with tab_res4:
        st.subheader("📊 Reseller Data Explorer")
        
        # Tabs for different datasets
        exp_tab1, exp_tab2, exp_tab3, exp_tab4 = st.tabs([
            "Forecast 2026",
            "Sales History",
            "Past Rofo",
            "Past PO"
        ])
        
        with exp_tab1:
            st.markdown("**Forecast 2026 Data**")
            if not df_reseller_forecast.empty:
                # Filter controls
                exp_col1, exp_col2 = st.columns(2)
                
                with exp_col1:
                    exp_brands = []
                    if 'Brand' in df_reseller_forecast.columns:
                        exp_brands = st.multiselect(
                            "Filter Brands",
                            options=df_reseller_forecast['Brand'].unique().tolist(),
                            default=[],
                            key="exp_brands_fcst"
                        )
                
                with exp_col2:
                    exp_months = st.multiselect(
                        "Months to Show",
                        options=reseller_forecast_cols,
                        default=reseller_forecast_cols[:6] if reseller_forecast_cols else [],
                        key="exp_months_fcst"
                    )
                
                # Apply filters
                df_exp = df_reseller_forecast.copy()
                if exp_brands and 'Brand' in df_exp.columns:
                    df_exp = df_exp[df_exp['Brand'].isin(exp_brands)]
                
                display_cols = ['SKU_ID', 'Product_Name', 'Brand', 'SKU_Tier', 'Floor_Price']
                if exp_months:
                    display_cols.extend(exp_months)
                
                # Filter available columns
                available_cols = [col for col in display_cols if col in df_exp.columns]
                
                st.dataframe(
                    df_exp[available_cols].head(100),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No forecast data available")
        
        with exp_tab2:
            st.markdown("**Sales History Data**")
            if not df_sales_reseller.empty:
                st.dataframe(
                    df_sales_reseller.sort_values('Month', ascending=False).head(100),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No sales data available")
        
        with exp_tab3:
            st.markdown("**Past Rofo Data**")
            if not df_past_rofo_reseller.empty:
                st.dataframe(
                    df_past_rofo_reseller,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No past rofo data available")
        
        with exp_tab4:
            st.markdown("**Past PO Data**")
            if not df_past_po_reseller.empty:
                st.dataframe(
                    df_past_po_reseller,
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No past PO data available")
        
        # Download Options
        st.divider()
        st.subheader("📥 Download Data")
        
        col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
        
        with col_dl1:
            if not df_reseller_forecast.empty:
                csv_fcst = df_reseller_forecast.to_csv(index=False)
                st.download_button(
                    label="Download Forecast 2026",
                    data=csv_fcst,
                    file_name="reseller_forecast_2026.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_fcst"
                )
        
        with col_dl2:
            if not df_sales_reseller.empty:
                csv_sales = df_sales_reseller.to_csv(index=False)
                st.download_button(
                    label="Download Sales",
                    data=csv_sales,
                    file_name="reseller_sales.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_sales"
                )
        
        with col_dl3:
            if not df_past_rofo_reseller.empty:
                csv_rofo = df_past_rofo_reseller.to_csv(index=False)
                st.download_button(
                    label="Download Past Rofo",
                    data=csv_rofo,
                    file_name="reseller_past_rofo.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_rofo"
                )
        
        with col_dl4:
            if not df_past_po_reseller.empty:
                csv_po = df_past_po_reseller.to_csv(index=False)
                st.download_button(
                    label="Download Past PO",
                    data=csv_po,
                    file_name="reseller_past_po.csv",
                    mime="text/csv",
                    use_container_width=True,
                    key="dl_po"
                )

# --- TAB 10: FULFILLMENT COST ANALYSIS (UNIT ECONOMICS) ---
with tab10:
    st.subheader("🚚 Fulfillment Cost Intelligence")
    st.caption("Executive Dashboard: Operational Efficiency, Unit Economics (BSA vs %Cost), and Cost Ratio Trends")

    # ==============================================================================
    # 1. DATA PREPARATION
    # ==============================================================================
    df_bs = all_data.get('fulfillment', pd.DataFrame())

    if not df_bs.empty:
        if 'Month_Date' not in df_bs.columns:
             df_bs['Month_Date'] = pd.to_datetime(df_bs['Month'], format='%b-%y', errors='coerce')
        
        df_bs = df_bs.sort_values('Month_Date')

        num_cols = ['Total Order(BS)', 'GMV (Fullfil By BS)', 'GMV Total (MP)', 'Total Cost', 'BSA', '%Cost']
        for c in num_cols:
            if c in df_bs.columns:
                df_bs[c] = pd.to_numeric(df_bs[c], errors='coerce').fillna(0)

        df_bs['CPO'] = np.where(df_bs['Total Order(BS)'] > 0, df_bs['Total Cost'] / df_bs['Total Order(BS)'], 0)

        avg_cpo = df_bs['CPO'].mean()
        avg_bsa = df_bs['BSA'].mean()
        avg_cost_pct = df_bs['%Cost'].mean()
        last_month_name = df_bs.iloc[-1]['Month']
        last_month_cost = df_bs.iloc[-1]['Total Cost']
        last_month_orders = df_bs.iloc[-1]['Total Order(BS)']

        # ==============================================================================
        # 2. COST & EFFICIENCY MATRIX DETAIL
        # ==============================================================================
        st.markdown("### 📋 Central Fulfillment (BS) - Cost & Efficiency Matrix")
        
        disp_cols = ['Month', 'Total Order(BS)', 'GMV (Fullfil By BS)', 'Total Cost', 'BSA', 'CPO', '%Cost']
        df_disp = df_bs[disp_cols].copy()
        
        styler = df_disp.style\
            .background_gradient(subset=['%Cost', 'CPO'], cmap='RdYlGn_r')\
            .background_gradient(subset=['BSA'], cmap='Greens')\
            .format({
                'Total Order(BS)': "{:,.0f}",
                'GMV (Fullfil By BS)': "Rp {:,.0f}",
                'Total Cost': "Rp {:,.0f}",
                'BSA': "Rp {:,.0f}",
                'CPO': "Rp {:,.0f}",
                '%Cost': "{:.2f}%"
            })
            
        st.dataframe(styler, use_container_width=True, hide_index=True, height=350)

        best_month = df_bs.loc[df_bs['%Cost'].idxmin()]
        st.success(f"🌟 **Best Efficiency Month:** {best_month['Month']} memiliki rasio biaya paling efisien yaitu **{best_month['%Cost']:.2f}%** dengan Cost per Order (CPO) sebesar **Rp {best_month['CPO']:,.0f}**.")

        # ==============================================================================
        # 3. UNIT ECONOMICS SPREAD (BSA vs % COST RATIO)
        # ==============================================================================
        st.divider()
        st.subheader("⚖️ Unit Economics Spread (BSA vs % Cost Ratio)")
        st.caption("Membuktikan Tesis: Semakin besar Basket Size (BSA), semakin kecil persentase biaya operasional (%Cost).")

        fig_unit = go.Figure()

        fig_unit.add_trace(go.Scatter(
            x=df_bs['Month'], y=df_bs['BSA'],
            name='Basket Size (BSA)',
            mode='lines+markers+text',
            text=[f"Rp {x/1000:,.0f}k" for x in df_bs['BSA']],
            textposition='top center',
            textfont=dict(color='#10B981', size=11, weight='bold'),
            line=dict(color='#10B981', width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='<b>%{x}</b><br>BSA: Rp %{y:,.0f}<extra></extra>'
        ))

        fig_unit.add_trace(go.Scatter(
            x=df_bs['Month'], y=df_bs['%Cost'],
            name='% Cost Ratio',
            mode='lines+markers+text',
            text=[f"{x:.2f}%" for x in df_bs['%Cost']],
            textposition='bottom center',
            textfont=dict(color='#EF4444', size=11, weight='bold'),
            line=dict(color='#EF4444', width=3, dash='dot'),
            marker=dict(size=8, symbol='diamond'),
            yaxis='y2',
            hovertemplate='<b>%{x}</b><br>% Cost Ratio: %{y:.2f}%<extra></extra>'
        ))

        fig_unit.update_layout(
            height=450,
            xaxis_title="",
            yaxis=dict(title=dict(text="Basket Size (Rp)", font=dict(color='#10B981')), showgrid=False, tickfont=dict(color='#10B981')),
            yaxis2=dict(
                title=dict(text="% Cost Ratio", font=dict(color='#EF4444')), 
                overlaying='y', side='right', showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                tickfont=dict(color='#EF4444'), ticksuffix="%"
            ),
            hovermode="x unified",
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
            plot_bgcolor='white',
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_unit, use_container_width=True)

        # ==============================================================================
        # 4. EXECUTIVE KPI CARDS
        # ==============================================================================
        st.divider()
        st.markdown("### 🎯 Central BS - Executive Summary KPI")
        
        st.markdown("""
        <style>
            .bs-card {
                background: white; border-radius: 12px; padding: 1.5rem;
                box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-top: 4px solid;
                text-align: center; transition: transform 0.3s ease;
            }
            .bs-card:hover { transform: translateY(-5px); }
            .bs-title { font-size: 0.85rem; color: #6B7280; font-weight: 700; text-transform: uppercase; margin-bottom: 8px;}
            .bs-val { font-size: 1.8rem; font-weight: 900; color: #1F2937; margin-bottom: 4px;}
            .bs-sub { font-size: 0.8rem; color: #9CA3AF; font-weight: 500;}
        </style>
        """, unsafe_allow_html=True)

        def render_bs_card(title, val, sub, color):
            return f"""<div class="bs-card" style="border-top-color: {color};"><div class="bs-title">{title}</div><div class="bs-val">{val}</div><div class="bs-sub">{sub}</div></div>"""

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(render_bs_card("Avg CPO", f"Rp {avg_cpo:,.0f}", "Cost Per Order", "#EF4444"), unsafe_allow_html=True)
        with c2: st.markdown(render_bs_card("Avg BSA", f"Rp {avg_bsa:,.0f}", "Basket Size", "#10B981"), unsafe_allow_html=True)
        with c3: st.markdown(render_bs_card("Avg Cost Ratio", f"{avg_cost_pct:.2f}%", "Target: Minimalis", "#F59E0B"), unsafe_allow_html=True)
        with c4: st.markdown(render_bs_card(f"Cost {last_month_name}", f"Rp {last_month_cost/1e6:,.0f} Jt", f"{last_month_orders:,.0f} Orders", "#6366F1"), unsafe_allow_html=True)

        # ==============================================================================
        # 5. SCALABILITY & COST RATIO TREND
        # ==============================================================================
        st.divider()
        col_vol, col_ratio = st.columns([1.2, 1])

        with col_vol:
            st.subheader("📦 Scalability: Orders vs Cost")
            fig_eff = go.Figure()
            fig_eff.add_trace(go.Bar(x=df_bs['Month'], y=df_bs['Total Order(BS)'], name='Total Orders', marker_color='rgba(59, 130, 246, 0.7)'))
            fig_eff.add_trace(go.Scatter(x=df_bs['Month'], y=df_bs['Total Cost'], name='Total Cost', mode='lines+markers', line=dict(color='#F97316', width=3), yaxis='y2'))
            fig_eff.update_layout(height=400, yaxis=dict(title="Orders (Qty)", showgrid=False), yaxis2=dict(title="Total Cost (Rp)", overlaying='y', side='right', showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                                legend=dict(orientation="h", y=1.1), plot_bgcolor='white', margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_eff, use_container_width=True)

        with col_ratio:
            st.subheader("📉 % Cost Ratio Trend")
            fig_ratio = go.Figure()
            fig_ratio.add_trace(go.Scatter(x=df_bs['Month'], y=df_bs['%Cost'], mode='lines+markers+text', text=[f"{x:.2f}%" for x in df_bs['%Cost']], textposition='top center',
                                fill='tozeroy', fillcolor='rgba(245, 158, 11, 0.1)', line=dict(color='#F59E0B', width=3), marker=dict(size=8), name='% Cost'))
            fig_ratio.add_hline(y=avg_cost_pct, line_dash="dash", line_color="gray", annotation_text=f"Avg: {avg_cost_pct:.2f}%")
            fig_ratio.update_layout(height=400, yaxis=dict(title="% Cost Ratio", range=[0, max(df_bs['%Cost'])*1.3]), plot_bgcolor='white', margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_ratio, use_container_width=True)

    else:
        st.warning("⚠️ Data 'BS_Fullfilment_Cost' belum tersedia.")


def _render_expansi_cost():
    # ==============================================================================
    # EXPANSI FULFILLMENT COST (dipindah ke tab Expansi Analytics)
    # ==============================================================================
    st.divider()
    st.subheader("🏪 Expansi Fulfillment Cost Analysis")
    st.caption("Deep-dive analisis biaya fulfillment per store/cabang — Efficiency Grading, Cost Contribution, Growth Scaling, dan Actionable Recommendations.")

    df_exp = all_data.get('expansi_fulfillment', pd.DataFrame())

    if not df_exp.empty:

        df_exp = df_exp.sort_values(['Store', 'Month_Date'])
        # Urutan bulan kronologis — dipakai paksa categoryarray sumbu-x (anti sort abjad)
        month_order_exp = df_exp.dropna(subset=['Month_Date']).drop_duplicates('Month').sort_values('Month_Date')['Month'].tolist()

        total_cost_exp = df_exp['Total Cost'].sum()
        total_gmv_exp = df_exp['GMV'].sum()
        total_orders_exp = df_exp['Order_Qty'].sum()
        avg_cost_pct_exp = (total_cost_exp / total_gmv_exp * 100) if total_gmv_exp > 0 else 0
        avg_cpo_exp = total_cost_exp / total_orders_exp if total_orders_exp > 0 else 0
        num_stores = df_exp['Store'].nunique()

        sorted_months_exp = sorted(df_exp['Month_Date'].dropna().unique())
        has_prev = len(sorted_months_exp) >= 2
        if has_prev:
            last_m = sorted_months_exp[-1]
            prev_m = sorted_months_exp[-2]
            df_last = df_exp[df_exp['Month_Date'] == last_m]
            df_prev = df_exp[df_exp['Month_Date'] == prev_m]
            cost_last = df_last['Total Cost'].sum()
            cost_prev = df_prev['Total Cost'].sum()
            gmv_last = df_last['GMV'].sum()
            gmv_prev = df_prev['GMV'].sum()
            orders_last = df_last['Order_Qty'].sum()
            orders_prev = df_prev['Order_Qty'].sum()
            pct_last = (cost_last / gmv_last * 100) if gmv_last > 0 else 0
            pct_prev = (cost_prev / gmv_prev * 100) if gmv_prev > 0 else 0
            cpo_last = cost_last / orders_last if orders_last > 0 else 0
            cpo_prev = cost_prev / orders_prev if orders_prev > 0 else 0
            d_cost = ((cost_last - cost_prev) / cost_prev * 100) if cost_prev > 0 else 0
            d_gmv = ((gmv_last - gmv_prev) / gmv_prev * 100) if gmv_prev > 0 else 0
            d_orders = ((orders_last - orders_prev) / orders_prev * 100) if orders_prev > 0 else 0
            d_pct = pct_last - pct_prev
            d_cpo = ((cpo_last - cpo_prev) / cpo_prev * 100) if cpo_prev > 0 else 0
        else:
            d_cost = d_gmv = d_orders = d_pct = d_cpo = 0
            cost_last = total_cost_exp
            gmv_last = total_gmv_exp
            orders_last = total_orders_exp
            pct_last = avg_cost_pct_exp
            cpo_last = avg_cpo_exp

        def _fmt_rp_short(v):
            if abs(v) >= 1e9: return f"Rp {v/1e9:,.1f} M"
            if abs(v) >= 1e6: return f"Rp {v/1e6:,.1f} Jt"
            return f"Rp {v:,.0f}"

        st.markdown("### 📊 Expansi Fulfillment Overview")

        st.markdown("""
<style>
.exp-card {
border-radius: 14px; padding: 1.2rem; color: white;
box-shadow: 0 4px 12px rgba(0,0,0,0.06); position: relative;
overflow: hidden; transition: transform 0.3s ease;
}
.exp-card:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(0,0,0,0.12); }
.exp-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; opacity: 0.92; margin-bottom: 5px; }
.exp-val { font-size: 1.6rem; font-weight: 800; margin-bottom: 2px; text-shadow: 0 1px 2px rgba(0,0,0,0.1); }
.exp-sub { font-size: 0.82rem; font-weight: 500; opacity: 0.95; display:flex; align-items:center; gap:5px; }
.exp-pill { background: rgba(255,255,255,0.22); padding: 2px 8px; border-radius: 6px; font-size: 0.72rem; font-weight: 700; backdrop-filter: blur(4px); }
</style>
""", unsafe_allow_html=True)

        def _exp_card(title, val, delta_str, sub_text, gradient):
            return f"""
<div class="exp-card" style="background:{gradient};">
<div class="exp-label">{title}</div>
<div class="exp-val">{val}</div>
<div class="exp-sub"><span class="exp-pill">{delta_str}</span> {sub_text}</div>
</div>"""

        ec1, ec2, ec3, ec4, ec5 = st.columns(5)
        with ec1:
            arrow_c = "▲" if d_cost > 0 else "▼" if d_cost < 0 else "—"
            st.markdown(_exp_card("Total Cost", _fmt_rp_short(total_cost_exp), f"{arrow_c} {abs(d_cost):.1f}%", "MoM Change", "linear-gradient(135deg, #ef5350 0%, #c62828 100%)"), unsafe_allow_html=True)
        with ec2:
            arrow_g = "▲" if d_gmv > 0 else "▼" if d_gmv < 0 else "—"
            st.markdown(_exp_card("Total GMV", _fmt_rp_short(total_gmv_exp), f"{arrow_g} {abs(d_gmv):.1f}%", "MoM Change", "linear-gradient(135deg, #42a5f5 0%, #1565c0 100%)"), unsafe_allow_html=True)
        with ec3:
            arrow_o = "▲" if d_orders > 0 else "▼" if d_orders < 0 else "—"
            st.markdown(_exp_card("Total Orders", f"{total_orders_exp:,.0f}", f"{arrow_o} {abs(d_orders):.1f}%", "MoM Change", "linear-gradient(135deg, #7e57c2 0%, #4527a0 100%)"), unsafe_allow_html=True)
        with ec4:
            arrow_p = "▲" if d_pct > 0 else "▼" if d_pct < 0 else "—"
            pct_col = "linear-gradient(135deg, #66bb6a 0%, #2e7d32 100%)" if pct_last < avg_cost_pct_exp else "linear-gradient(135deg, #ffa726 0%, #e65100 100%)"
            st.markdown(_exp_card("Avg %Cost", f"{avg_cost_pct_exp:.2f}%", f"{arrow_p} {abs(d_pct):.2f}pp", "MoM Change", pct_col), unsafe_allow_html=True)
        with ec5:
            arrow_cp = "▲" if d_cpo > 0 else "▼" if d_cpo < 0 else "—"
            st.markdown(_exp_card("Avg CPO", f"Rp {avg_cpo_exp:,.0f}", f"{arrow_cp} {abs(d_cpo):.1f}%", "MoM Change", "linear-gradient(135deg, #26a69a 0%, #00695c 100%)"), unsafe_allow_html=True)

        st.divider()
        st.subheader("🏆 Store Efficiency Scorecard")
        st.caption("Grading otomatis setiap store berdasarkan %Cost Ratio — **A** (Excellent) hingga **D** (Critical).")

        store_summary = df_exp.groupby('Store').agg({
            'Total Cost': 'sum',
            'GMV': 'sum',
            'Order_Qty': 'sum',
            'BSA': 'mean',
            'Cost_per_Order': 'mean'
        }).reset_index()
        store_summary['Cost/GMV %'] = np.where(store_summary['GMV'] > 0, store_summary['Total Cost'] / store_summary['GMV'] * 100, 0)
        store_summary['Avg CPO'] = np.where(store_summary['Order_Qty'] > 0, store_summary['Total Cost'] / store_summary['Order_Qty'], 0)
        store_summary['Cost_Share %'] = (store_summary['Total Cost'] / total_cost_exp * 100) if total_cost_exp > 0 else 0

        if has_prev:
            prev_store = df_prev.groupby('Store').agg({'Total Cost': 'sum', 'GMV': 'sum'}).reset_index()
            prev_store['Prev_%Cost'] = np.where(prev_store['GMV'] > 0, prev_store['Total Cost'] / prev_store['GMV'] * 100, 0)
            last_store = df_last.groupby('Store').agg({'Total Cost': 'sum', 'GMV': 'sum'}).reset_index()
            last_store['Last_%Cost'] = np.where(last_store['GMV'] > 0, last_store['Total Cost'] / last_store['GMV'] * 100, 0)
            mom_store = pd.merge(last_store[['Store', 'Last_%Cost']], prev_store[['Store', 'Prev_%Cost']], on='Store', how='outer').fillna(0)
            mom_store['Trend'] = mom_store['Last_%Cost'] - mom_store['Prev_%Cost']
            store_summary = pd.merge(store_summary, mom_store[['Store', 'Trend']], on='Store', how='left')
            store_summary['Trend'] = store_summary['Trend'].fillna(0)
        else:
            store_summary['Trend'] = 0

        def _grade(pct):
            if pct <= 3: return 'A'
            if pct <= 5: return 'B'
            if pct <= 8: return 'C'
            return 'D'
        store_summary['Grade'] = store_summary['Cost/GMV %'].apply(_grade)
        store_summary = store_summary.sort_values('Cost/GMV %', ascending=True)
        store_summary['Rank'] = range(1, len(store_summary) + 1)

        grade_colors = {'A': '#10B981', 'B': '#3B82F6', 'C': '#F59E0B', 'D': '#EF4444'}
        grade_labels = {'A': 'Excellent', 'B': 'Good', 'C': 'Warning', 'D': 'Critical'}

        cols_per_row = min(len(store_summary), 4)
        for row_start in range(0, len(store_summary), cols_per_row):
            row_slice = store_summary.iloc[row_start:row_start + cols_per_row]
            cols_sc = st.columns(cols_per_row)
            for idx_sc, (_, row_s) in enumerate(row_slice.iterrows()):
                g = row_s['Grade']
                gc = grade_colors.get(g, '#9CA3AF')
                trend_arrow = "▲" if row_s['Trend'] > 0 else "▼" if row_s['Trend'] < 0 else "—"
                trend_col_txt = "#EF4444" if row_s['Trend'] > 0 else "#10B981" if row_s['Trend'] < 0 else "#9CA3AF"
                with cols_sc[idx_sc]:
                    st.markdown(f"""
<div style="background:white; border-radius:14px; padding:1.2rem; box-shadow:0 4px 15px rgba(0,0,0,0.06); border-left:5px solid {gc}; margin-bottom:0.5rem;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
<span style="font-size:0.8rem; font-weight:700; color:#6B7280; text-transform:uppercase;">#{int(row_s['Rank'])} {row_s['Store']}</span>
<span style="background:{gc}; color:white; padding:3px 10px; border-radius:6px; font-size:0.85rem; font-weight:800;">{g}</span>
</div>
<div style="font-size:1.5rem; font-weight:800; color:#1F2937; margin-bottom:4px;">{row_s['Cost/GMV %']:.2f}%</div>
<div style="font-size:0.8rem; color:#6B7280;">Cost/GMV Ratio | <span style="color:{gc}; font-weight:700;">{grade_labels[g]}</span></div>
<div style="display:flex; justify-content:space-between; margin-top:10px; font-size:0.78rem; color:#9CA3AF;">
<span>CPO: Rp {row_s['Avg CPO']:,.0f}</span>
<span>BSA: Rp {row_s['BSA']:,.0f}</span>
<span style="color:{trend_col_txt}; font-weight:700;">{trend_arrow} {abs(row_s['Trend']):.2f}pp</span>
</div>
</div>
""", unsafe_allow_html=True)

        st.divider()
        col_tree, col_growth = st.columns([1, 1])

        with col_tree:
            st.subheader("🧩 Cost Contribution by Store")
            st.caption("Proporsi kontribusi biaya masing-masing store terhadap total cost Expansi.")
            fig_treemap = px.treemap(
                store_summary,
                path=['Store'],
                values='Total Cost',
                color='Cost/GMV %',
                color_continuous_scale='RdYlGn_r',
                custom_data=['Cost/GMV %', 'Order_Qty', 'Cost_Share %']
            )
            fig_treemap.update_traces(
                texttemplate="<b>%{label}</b><br>%{customdata[2]:.1f}% share<br>Cost/GMV: %{customdata[0]:.2f}%",
                hovertemplate="<b>%{label}</b><br>Total Cost: Rp %{value:,.0f}<br>Cost/GMV: %{customdata[0]:.2f}%<br>Orders: %{customdata[1]:,.0f}<extra></extra>"
            )
            fig_treemap.update_layout(height=420, margin=dict(t=10, l=5, r=5, b=10), coloraxis_colorbar=dict(title="%Cost"))
            st.plotly_chart(fig_treemap, use_container_width=True)

        with col_growth:
            st.subheader("📈 GMV vs Cost Growth Scaling")
            st.caption("Apakah pertumbuhan biaya selaras atau lebih cepat dari pertumbuhan revenue?")

            monthly_agg = df_exp.groupby('Month_Date').agg({'Total Cost': 'sum', 'GMV': 'sum', 'Order_Qty': 'sum'}).reset_index().sort_values('Month_Date')
            if len(monthly_agg) >= 2:
                base_cost = monthly_agg['Total Cost'].iloc[0]
                base_gmv = monthly_agg['GMV'].iloc[0]
                monthly_agg['Cost_Index'] = np.where(base_cost > 0, monthly_agg['Total Cost'] / base_cost * 100, 100)
                monthly_agg['GMV_Index'] = np.where(base_gmv > 0, monthly_agg['GMV'] / base_gmv * 100, 100)
                monthly_agg['Month_Lbl'] = monthly_agg['Month_Date'].dt.strftime('%b-%y')

                fig_idx = go.Figure()
                fig_idx.add_trace(go.Scatter(x=monthly_agg['Month_Lbl'], y=monthly_agg['GMV_Index'], name='GMV Growth Index', mode='lines+markers+text', text=[f"{x:.0f}" for x in monthly_agg['GMV_Index']], textposition='top center', textfont=dict(size=10, color='#1565C0', weight='bold'), line=dict(color='#42A5F5', width=3), marker=dict(size=8)))
                fig_idx.add_trace(go.Scatter(x=monthly_agg['Month_Lbl'], y=monthly_agg['Cost_Index'], name='Cost Growth Index', mode='lines+markers+text', text=[f"{x:.0f}" for x in monthly_agg['Cost_Index']], textposition='bottom center', textfont=dict(size=10, color='#C62828', weight='bold'), line=dict(color='#EF5350', width=3, dash='dot'), marker=dict(size=8)))
                fig_idx.add_hline(y=100, line_dash="dash", line_color="#9CA3AF", annotation_text="Baseline (Month 1)")
                fig_idx.update_layout(height=420, plot_bgcolor='white', yaxis_title="Index (100 = Base Month)", legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"), margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_idx, use_container_width=True)

                last_cost_idx = monthly_agg['Cost_Index'].iloc[-1]
                last_gmv_idx = monthly_agg['GMV_Index'].iloc[-1]
                if last_cost_idx > last_gmv_idx * 1.05:
                    st.error(f"🚨 **Cost Outpacing GMV!** Cost index ({last_cost_idx:.0f}) tumbuh lebih cepat dari GMV index ({last_gmv_idx:.0f}). Biaya tidak scaling efisien.")
                elif last_gmv_idx > last_cost_idx * 1.05:
                    st.success(f"✅ **Healthy Scaling!** GMV index ({last_gmv_idx:.0f}) tumbuh lebih cepat dari Cost index ({last_cost_idx:.0f}). Efisiensi membaik.")
                else:
                    st.info(f"⚖️ **Proportional Growth.** GMV ({last_gmv_idx:.0f}) dan Cost ({last_cost_idx:.0f}) tumbuh seimbang.")
            else:
                st.info("Data belum cukup untuk analisis growth (minimal 2 bulan).")

        st.divider()
        st.subheader("📋 Per-Store Cost Detail Matrix")

        disp_store = store_summary[['Rank', 'Store', 'Grade', 'Total Cost', 'GMV', 'Order_Qty', 'BSA', 'Avg CPO', 'Cost/GMV %', 'Cost_Share %', 'Trend']].copy()
        disp_store['Trend'] = disp_store['Trend'].apply(lambda x: f"{x:+.2f}pp")

        styler_exp = disp_store.style\
            .background_gradient(subset=['Cost/GMV %', 'Avg CPO'], cmap='RdYlGn_r')\
            .background_gradient(subset=['BSA'], cmap='Greens')\
            .background_gradient(subset=['Cost_Share %'], cmap='Purples')\
            .format({
                'Total Cost': 'Rp {:,.0f}',
                'GMV': 'Rp {:,.0f}',
                'Order_Qty': '{:,.0f}',
                'BSA': 'Rp {:,.0f}',
                'Avg CPO': 'Rp {:,.0f}',
                'Cost/GMV %': '{:.2f}%',
                'Cost_Share %': '{:.1f}%'
            })

        st.dataframe(styler_exp, use_container_width=True, height=400, hide_index=True)

        best_store = store_summary.iloc[0]
        worst_store = store_summary.iloc[-1]
        col_ins1, col_ins2 = st.columns(2)
        with col_ins1:
            st.success(f"🌟 **Most Efficient:** {best_store['Store']} — Cost/GMV **{best_store['Cost/GMV %']:.2f}%** | Grade **{best_store['Grade']}**")
        with col_ins2:
            st.error(f"🚨 **Least Efficient:** {worst_store['Store']} — Cost/GMV **{worst_store['Cost/GMV %']:.2f}%** | Grade **{worst_store['Grade']}**")

        st.divider()
        st.subheader("📉 Monthly %Cost Trend by Store")

        stores = sorted(df_exp['Store'].unique())
        selected_stores = st.multiselect("Pilih Store:", options=stores, default=stores, key="exp_store_selector")

        if selected_stores:
            df_trend_exp = df_exp[df_exp['Store'].isin(selected_stores)]

            fig_cost_pct = go.Figure()
            for store in selected_stores:
                store_data = df_trend_exp[df_trend_exp['Store'] == store].sort_values('Month_Date')
                fig_cost_pct.add_trace(go.Scatter(x=store_data['Month'], y=store_data['%Cost'], name=store, mode='lines+markers+text', text=[f"{x:.1f}%" for x in store_data['%Cost']], textposition='top center', textfont=dict(size=9)))

            fig_cost_pct.add_hline(y=avg_cost_pct_exp, line_dash="dash", line_color="#9CA3AF", annotation_text=f"Avg All Stores: {avg_cost_pct_exp:.2f}%")
            fig_cost_pct.update_layout(height=420, yaxis_title="% Cost Ratio", yaxis=dict(ticksuffix="%"), xaxis=dict(categoryorder='array', categoryarray=month_order_exp), hovermode="x unified", plot_bgcolor='white', legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"), margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_cost_pct, use_container_width=True)

            st.markdown("#### 💰 Cost per Order (CPO) Trend by Store")
            fig_cpo = go.Figure()
            for store in selected_stores:
                store_data = df_trend_exp[df_trend_exp['Store'] == store].sort_values('Month_Date')
                fig_cpo.add_trace(go.Bar(x=store_data['Month'], y=store_data['Cost_per_Order'], name=store, text=[f"Rp{x/1000:,.0f}k" for x in store_data['Cost_per_Order']], textposition='auto', textfont=dict(size=9)))

            fig_cpo.update_layout(height=420, yaxis_title="Cost per Order (Rp)", barmode='group', xaxis=dict(categoryorder='array', categoryarray=month_order_exp), hovermode="x unified", plot_bgcolor='white', legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"), margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(fig_cpo, use_container_width=True)

        st.divider()
        st.subheader("🗺️ Store Performance Heatmap")

        try:
            pivot_store = df_exp.pivot_table(values='%Cost', index='Store', columns='Month', aggfunc='mean').fillna(0)
            pivot_store = pivot_store.reindex(columns=[m for m in month_order_exp if m in pivot_store.columns])
            pivot_store = pivot_store.sort_index()

            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot_store.values, x=list(pivot_store.columns), y=list(pivot_store.index),
                colorscale='RdYlGn_r',
                text=[[f"{val:.2f}%" for val in row] for row in pivot_store.values],
                texttemplate="%{text}", textfont={"size": 11, "weight": "bold"},
                hovertemplate="<b>%{y}</b><br>Period: %{x}<br>%Cost: %{z:.2f}%<extra></extra>"
            ))
            fig_heat.update_layout(height=max(350, num_stores * 55), xaxis_title="Period", yaxis_title="Store", plot_bgcolor='white', yaxis=dict(autorange="reversed"), margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.warning(f"Gagal membuat heatmap: {str(e)}")

        st.divider()
        st.subheader("⚖️ Unit Economics Quadrant: BSA vs %Cost")
        st.caption("Kuadran strategis — Sumbu X: Basket Size (BSA), Sumbu Y: %Cost. Ideal: Kanan-Bawah (BSA besar, cost rendah).")

        median_bsa = store_summary['BSA'].median()

        fig_scatter_exp = px.scatter(
            store_summary, x='BSA', y='Cost/GMV %', size='Order_Qty', color='Grade',
            text='Store', size_max=50,
            color_discrete_map=grade_colors,
            custom_data=['Store', 'Avg CPO', 'Order_Qty', 'Grade']
        )

        max_bsa = store_summary['BSA'].max() * 1.3
        max_pct = store_summary['Cost/GMV %'].max() * 1.3
        fig_scatter_exp.add_shape(type="rect", x0=median_bsa, y0=0, x1=max_bsa, y1=avg_cost_pct_exp, fillcolor="rgba(16,185,129,0.06)", line_width=0, layer="below")
        fig_scatter_exp.add_shape(type="rect", x0=0, y0=avg_cost_pct_exp, x1=median_bsa, y1=max_pct, fillcolor="rgba(239,68,68,0.06)", line_width=0, layer="below")

        fig_scatter_exp.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text="🌟 EFFICIENT<br><span style='font-size:11px'>High BSA, Low Cost</span>", showarrow=False, font=dict(size=14, color="#10B981"), opacity=0.4, align="right")
        fig_scatter_exp.add_annotation(x=0.05, y=0.95, xref="paper", yref="paper", text="🚨 INEFFICIENT<br><span style='font-size:11px'>Low BSA, High Cost</span>", showarrow=False, font=dict(size=14, color="#EF4444"), opacity=0.4, align="left")

        fig_scatter_exp.add_hline(y=avg_cost_pct_exp, line_dash="dash", line_color="#9CA3AF", annotation_text=f"Avg %Cost: {avg_cost_pct_exp:.2f}%")
        fig_scatter_exp.add_vline(x=median_bsa, line_dash="dash", line_color="#9CA3AF", annotation_text=f"Median BSA: Rp {median_bsa:,.0f}")

        fig_scatter_exp.update_traces(textposition='top center', textfont=dict(size=10, weight='bold'), hovertemplate="<b>%{customdata[0]}</b><br>BSA: Rp %{x:,.0f}<br>%Cost: %{y:.2f}%<br>CPO: Rp %{customdata[1]:,.0f}<br>Orders: %{customdata[2]:,.0f}<br>Grade: %{customdata[3]}<extra></extra>")
        fig_scatter_exp.update_layout(height=500, plot_bgcolor='white', xaxis=dict(title='Basket Size (Rp) →', showgrid=True, gridcolor='rgba(0,0,0,0.05)'), yaxis=dict(title='% Cost Ratio →', showgrid=True, gridcolor='rgba(0,0,0,0.05)'), legend=dict(title="Grade"), margin=dict(t=20, b=10, l=10, r=10))
        st.plotly_chart(fig_scatter_exp, use_container_width=True)

        st.divider()
        st.subheader("💡 Auto-Generated Actionable Insights")

        insights = []

        grade_d_stores = store_summary[store_summary['Grade'] == 'D']
        if not grade_d_stores.empty:
            names_d = ", ".join(grade_d_stores['Store'].tolist())
            total_d_cost = grade_d_stores['Total Cost'].sum()
            insights.append(("🚨", "Critical Stores", f"**{len(grade_d_stores)} store** mendapat Grade D (Cost/GMV > 8%): {names_d}. Total cost: {_fmt_rp_short(total_d_cost)}. **Action:** Review kontrak logistik, negosiasi tarif, atau evaluasi kelayakan operasional.", "#FEE2E2", "#EF4444"))

        improving = store_summary[store_summary['Trend'] < -0.5]
        if not improving.empty:
            names_imp = ", ".join(improving['Store'].tolist())
            insights.append(("📉", "Improving Efficiency", f"**{len(improving)} store** menunjukkan penurunan %Cost MoM (membaik): {names_imp}. **Action:** Pertahankan strategi, identifikasi best practice untuk direplikasi.", "#D1FAE5", "#10B981"))

        worsening = store_summary[store_summary['Trend'] > 0.5]
        if not worsening.empty:
            names_wrs = ", ".join(worsening['Store'].tolist())
            insights.append(("📈", "Rising Cost Alert", f"**{len(worsening)} store** mengalami kenaikan %Cost MoM: {names_wrs}. **Action:** Investigasi penyebab — volume turun? Tarif naik? BSA mengecil?", "#FEF3C7", "#F59E0B"))

        top_cost_store = store_summary.sort_values('Cost_Share %', ascending=False).iloc[0]
        if top_cost_store['Cost_Share %'] > 40:
            insights.append(("🎯", "Cost Concentration Risk", f"**{top_cost_store['Store']}** menyumbang **{top_cost_store['Cost_Share %']:.1f}%** dari total cost. **Action:** Diversifikasi fulfillment center atau negosiasi volume discount.", "#EDE9FE", "#7C3AED"))

        low_bsa_stores = store_summary[store_summary['BSA'] < median_bsa * 0.7]
        if not low_bsa_stores.empty:
            names_lb = ", ".join(low_bsa_stores['Store'].tolist())
            insights.append(("🛒", "Low Basket Size", f"**{len(low_bsa_stores)} store** memiliki BSA jauh di bawah median: {names_lb}. **Action:** Dorong bundling, minimum order, atau promo free shipping threshold.", "#DBEAFE", "#3B82F6"))

        if not insights:
            insights.append(("✅", "All Clear", "Semua store beroperasi dalam parameter normal. Terus monitor secara berkala.", "#D1FAE5", "#10B981"))

        for icon, title, desc, bg, border in insights:
            st.markdown(f"""
<div style="background:{bg}; border-left:5px solid {border}; padding:14px 18px; border-radius:10px; margin-bottom:10px;">
<div style="font-weight:800; color:#1F2937; display:flex; align-items:center; gap:8px; font-size:0.95rem;">
<span style="font-size:1.2rem;">{icon}</span> {title}
</div>
<div style="font-size:0.88rem; color:#374151; margin-left:32px; margin-top:4px;">{desc}</div>
</div>
""", unsafe_allow_html=True)

    else:
        st.info("ℹ️ Data 'Expansi_Fullfilment_Cost' belum tersedia. Silakan tambahkan sheet dengan kolom: Store, Month, Total Cost, GMV, Order_Qty, BSA, %Cost, Cost_per_Order")


def _render_expansi_inventory():
    # ==============================================================================
    # EXPANSI STORE 360 — Inventory & Distribution (dipindah ke tab Expansi Analytics)
    # ==============================================================================
    st.divider()
    st.subheader("🔗 Expansi Store 360: Outbound → Stock → Cost")
    st.caption("Korelasi 3 sheet pada kunci Store — volume replenishment (Outbound), kesehatan stok (Inventory Branch), & biaya fulfillment (%Cost). Fokus 7 store ekspansi. PCA = gabungan Bandung + Yogyakarta (cost hanya tersedia gabungan).")

    df_invb = all_data.get('inventory_branch', pd.DataFrame())
    df_out_long = all_data.get('outbound_replen', pd.DataFrame())
    df_cost_src = all_data.get('expansi_fulfillment', pd.DataFrame())

    if df_invb.empty or df_out_long.empty:
        st.info("ℹ️ Section ini butuh sheet **'Inventory Branch'** & **'Expansi Replenishment(Outbound)'**. Pastikan keduanya sudah ada di Google Sheets (kolom Branch Store + data bulanan).")
    else:
        # ---- 7 store ekspansi = semua Inventory Branch kecuali Blibli ----
        inv7 = df_invb[~df_invb['Store'].str.contains('Blibli', case=False, na=False)].copy()
        for _c in ['Stock_Onhand', 'Replenishment', 'Avg_Sales', 'Month_Cover']:
            if _c not in inv7.columns:
                inv7[_c] = 0
        stores7 = inv7['Store'].unique().tolist()
        out7 = df_out_long[df_out_long['Store'].isin(stores7)].copy()

        # ---- CostKey: PCA Bandung/Yogya digabung jadi 'PCA' (hybrid) ----
        def _cost_key(s):
            s = str(s).strip()
            return 'PCA' if s.upper().startswith('PCA') else s
        inv7['CostKey'] = inv7['Store'].apply(_cost_key)
        out7['CostKey'] = out7['Store'].apply(_cost_key)

        # ---- Ambang coverage per lead time: PCA/retail butuh coverage lebih tinggi ----
        with st.expander("⚙️ Pengaturan ambang coverage (sesuaikan lead time)", expanded=False):
            _ct1, _ct2, _ct3 = st.columns(3)
            under_thr = _ct1.number_input("Understock di bawah (bulan)", 0.0, 5.0, 1.0, 0.1, key="cov_under_thr")
            mp_over_thr = _ct2.number_input("Overstock marketplace di atas (bulan)", 1.0, 6.0, 2.0, 0.1, key="cov_mp_thr")
            pca_over_thr = _ct3.number_input("Overstock PCA / retail di atas (bulan)", 1.0, 8.0, 3.0, 0.1, key="cov_pca_thr",
                                             help="Lead time PCA lebih panjang, jadi coverage lebih tinggi masih dianggap wajar.")

        def _over_thr(store):
            return pca_over_thr if str(store).upper().startswith('PCA') else mp_over_thr

        def _cover_status(store, cover):
            if cover < under_thr:
                return 'Understock'
            return 'Sehat' if cover <= _over_thr(store) else 'Overstock'

        _status_color = {'Understock': '#EF4444', 'Sehat': '#10B981', 'Overstock': '#F59E0B'}
        _status_color_light = {'Understock': '#FCEBEB', 'Sehat': '#EAF3DE', 'Overstock': '#FAEEDA'}
        _status_emoji = {'Understock': '🔴 Understock', 'Sehat': '🟢 Optimal', 'Overstock': '🟠 Overstock'}

        grade_colors360 = {'A': '#10B981', 'B': '#3B82F6', 'C': '#F59E0B', 'D': '#EF4444'}
        def _grade360(p):
            if p <= 3: return 'A'
            if p <= 5: return 'B'
            if p <= 8: return 'C'
            return 'D'

        # ---- Cost summary per entity (PCA combined) ----
        cost_ce = pd.DataFrame()
        if not df_cost_src.empty and 'Store' in df_cost_src.columns:
            _tmp = df_cost_src.copy()
            _tmp['Store'] = _tmp['Store'].astype(str).str.strip()
            for _c in ['Total Cost', 'GMV', 'Order_Qty', 'BSA']:
                if _c in _tmp.columns:
                    _tmp[_c] = pd.to_numeric(_tmp[_c], errors='coerce').fillna(0)
            cost_ce = _tmp.groupby('Store').agg(
                Total_Cost=('Total Cost', 'sum'),
                GMV=('GMV', 'sum'),
                Orders=('Order_Qty', 'sum'),
                BSA=('BSA', 'mean')
            ).reset_index()
            cost_ce['PctCost'] = np.where(cost_ce['GMV'] > 0, cost_ce['Total_Cost'] / cost_ce['GMV'] * 100, np.nan)
            cost_ce['Grade'] = cost_ce['PctCost'].apply(lambda p: _grade360(p) if pd.notna(p) else '-')

        # ---- Entity-level inventory + outbound (PCA combined) ----
        inv_ce = inv7.groupby('CostKey').agg(
            Stock=('Stock_Onhand', 'sum'),
            Replen=('Replenishment', 'sum'),
            AvgSales=('Avg_Sales', 'sum')
        ).reset_index()
        inv_ce['Cover'] = np.where(inv_ce['AvgSales'] > 0, inv_ce['Stock'] / inv_ce['AvgSales'], 0)

        out_ce_total = out7.groupby('CostKey')['Outbound_Qty'].sum().reset_index().rename(columns={'Outbound_Qty': 'OutboundTotal'})
        out_ce_month = out7.groupby(['CostKey', 'Month_Date'])['Outbound_Qty'].sum().reset_index().sort_values('Month_Date')
        spark_map = out_ce_month.groupby('CostKey')['Outbound_Qty'].apply(list)

        store360 = inv_ce.merge(out_ce_total, on='CostKey', how='left')
        store360['OutboundTotal'] = store360['OutboundTotal'].fillna(0)
        if not cost_ce.empty:
            store360 = store360.merge(cost_ce[['Store', 'PctCost', 'Grade', 'BSA', 'Orders']],
                                      left_on='CostKey', right_on='Store', how='left').drop(columns=['Store'])
        else:
            store360['PctCost'] = np.nan
            store360['Grade'] = '-'
        store360['Outbound_Trend'] = store360['CostKey'].map(spark_map).apply(lambda x: x if isinstance(x, list) else [])
        store360['Status'] = store360.apply(lambda r: _status_emoji[_cover_status(r['CostKey'], r['Cover'])], axis=1)
        store360 = store360.sort_values('Cover').reset_index(drop=True)

        # ===== KPI ringkas (7 cabang, split; status pakai ambang lead time) =====
        inv7 = inv7.copy()
        inv7['_cstatus'] = inv7.apply(lambda r: _cover_status(r['Store'], r['Month_Cover']), axis=1)
        n_under = int((inv7['_cstatus'] == 'Understock').sum())
        n_health = int((inv7['_cstatus'] == 'Sehat').sum())
        n_over = int((inv7['_cstatus'] == 'Overstock').sum())
        _sum_avg = inv7['Avg_Sales'].sum()
        blended = inv7['Stock_Onhand'].sum() / _sum_avg if _sum_avg > 0 else 0
        blended_proj = (inv7['Stock_Onhand'].sum() + inv7['Replenishment'].sum()) / _sum_avg if _sum_avg > 0 else 0
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🔴 Understock", f"{n_under} cabang", help=f"Coverage di bawah {under_thr:.1f} bulan — risiko stockout")
        k2.metric("🟢 Optimal", f"{n_health} cabang", help="Coverage dalam rentang target (disesuaikan lead time)")
        k3.metric("🟠 Overstock", f"{n_over} cabang", help="Coverage melebihi ambang lead time — modal mengendap")
        k4.metric("Blended Coverage", f"{blended:.1f} bulan", delta=f"proyeksi {blended_proj:.1f} bln pasca-replenishment", delta_color="off", help="Total stok ÷ total avg sales. Delta = proyeksi setelah rencana replenishment.")

        # ===== Store 360 Scorecard (PCA combined) =====
        st.markdown("#### 📋 Store 360 Scorecard")
        st.caption("Gabungan ketiga sheet per store (PCA digabung). Diurutkan dari cover terendah. Sparkline = outbound per bulan.")
        tbl = store360[['CostKey', 'Outbound_Trend', 'OutboundTotal', 'Stock', 'Replen', 'Cover', 'Status', 'PctCost', 'Grade']].copy()
        cfg360 = {
            'CostKey': st.column_config.TextColumn('Store'),
            'OutboundTotal': st.column_config.NumberColumn('Total Out', format="%.0f"),
            'Stock': st.column_config.NumberColumn('Stock', format="%.0f"),
            'Replen': st.column_config.NumberColumn('Replen', format="%.0f"),
            'Cover': st.column_config.NumberColumn('Cover', format="%.1f"),
            'Status': st.column_config.TextColumn('Status'),
            'PctCost': st.column_config.NumberColumn('%Cost', format="%.1f%%"),
            'Grade': st.column_config.TextColumn('Grade'),
        }
        try:
            cfg360['Outbound_Trend'] = st.column_config.BarChartColumn('Outbound Jan–Mei', y_min=0)
        except Exception:
            tbl = tbl.drop(columns=['Outbound_Trend'])
        st.dataframe(tbl, use_container_width=True, hide_index=True, column_config=cfg360)

        # ===== Inventory coverage: posisi saat ini vs proyeksi pasca-replenishment =====
        st.markdown("#### 📊 Inventory Coverage per Cabang — Posisi Saat Ini vs Proyeksi Pasca-Replenishment")
        st.caption("Batang penuh = coverage stok saat ini; batang transparan = tambahan dari rencana replenishment (inbound). Total batang = proyeksi coverage setelah inbound diterima. Ambang overstock disesuaikan lead time tiap cabang (PCA lebih panjang → toleransi lebih tinggi).")
        cb = inv7.sort_values('Month_Cover').copy()
        cb['Cover_Now'] = np.where(cb['Avg_Sales'] > 0, cb['Stock_Onhand'] / cb['Avg_Sales'], 0)
        cb['Cover_Replen'] = np.where(cb['Avg_Sales'] > 0, cb['Replenishment'] / cb['Avg_Sales'], 0)
        cb['Cover_Proj'] = cb['Cover_Now'] + cb['Cover_Replen']
        _now_colors = [_status_color[_cover_status(s, c)] for s, c in zip(cb['Store'], cb['Cover_Now'])]
        fig_cov = go.Figure()
        fig_cov.add_trace(go.Bar(
            y=cb['Store'], x=cb['Cover_Now'], orientation='h', name='Coverage saat ini',
            marker_color=_now_colors,
            text=[f"{c:.1f}" for c in cb['Cover_Now']], textposition='inside', insidetextanchor='middle', textfont=dict(size=10, color='white'),
            customdata=cb[['Stock_Onhand', 'Avg_Sales']].values,
            hovertemplate="<b>%{y}</b><br>Coverage saat ini: %{x:.1f} bulan<br>On-hand: %{customdata[0]:,.0f}<br>Avg sales: %{customdata[1]:,.0f}<extra></extra>"
        ))
        fig_cov.add_trace(go.Bar(
            y=cb['Store'], x=cb['Cover_Replen'], orientation='h', name='Tambahan replenishment (plan)',
            marker_color='rgba(59,130,246,0.45)',
            text=[f"+{c:.1f}" for c in cb['Cover_Replen']], textposition='inside', insidetextanchor='middle', textfont=dict(size=10),
            customdata=cb[['Replenishment', 'Cover_Proj']].values,
            hovertemplate="<b>%{y}</b><br>Qty replenishment: %{customdata[0]:,.0f}<br>Proyeksi coverage: %{customdata[1]:.1f} bulan<extra></extra>"
        ))
        fig_cov.add_vline(x=under_thr, line_dash="dash", line_color="#9CA3AF", annotation_text=f"Min {under_thr:.0f}")
        fig_cov.add_vline(x=mp_over_thr, line_dash="dot", line_color="#9CA3AF", annotation_text=f"Ambang MP {mp_over_thr:.0f}")
        fig_cov.add_vline(x=pca_over_thr, line_dash="dot", line_color="#C2853A", annotation_text=f"Ambang PCA {pca_over_thr:.0f}")
        fig_cov.update_layout(barmode='stack', height=380, plot_bgcolor='white', xaxis_title="Months of coverage",
                              yaxis=dict(autorange="reversed"),
                              legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
                              margin=dict(t=30, b=10, l=10, r=60))
        st.plotly_chart(fig_cov, use_container_width=True)

        # ===== Outbound replenishment — TABEL semua channel + konteks inventory =====
        st.markdown("#### 📦 Outbound Replenishment per Channel — dengan Konteks Inventory")
        st.caption("Qty outbound (replenishment) seluruh channel per bulan, digabung dengan AVG sales, stok on-hand, coverage, & status inventory di sebelahnya. Tanda '—' = channel central yang tidak menyimpan stok cabang.")
        ob = df_out_long.copy()
        ob_month_order = ob.dropna(subset=['Month_Date']).drop_duplicates('Month_Label').sort_values('Month_Date')['Month_Label'].tolist()
        piv_ob = ob.pivot_table(index='Store', columns='Month_Label', values='Outbound_Qty', aggfunc='sum')
        piv_ob = piv_ob.reindex(columns=[m for m in ob_month_order if m in piv_ob.columns]).fillna(0)
        month_cols_disp = list(piv_ob.columns)
        piv_ob['Total Outbound'] = piv_ob.sum(axis=1)
        ob_tbl = piv_ob.reset_index().merge(
            df_invb[['Store', 'Avg_Sales', 'Stock_Onhand', 'Month_Cover']], on='Store', how='left')
        ob_tbl['Status'] = ob_tbl.apply(
            lambda r: _status_emoji[_cover_status(r['Store'], r['Month_Cover'])] if pd.notna(r['Month_Cover']) else '—', axis=1)
        ob_tbl = ob_tbl.rename(columns={'Avg_Sales': 'AVG Sales', 'Stock_Onhand': 'On-hand', 'Month_Cover': 'Coverage'})
        ob_tbl = ob_tbl.sort_values('Total Outbound', ascending=False).reset_index(drop=True)
        ob_tbl = ob_tbl[['Store'] + month_cols_disp + ['Total Outbound', 'AVG Sales', 'On-hand', 'Coverage', 'Status']]

        _num_fmt = {m: '{:,.0f}' for m in month_cols_disp}
        _num_fmt.update({'Total Outbound': '{:,.0f}', 'AVG Sales': '{:,.0f}', 'On-hand': '{:,.0f}', 'Coverage': '{:.1f}'})

        def _style_cov_row(row):
            styles = [''] * len(row)
            if pd.notna(row['Coverage']):
                styles[row.index.get_loc('Coverage')] = f"background-color: {_status_color_light[_cover_status(row['Store'], row['Coverage'])]};"
            return styles

        try:
            _sty_ob = (ob_tbl.style
                       .background_gradient(subset=['Total Outbound'], cmap='Blues')
                       .background_gradient(subset=['On-hand'], cmap='Greens')
                       .background_gradient(subset=['AVG Sales'], cmap='Purples')
                       .apply(_style_cov_row, axis=1)
                       .format(_num_fmt, na_rep='—'))
            st.dataframe(_sty_ob, use_container_width=True, hide_index=True, height=min(560, 60 + len(ob_tbl) * 35))
        except Exception as _e_tbl:
            st.warning(f"Styling tabel outbound gagal: {_e_tbl}")
            st.dataframe(ob_tbl, use_container_width=True, hide_index=True)

        # ===== Quadrant korelasi: Cover vs %Cost (bubble = outbound) =====
        if not cost_ce.empty and store360['PctCost'].notna().any():
            st.markdown("#### ⚖️ Korelasi: Stock Cover vs %Cost")
            st.caption("Sumbu Y = efisiensi biaya (%Cost); sumbu X = coverage; bubble = volume outbound. Perhatian utama: %Cost tinggi (atas). Garis putus-putus = ambang overstock yang disesuaikan lead time.")
            q = store360.dropna(subset=['PctCost']).copy()
            avg_pct = q['PctCost'].mean()
            fig_q = px.scatter(q, x='Cover', y='PctCost', size='OutboundTotal', color='Grade',
                               text='CostKey', size_max=55, color_discrete_map=grade_colors360,
                               custom_data=['CostKey', 'OutboundTotal', 'Stock'])
            fig_q.add_vline(x=mp_over_thr, line_dash="dot", line_color="#9CA3AF", annotation_text=f"Ambang MP {mp_over_thr:.0f}")
            fig_q.add_vline(x=pca_over_thr, line_dash="dot", line_color="#C2853A", annotation_text=f"Ambang PCA {pca_over_thr:.0f}")
            fig_q.add_hline(y=avg_pct, line_dash="dash", line_color="#9CA3AF", annotation_text=f"Avg %Cost {avg_pct:.1f}%")
            fig_q.update_traces(textposition='top center', textfont=dict(size=11),
                                hovertemplate="<b>%{customdata[0]}</b><br>Cover: %{x:.1f} bln<br>%Cost: %{y:.1f}%<br>Outbound: %{customdata[1]:,.0f}<br>Stock: %{customdata[2]:,.0f}<extra></extra>")
            fig_q.update_layout(height=460, plot_bgcolor='white', xaxis_title="Month Cover (bln) →",
                                yaxis=dict(title="% Cost →", ticksuffix="%"), margin=dict(t=20, b=10, l=10, r=10))
            st.plotly_chart(fig_q, use_container_width=True)

        # ===== Auto-insight & rekomendasi =====
        st.markdown("#### 💡 Rekomendasi & Insight")
        ins360 = []
        store360['_status'] = store360.apply(lambda r: _cover_status(r['CostKey'], r['Cover']), axis=1)
        store360['_proj'] = np.where(store360['AvgSales'] > 0, (store360['Stock'] + store360['Replen']) / store360['AvgSales'], 0)

        # 1) Biaya fulfillment tinggi (Grade C/D) — fokus efisiensi, bukan stok
        if not cost_ce.empty and store360['PctCost'].notna().any():
            expensive = store360[store360['Grade'].isin(['C', 'D'])].sort_values('PctCost', ascending=False)
            for _, r in expensive.iterrows():
                ins360.append(("💸", f"{r['CostKey']}: biaya fulfillment tinggi",
                               f"%Cost <b>{r['PctCost']:.1f}%</b> (Grade {r['Grade']}) — tertinggi di antara cabang ekspansi. Coverage {r['Cover']:.1f} bulan masih wajar untuk lead time-nya, jadi fokus ke <b>efisiensi biaya</b> (review tarif logistik, dorong BSA/basket size), bukan menahan stok.",
                               "#FEF3C7", "#F59E0B"))
        # 2) Coverage rendah sekarang tapi terbantu rencana replenishment
        under_now = store360[store360['_status'] == 'Understock'].copy()
        if not under_now.empty:
            _names = ", ".join([f"{row['CostKey']} ({row['Cover']:.1f}→{row['_proj']:.1f})" for _, row in under_now.iterrows()])
            ins360.append(("📈", "Coverage rendah — terbantu rencana replenishment",
                           f"<b>{len(under_now)} cabang</b> di bawah ambang saat ini, namun rencana replenishment mengangkat proyeksi coverage (saat ini→proyeksi): {_names}. <b>Action:</b> pastikan inbound on-track agar tidak stockout sebelum barang tiba.",
                           "#D1FAE5", "#10B981"))
        # 3) Sudah cukup stok tapi rencana replen mendorong melewati ambang lead time
        over_plan = store360[(store360['_status'] != 'Understock') & (store360['_proj'] > store360['CostKey'].apply(_over_thr))].copy()
        if not over_plan.empty:
            _names2 = ", ".join([f"{row['CostKey']} ({row['Cover']:.1f}→{row['_proj']:.1f})" for _, row in over_plan.iterrows()])
            ins360.append(("⚠️", "Tinjau alokasi replenishment",
                           f"<b>{len(over_plan)} cabang</b> sudah cukup stok & proyeksi coverage-nya melewati ambang lead time setelah replen: {_names2}. <b>Action:</b> pertimbangkan tahan/kurangi alokasi agar modal tidak mengendap.",
                           "#FEE2E2", "#EF4444"))
        if not ins360:
            ins360.append(("✅", "Kondisi seimbang", "Seluruh cabang dalam parameter coverage & biaya yang wajar. Lanjutkan monitoring berkala.", "#D1FAE5", "#10B981"))
        for icon, title, desc, bg, border in ins360:
            st.markdown(f"""
<div style="background:{bg}; border-left:5px solid {border}; padding:14px 18px; border-radius:10px; margin-bottom:10px;">
<div style="font-weight:800; color:#1F2937; display:flex; align-items:center; gap:8px; font-size:0.95rem;">
<span style="font-size:1.2rem;">{icon}</span> {title}
</div>
<div style="font-size:0.88rem; color:#374151; margin-left:32px; margin-top:4px;">{desc}</div>
</div>
""", unsafe_allow_html=True)

with tab_exp:
    st.subheader("🏬 Expansi Analytics")
    st.caption("Analitik channel ekspansi: kesehatan stok, distribusi (outbound/replenishment), & biaya fulfillment. PCA = gabungan Bandung + Yogyakarta untuk biaya.")
    _exp_sub1, _exp_sub2 = st.tabs(["📦 Inventory & Distribution", "💰 Fulfillment Cost"])
    with _exp_sub1:
        _render_expansi_inventory()
    with _exp_sub2:
        _render_expansi_cost()

with tab_off:
    st.subheader("🏪 Offline Store Analytics")
    st.caption("Performa toko offline: sales bulanan, replenishment (outbound), & kesehatan stok (coverage).")

    df_off_inv = all_data.get('offline_inventory', pd.DataFrame())
    df_off_sales = all_data.get('offline_sales_long', pd.DataFrame())
    df_off_out = all_data.get('offline_outbound_long', pd.DataFrame())

    if df_off_inv.empty:
        st.info("ℹ️ Sheet Offline Store belum terbaca. Pastikan nama sheet sesuai (mis. 'Offline Store') & formatnya: tabel atas = Stock Onhand + sales bulanan + AVG Sales + Stock Cover; tabel bawah (dipisah baris kosong) = outbound/replenishment bulanan.")
    else:
        with st.expander("⚙️ Pengaturan ambang coverage offline (retail, lead time panjang)", expanded=False):
            _o1, _o2 = st.columns(2)
            off_under = _o1.number_input("Understock di bawah (bulan)", 0.0, 6.0, 1.5, 0.1, key="off_under_thr")
            off_over = _o2.number_input("Overstock di atas (bulan)", 1.0, 15.0, 4.0, 0.1, key="off_over_thr")

        def _off_status(c):
            if c < off_under:
                return 'Understock'
            return 'Sehat' if c <= off_over else 'Overstock'
        _ostatus_color = {'Understock': '#EF4444', 'Sehat': '#10B981', 'Overstock': '#F59E0B'}
        _ostatus_emoji = {'Understock': '🔴 Understock', 'Sehat': '🟢 Optimal', 'Overstock': '🟠 Overstock'}

        df_off_inv = df_off_inv.copy()
        _st_series = df_off_inv['Stock_Cover'].apply(_off_status)
        df_off_inv['Status'] = _st_series.map(_ostatus_emoji)

        _bl = df_off_inv['Stock_Onhand'].sum() / df_off_inv['Avg_Sales'].sum() if df_off_inv['Avg_Sales'].sum() > 0 else 0
        ok1, ok2, ok3, ok4 = st.columns(4)
        ok1.metric("🏪 Total Store", f"{len(df_off_inv)}")
        ok2.metric("🔴 Understock", f"{int((_st_series == 'Understock').sum())} store", help=f"Cover < {off_under:.1f} bulan")
        ok3.metric("🟠 Overstock", f"{int((_st_series == 'Overstock').sum())} store", help=f"Cover > {off_over:.1f} bulan")
        ok4.metric("Blended Coverage", f"{_bl:.1f} bulan", help="Total on-hand ÷ total avg sales")

        st.markdown("#### 📋 Ringkasan Inventory per Store")
        _disp = df_off_inv[['Store', 'Stock_Onhand', 'Avg_Sales', 'Stock_Cover', 'Status']].copy()
        try:
            _osty = (_disp.style
                     .background_gradient(subset=['Stock_Onhand'], cmap='Greens')
                     .background_gradient(subset=['Avg_Sales'], cmap='Purples')
                     .format({'Stock_Onhand': '{:,.0f}', 'Avg_Sales': '{:,.0f}', 'Stock_Cover': '{:.1f}'}))
            st.dataframe(_osty, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(_disp, use_container_width=True, hide_index=True)

        st.markdown("#### 📊 Stock Coverage per Store")
        _cbar = df_off_inv.sort_values('Stock_Cover').copy()
        _ccolors = [_ostatus_color[_off_status(c)] for c in _cbar['Stock_Cover']]
        fig_off_cov = go.Figure(go.Bar(
            x=_cbar['Stock_Cover'], y=_cbar['Store'], orientation='h', marker_color=_ccolors,
            text=[f"{c:.1f}" for c in _cbar['Stock_Cover']], textposition='outside',
            customdata=_cbar[['Stock_Onhand', 'Avg_Sales']].values,
            hovertemplate="<b>%{y}</b><br>Coverage: %{x:.1f} bulan<br>On-hand: %{customdata[0]:,.0f}<br>Avg sales: %{customdata[1]:,.0f}<extra></extra>"
        ))
        fig_off_cov.add_vline(x=off_under, line_dash="dash", line_color="#9CA3AF", annotation_text=f"Min {off_under:.0f}")
        fig_off_cov.add_vline(x=off_over, line_dash="dot", line_color="#9CA3AF", annotation_text=f"Overstock {off_over:.0f}")
        fig_off_cov.update_layout(height=max(260, len(_cbar) * 70), plot_bgcolor='white', xaxis_title="Months of coverage",
                                  yaxis=dict(autorange="reversed"), margin=dict(t=20, b=10, l=10, r=50))
        st.plotly_chart(fig_off_cov, use_container_width=True)

        oc1, oc2 = st.columns(2)
        with oc1:
            st.markdown("#### 📈 Tren Sales Bulanan")
            if not df_off_sales.empty:
                _s = df_off_sales.dropna(subset=['Month_Date']).sort_values('Month_Date')
                _order = _s.drop_duplicates('Month_Label')['Month_Label'].tolist()
                fig_os = go.Figure()
                for _stq in _s['Store'].unique():
                    _sd = _s[_s['Store'] == _stq]
                    fig_os.add_trace(go.Scatter(x=_sd['Month_Label'], y=_sd['Sales'], name=_stq, mode='lines+markers'))
                fig_os.update_layout(height=360, plot_bgcolor='white', yaxis_title="Sales (qty)",
                                     xaxis=dict(categoryorder='array', categoryarray=_order),
                                     legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'), margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_os, use_container_width=True)
            else:
                st.info("Data sales offline belum tersedia.")
        with oc2:
            st.markdown("#### 📦 Replenishment (Outbound) Bulanan")
            if not df_off_out.empty:
                _o = df_off_out.dropna(subset=['Month_Date']).sort_values('Month_Date')
                _order2 = _o.drop_duplicates('Month_Label')['Month_Label'].tolist()
                fig_oo = go.Figure()
                for _stq in _o['Store'].unique():
                    _od = _o[_o['Store'] == _stq]
                    fig_oo.add_trace(go.Bar(x=_od['Month_Label'], y=_od['Outbound'], name=_stq))
                fig_oo.update_layout(height=360, plot_bgcolor='white', barmode='group', yaxis_title="Outbound (qty)",
                                     xaxis=dict(categoryorder='array', categoryarray=_order2),
                                     legend=dict(orientation='h', y=1.15, x=0.5, xanchor='center'), margin=dict(t=30, b=10, l=10, r=10))
                st.plotly_chart(fig_oo, use_container_width=True)
            else:
                st.info("Data outbound offline belum tersedia.")

        _ov_stores = df_off_inv[_st_series.values == 'Overstock']
        if not _ov_stores.empty:
            _nm = ", ".join([f"{r['Store']} ({r['Stock_Cover']:.1f} bln)" for _, r in _ov_stores.iterrows()])
            st.markdown(f"""
<div style="background:#FEF3C7; border-left:5px solid #F59E0B; padding:14px 18px; border-radius:10px; margin-top:8px;">
<div style="font-weight:800; color:#1F2937;">🟠 Overstock terdeteksi</div>
<div style="font-size:0.88rem; color:#374151; margin-top:4px;">{_nm} — coverage jauh di atas ambang. Pertimbangkan tahan replenishment & dorong sell-through.</div>
</div>
""", unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
    <p>🚀 <strong>Inventory Intelligence Dashboard v6.0</strong> | Professional Inventory Control & Financial Analytics</p>
    <p>✅ Product Name Auto-Lookup | ✅ Financial Analysis with Price Data | ✅ Inventory Value Analysis</p>
    <p>💰 Profitability Dashboard | 📊 Seasonality Analysis | 🎯 Margin Segmentation</p>
    <p>📈 Data since January 2024 | 🔄 Real-time Google Sheets Integration</p>
</div>
""", unsafe_allow_html=True)
