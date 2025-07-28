import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import os
from io import BytesIO
import re
from typing import Dict, List, Optional, Tuple, Union
import time
import logging
import numpy as np
import hashlib
import json
from dataclasses import dataclass, asdict
from enum import Enum
import calendar
import warnings
warnings.filterwarnings('ignore')

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Konfiguration
st.set_page_config(
    page_title="TimeMoto Analytics Pro",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Erweiterte CSS für professionelles Design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
    }
    
    .insight-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #48bb78;
    }
    
    .warning-card {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #c53030;
    }
    
    .analytics-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .metric-trend {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: #718096;
    }
    
    .trend-up {
        color: #48bb78;
    }
    
    .trend-down {
        color: #f56565;
    }
    
    .stat-box {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2d3748;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #718096;
        margin-top: 0.25rem;
    }
    
    .import-status {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
    
    .duplicate-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        color: #721c24;
    }
    
    .insight-box {
        background: linear-gradient(45deg, #e3f2fd, #f3e5f5);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# Konstanten und Enums
class TimeType(Enum):
    NORMAL = "Normal"
    ABSENCE = "Abwesenheit"
    NO_RECORDING = "Keine Erfassung"
    EARLY_START = "Früher Beginn"
    LATE_END = "Spätes Ende"
    UNKNOWN = "Unbekannt"

@dataclass
class EmployeeMetrics:
    """Mitarbeiter-Metriken"""
    username: str
    total_days: int
    work_days: int
    absence_days: int
    avg_daily_hours: float
    total_hours: float
    overtime_hours: float
    balance_hours: float
    productivity_score: float
    punctuality_score: float
    absence_rate: float

@dataclass
class TeamMetrics:
    """Team-Metriken"""
    total_employees: int
    avg_work_hours: float
    total_overtime: float
    avg_absence_rate: float
    team_efficiency: float
    workload_distribution: float

class RobustDatabaseManager:
    """Robuste Datenbankverbindung ohne problematische SQL-Features"""
    
    def __init__(self):
        self.connection_string = self._get_connection_string()
        self.max_retries = 3
        self.retry_delay = 2
    
    def _get_connection_string(self) -> str:
        """Erstellt die Verbindungszeichenfolge für neon.tech"""
        database_url = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
        
        if database_url:
            # Bereinige problematische Parameter
            clean_url = database_url.replace("&channel_binding=require", "")
            clean_url = clean_url.replace("channel_binding=require&", "")
            clean_url = clean_url.replace("?channel_binding=require", "?sslmode=require")
            
            if "sslmode=" not in clean_url:
                separator = "&" if "?" in clean_url else "?"
                clean_url += f"{separator}sslmode=require"
            
            return clean_url
        
        # Fallback auf einzelne Parameter
        db_host = st.secrets.get("DB_HOST", os.getenv("DB_HOST"))
        db_port = st.secrets.get("DB_PORT", os.getenv("DB_PORT", "5432"))
        db_name = st.secrets.get("DB_NAME", os.getenv("DB_NAME"))
        db_user = st.secrets.get("DB_USER", os.getenv("DB_USER"))
        db_password = st.secrets.get("DB_PASSWORD", os.getenv("DB_PASSWORD"))
        
        if not all([db_host, db_name, db_user, db_password]):
            st.error("⚠️ Datenbankverbindung nicht konfiguriert!")
            st.stop()
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
    
    def get_connection(self):
        """Erstellt eine robuste Datenbankverbindung"""
        for attempt in range(self.max_retries):
            try:
                conn = psycopg2.connect(
                    self.connection_string,
                    connect_timeout=30,
                    keepalives=1,
                    keepalives_idle=600
                )
                
                # Teste die Verbindung
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                
                return conn
                
            except psycopg2.OperationalError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                    continue
                else:
                    st.error(f"❌ Datenbankverbindung fehlgeschlagen: {e}")
                    return None
            except Exception as e:
                st.error(f"❌ Datenbankfehler: {e}")
                return None
        
        return None
    
    def ensure_tables(self) -> bool:
        """Erstellt Tabellen mit vereinfachter Struktur"""
        
        create_sql = """
        -- Haupttabelle für Zeiteinträge
        CREATE TABLE IF NOT EXISTS time_entries (
            id SERIAL PRIMARY KEY,
            username VARCHAR(100) NOT NULL,
            entry_date DATE NOT NULL,
            start_time VARCHAR(10),
            end_time VARCHAR(10),
            breaks_duration VARCHAR(50),
            total_duration VARCHAR(10),
            duration_excluding_breaks VARCHAR(10),
            work_schedule VARCHAR(10),
            balance VARCHAR(10),
            absence_name VARCHAR(100),
            remarks TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            import_hash VARCHAR(64)
        );
        
        -- Import-Log Tabelle
        CREATE TABLE IF NOT EXISTS import_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(100) UNIQUE NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_rows INTEGER,
            processed_rows INTEGER,
            inserted_rows INTEGER,
            updated_rows INTEGER,
            skipped_rows INTEGER,
            error_rows INTEGER,
            duplicates_found INTEGER,
            import_strategy VARCHAR(50),
            errors_json TEXT,
            warnings_json TEXT
        );
        
        -- Unique Index für Duplikatsvermeidung
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_user_date 
        ON time_entries(username, entry_date);
        
        -- Import Hash Index für schnelle Duplikatsprüfung
        CREATE INDEX IF NOT EXISTS idx_import_hash
        ON time_entries(import_hash);
        
        -- Performance Indices
        CREATE INDEX IF NOT EXISTS idx_time_entries_date 
        ON time_entries(entry_date);
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_username 
        ON time_entries(username);
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_created_at 
        ON time_entries(created_at);
        """
        
        conn = self.get_connection()
        if not conn:
            return False
        
        try:
            with conn:
                with conn.cursor() as cur:
                    # Führe CREATE-Statements aus
                    statements = [stmt.strip() for stmt in create_sql.split(';') if stmt.strip() and not stmt.strip().startswith('--')]
                    
                    for statement in statements:
                        if statement:
                            try:
                                cur.execute(statement)
                            except Exception as stmt_error:
                                logger.warning(f"Statement übersprungen: {stmt_error}")
                                continue
                    
                    conn.commit()
                    
                    # Validierung
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name IN ('time_entries', 'import_sessions')
                    """)
                    
                    tables = [row[0] for row in cur.fetchall()]
                    
                    if 'time_entries' in tables:
                        st.success("✅ Datenbank-Tabellen erfolgreich erstellt/validiert")
                        return True
                    else:
                        st.error("❌ Tabellenerstellung fehlgeschlagen")
                        return False
                    
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Tabellen: {e}")
            st.error(f"❌ Schema-Fehler: {str(e)}")
            return False
        finally:
            conn.close()
    
    def test_connection(self) -> tuple[bool, str]:
        """Testet die Datenbankverbindung"""
        try:
            conn = self.get_connection()
            if conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                conn.close()
                return True, f"✅ Verbindung erfolgreich - PostgreSQL aktiv"
            else:
                return False, "❌ Verbindung fehlgeschlagen"
        except Exception as e:
            return False, f"❌ Fehler: {str(e)}"
    
    def get_statistics(self) -> Dict:
        """Holt Basis-Statistiken"""
        conn = self.get_connection()
        if not conn:
            return {'error': 'Keine Verbindung'}
        
        try:
            with conn.cursor() as cur:
                # Prüfe ob Tabelle existiert
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'time_entries'
                    )
                """)
                
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    return {'total_entries': 0, 'total_users': 0, 'table_exists': False}
                
                # Statistiken abrufen
                cur.execute("SELECT COUNT(*) as total FROM time_entries")
                total_entries = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(DISTINCT username) as users FROM time_entries")
                total_users = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) as absences FROM time_entries WHERE absence_name IS NOT NULL AND absence_name != ''")
                total_absences = cur.fetchone()[0]
                
                cur.execute("SELECT MIN(entry_date) as first_date, MAX(entry_date) as last_date FROM time_entries")
                date_result = cur.fetchone()
                
                cur.execute("SELECT MAX(created_at) as last_import FROM time_entries")
                last_import = cur.fetchone()[0]
                
                return {
                    'total_entries': total_entries,
                    'total_users': total_users,
                    'total_absences': total_absences,
                    'first_date': date_result[0] if date_result[0] else None,
                    'last_date': date_result[1] if date_result[1] else None,
                    'last_import': last_import,
                    'table_exists': True
                }
                
        except Exception as e:
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_time_entries(self, limit: int = 100, offset: int = 0, 
                        username: Optional[str] = None, 
                        start_date: Optional[date] = None, 
                        end_date: Optional[date] = None) -> pd.DataFrame:
        """Holt Zeiterfassungsdaten mit optionalen Filtern"""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            # Basis-Query
            query = """
            SELECT id, username, entry_date, start_time, end_time, 
                   breaks_duration, total_duration, duration_excluding_breaks,
                   work_schedule, balance, absence_name, remarks, 
                   created_at, updated_at
            FROM time_entries 
            WHERE 1=1
            """
            params = []
            
            # Filter hinzufügen
            if username:
                query += " AND username = %s"
                params.append(username)
            
            if start_date:
                query += " AND entry_date >= %s"
                params.append(start_date)
            
            if end_date:
                query += " AND entry_date <= %s"
                params.append(end_date)
            
            query += " ORDER BY entry_date DESC, username LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            df = pd.read_sql(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Daten: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def get_employee_metrics(self, start_date: Optional[date] = None, end_date: Optional[date] = None) -> pd.DataFrame:
        """Holt erweiterte Mitarbeiter-Metriken"""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            date_filter = ""
            params = []
            
            if start_date:
                date_filter += " AND entry_date >= %s"
                params.append(start_date)
            if end_date:
                date_filter += " AND entry_date <= %s"
                params.append(end_date)
            
            query = f"""
            WITH employee_stats AS (
                SELECT 
                    username,
                    COUNT(DISTINCT entry_date) as total_days,
                    COUNT(DISTINCT CASE WHEN absence_name IS NULL THEN entry_date END) as work_days,
                    COUNT(DISTINCT CASE WHEN absence_name IS NOT NULL THEN entry_date END) as absence_days,
                    SUM(CASE 
                        WHEN duration_excluding_breaks ~ '^[0-9]+:[0-9]+$' 
                        THEN EXTRACT(HOUR FROM duration_excluding_breaks::interval) * 60 + 
                             EXTRACT(MINUTE FROM duration_excluding_breaks::interval)
                        ELSE 0 
                    END) as total_minutes,
                    AVG(CASE 
                        WHEN start_time != '-' AND start_time != '' AND start_time ~ '^[0-9]+:[0-9]+$'
                        THEN EXTRACT(HOUR FROM start_time::time) * 60 + EXTRACT(MINUTE FROM start_time::time)
                        ELSE NULL 
                    END) as avg_start_minutes
                FROM time_entries
                WHERE 1=1 {date_filter}
                GROUP BY username
            )
            SELECT 
                username,
                total_days,
                work_days,
                absence_days,
                ROUND(total_minutes / 60.0, 2) as total_hours,
                ROUND(total_minutes / NULLIF(work_days, 0) / 60.0, 2) as avg_daily_hours,
                ROUND(100.0 * absence_days / NULLIF(total_days, 0), 2) as absence_rate,
                avg_start_minutes
            FROM employee_stats
            ORDER BY username
            """
            
            df = pd.read_sql(query, conn, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Mitarbeiter-Metriken: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

class ImportManager:
    """Verwaltet den robusten Import-Prozess"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.import_session_id = self._generate_session_id()
        
    def _generate_session_id(self) -> str:
        """Generiert eine eindeutige Session-ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"import_{timestamp}_{random_part}"
    
    def _generate_row_hash(self, row: pd.Series) -> str:
        """Generiert einen Hash für eine Zeile zur Duplikatserkennung"""
        # Erstelle einen einzigartigen String aus den wichtigen Spalten
        hash_string = f"{row['Username']}_{row['normalized_date']}_{row.get('StartTime', '')}_{row.get('EndTime', '')}"
        return hashlib.sha256(hash_string.encode()).hexdigest()
    
    def validate_and_import(self, df: pd.DataFrame, import_strategy: str = "skip_duplicates") -> Dict:
        """Hauptfunktion für validierten Import mit verbesserter Duplikatserkennung"""
        
        result = {
            'session_id': self.import_session_id,
            'timestamp': datetime.now(),
            'total_rows': len(df),
            'processed_rows': 0,
            'inserted_rows': 0,
            'updated_rows': 0,
            'skipped_rows': 0,
            'error_rows': 0,
            'duplicates_found': 0,
            'errors': [],
            'warnings': [],
            'success': False
        }
        
        try:
            # Schritt 1: Datenvalidierung
            st.info("🔍 **Schritt 1:** Validiere Datenstruktur...")
            validation_result = self._validate_data_structure(df)
            
            if not validation_result['valid']:
                result['errors'] = validation_result['errors']
                return result
            
            # Schritt 2: Datenbereinigung
            st.info("🧹 **Schritt 2:** Bereinige und normalisiere Daten...")
            cleaned_df = self._clean_and_normalize_data(df)
            result['processed_rows'] = len(cleaned_df)
            
            # Schritt 3: Duplikatserkennung
            st.info("🔍 **Schritt 3:** Erkenne Duplikate...")
            duplicate_analysis = self._analyze_duplicates(cleaned_df)
            result['duplicates_found'] = duplicate_analysis['duplicate_count']
            
            if duplicate_analysis['duplicate_count'] > 0:
                self._show_duplicate_analysis(duplicate_analysis)
                
                if import_strategy == "error_on_duplicates":
                    result['errors'].append(f"Import abgebrochen: {duplicate_analysis['duplicate_count']} Duplikate gefunden")
                    return result
            
            # Schritt 4: Import durchführen
            st.info("💾 **Schritt 4:** Führe Import durch...")
            import_result = self._execute_import(cleaned_df, import_strategy)
            
            # Ergebnisse zusammenführen
            result.update(import_result)
            result['success'] = True
            
            # Schritt 5: Import-Log erstellen
            self._log_import_session(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Import-Fehler: {e}")
            result['errors'].append(f"Unerwarteter Fehler: {str(e)}")
            return result
    
    def _validate_data_structure(self, df: pd.DataFrame) -> Dict:
        """Validiert die Datenstruktur"""
        errors = []
        warnings = []
        
        # Erforderliche Spalten
        required_columns = ['Username', 'Date', 'StartTime', 'EndTime', 'Duration']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            errors.append(f"Fehlende erforderliche Spalten: {', '.join(missing_columns)}")
        
        # Datenqualität prüfen
        if 'Username' in df.columns:
            empty_usernames = df['Username'].isna().sum()
            if empty_usernames > 0:
                warnings.append(f"{empty_usernames} Zeilen haben leere Benutzernamen")
        
        if 'Date' in df.columns:
            empty_dates = df['Date'].isna().sum()
            if empty_dates > 0:
                errors.append(f"{empty_dates} Zeilen haben leere Datumsangaben")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _clean_and_normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereinigt und normalisiert die Daten"""
        
        # Kopie erstellen
        cleaned_df = df.copy()
        
        # Total-Zeilen entfernen
        cleaned_df = cleaned_df[cleaned_df['Username'].str.lower() != 'total'].copy()
        
        # Leere Zeilen entfernen
        cleaned_df = cleaned_df.dropna(subset=['Username', 'Date'], how='any')
        
        # Datum normalisieren
        cleaned_df['normalized_date'] = cleaned_df['Date'].apply(self._parse_german_date)
        
        # Ungültige Datumszeilen entfernen
        valid_dates = cleaned_df['normalized_date'].notna()
        if not valid_dates.all():
            invalid_count = (~valid_dates).sum()
            st.warning(f"⚠️ {invalid_count} Zeilen mit ungültigen Datumsangaben entfernt")
            cleaned_df = cleaned_df[valid_dates].copy()
        
        # String-Spalten bereinigen
        string_columns = ['Username', 'StartTime', 'EndTime', 'Duration', 'DurationExcludingBreaks', 
                         'Balance', 'WorkSchedule', 'Breaks', 'AbsenceName', 'Remarks']
        
        for col in string_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
                cleaned_df[col] = cleaned_df[col].replace('nan', '')
                cleaned_df[col] = cleaned_df[col].replace('None', '')
        
        # Import-Hash generieren
        cleaned_df['import_hash'] = cleaned_df.apply(self._generate_row_hash, axis=1)
        
        return cleaned_df
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict:
        """Analysiert Duplikate in den Daten"""
        
        # Duplikate basierend auf Username + Datum
        date_duplicates = df.groupby(['Username', 'normalized_date']).size()
        date_duplicates = date_duplicates[date_duplicates > 1]
        
        # Bestehende Duplikate in der Datenbank prüfen
        existing_duplicates = self._check_existing_duplicates(df)
        
        return {
            'duplicate_count': len(date_duplicates) + len(existing_duplicates),
            'date_duplicates': date_duplicates,
            'existing_duplicates': existing_duplicates,
            'analysis': {
                'same_date_entries': len(date_duplicates),
                'already_in_database': len(existing_duplicates)
            }
        }
    
    def _check_existing_duplicates(self, df: pd.DataFrame) -> List[Dict]:
        """Prüft welche Daten bereits in der Datenbank existieren"""
        existing_duplicates = []
        
        conn = self.db_manager.get_connection()
        if not conn:
            return existing_duplicates
        
        try:
            with conn.cursor() as cur:
                # Batch-Prüfung für bessere Performance
                user_date_pairs = [(row['Username'], row['normalized_date']) 
                                 for _, row in df.iterrows()]
                
                # Erstelle temporäre Tabelle für Batch-Vergleich
                cur.execute("""
                    CREATE TEMP TABLE temp_import_check (
                        username VARCHAR(100),
                        entry_date DATE
                    )
                """)
                
                # Füge Daten in temporäre Tabelle ein
                insert_query = "INSERT INTO temp_import_check (username, entry_date) VALUES (%s, %s)"
                cur.executemany(insert_query, user_date_pairs)
                
                # Prüfe Duplikate in einem Query
                cur.execute("""
                    SELECT t.username, t.entry_date, e.id, e.created_at
                    FROM temp_import_check t
                    INNER JOIN time_entries e 
                    ON t.username = e.username AND t.entry_date = e.entry_date
                """)
                
                for row in cur.fetchall():
                    existing_duplicates.append({
                        'username': row[0],
                        'date': row[1],
                        'existing_id': row[2],
                        'existing_created_at': row[3]
                    })
                
                # Temporäre Tabelle löschen
                cur.execute("DROP TABLE temp_import_check")
        
        except Exception as e:
            logger.error(f"Fehler bei Duplikatsprüfung: {e}")
        finally:
            conn.close()
        
        return existing_duplicates
    
    def _show_duplicate_analysis(self, duplicate_analysis: Dict):
        """Zeigt Duplikatsanalyse in der UI"""
        
        st.markdown("""
        <div class="duplicate-warning">
            <h4>⚠️ Duplikate gefunden</h4>
        </div>
        """, unsafe_allow_html=True)
        
        analysis = duplicate_analysis['analysis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("📅 Gleiche Daten (Datei)", analysis['same_date_entries'])
        
        with col2:
            st.metric("💾 Bereits in DB", analysis['already_in_database'])
        
        # Details anzeigen
        if analysis['already_in_database'] > 0:
            with st.expander("🔍 Details zu existierenden Einträgen"):
                existing_df = pd.DataFrame(duplicate_analysis['existing_duplicates'])
                if not existing_df.empty:
                    st.dataframe(existing_df[['username', 'date', 'existing_created_at']], use_container_width=True)
    
    def _execute_import(self, df: pd.DataFrame, strategy: str) -> Dict:
        """Führt den eigentlichen Import durch mit Batch-Operationen"""
        
        result = {
            'inserted_rows': 0,
            'updated_rows': 0,
            'skipped_rows': 0,
            'error_rows': 0,
            'errors': []
        }
        
        conn = self.db_manager.get_connection()
        if not conn:
            result['errors'].append("Keine Datenbankverbindung")
            return result
        
        # SQL-Statements
        insert_sql = """
        INSERT INTO time_entries (
            username, entry_date, start_time, end_time, breaks_duration,
            total_duration, duration_excluding_breaks, work_schedule,
            balance, absence_name, remarks, import_hash, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
        )
        ON CONFLICT (username, entry_date) DO NOTHING
        RETURNING id
        """
        
        update_sql = """
        UPDATE time_entries SET
            start_time = %s,
            end_time = %s,
            breaks_duration = %s,
            total_duration = %s,
            duration_excluding_breaks = %s,
            work_schedule = %s,
            balance = %s,
            absence_name = %s,
            remarks = %s,
            import_hash = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE username = %s AND entry_date = %s
        """
        
        try:
            with conn:
                with conn.cursor() as cur:
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    
                    # Batch-Verarbeitung für bessere Performance
                    batch_size = 100
                    total_rows = len(df)
                    
                    for batch_start in range(0, total_rows, batch_size):
                        batch_end = min(batch_start + batch_size, total_rows)
                        batch_df = df.iloc[batch_start:batch_end]
                        
                        # Progress Update
                        progress = batch_end / total_rows
                        progress_bar.progress(progress)
                        progress_text.text(f"Verarbeite Batch {batch_start + 1}-{batch_end} von {total_rows}")
                        
                        if strategy == "skip_duplicates":
                            # Batch-Insert mit ON CONFLICT DO NOTHING
                            for _, row in batch_df.iterrows():
                                try:
                                    values = self._prepare_row_values(row)
                                    cur.execute(insert_sql, values)
                                    
                                    if cur.rowcount > 0:
                                        result['inserted_rows'] += 1
                                    else:
                                        result['skipped_rows'] += 1
                                        
                                except Exception as row_error:
                                    result['errors'].append(f"Zeile {row.name}: {str(row_error)}")
                                    result['error_rows'] += 1
                        
                        elif strategy == "update_duplicates":
                            # Prüfe und aktualisiere existierende Einträge
                            for _, row in batch_df.iterrows():
                                try:
                                    # Prüfe ob Eintrag existiert
                                    cur.execute(
                                        "SELECT id FROM time_entries WHERE username = %s AND entry_date = %s",
                                        (row['Username'], row['normalized_date'])
                                    )
                                    existing = cur.fetchone()
                                    
                                    if existing:
                                        # Update
                                        update_values = self._prepare_update_values(row)
                                        cur.execute(update_sql, update_values)
                                        result['updated_rows'] += 1
                                    else:
                                        # Insert
                                        values = self._prepare_row_values(row)
                                        cur.execute(insert_sql, values)
                                        result['inserted_rows'] += 1
                                        
                                except Exception as row_error:
                                    result['errors'].append(f"Zeile {row.name}: {str(row_error)}")
                                    result['error_rows'] += 1
                    
                    # Commit der Transaktion
                    conn.commit()
                    progress_bar.empty()
                    progress_text.empty()
                    
                    # Erfolgs-Validierung
                    cur.execute("SELECT COUNT(*) FROM time_entries")
                    total_in_db = cur.fetchone()[0]
                    
                    st.success(f"✅ **Import abgeschlossen!** Gesamt in Datenbank: {total_in_db} Einträge")
                    
        except Exception as e:
            error_msg = f"Import-Fehler: {str(e)}"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        finally:
            conn.close()
        
        return result
    
    def _prepare_row_values(self, row) -> tuple:
        """Bereitet Datenwerte für Insert vor"""
        return (
            row['Username'],
            row['normalized_date'],
            row.get('StartTime', ''),
            row.get('EndTime', ''),
            row.get('Breaks', ''),
            row.get('Duration', ''),
            row.get('DurationExcludingBreaks', ''),
            row.get('WorkSchedule', ''),
            row.get('Balance', ''),
            row.get('AbsenceName') if pd.notna(row.get('AbsenceName')) else None,
            row.get('Remarks') if pd.notna(row.get('Remarks')) else None,
            row.get('import_hash', '')
        )
    
    def _prepare_update_values(self, row) -> tuple:
        """Bereitet Datenwerte für Update vor"""
        return (
            row.get('StartTime', ''),
            row.get('EndTime', ''),
            row.get('Breaks', ''),
            row.get('Duration', ''),
            row.get('DurationExcludingBreaks', ''),
            row.get('WorkSchedule', ''),
            row.get('Balance', ''),
            row.get('AbsenceName') if pd.notna(row.get('AbsenceName')) else None,
            row.get('Remarks') if pd.notna(row.get('Remarks')) else None,
            row.get('import_hash', ''),
            row['Username'],
            row['normalized_date']
        )
    
    def _log_import_session(self, result: Dict):
        """Protokolliert die Import-Session"""
        
        conn = self.db_manager.get_connection()
        if not conn:
            return
        
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO import_sessions (
                            session_id, total_rows, processed_rows, inserted_rows,
                            updated_rows, skipped_rows, error_rows, duplicates_found,
                            import_strategy, errors_json, warnings_json
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                        )
                    """, (
                        result['session_id'],
                        result['total_rows'],
                        result['processed_rows'],
                        result['inserted_rows'],
                        result['updated_rows'],
                        result['skipped_rows'],
                        result['error_rows'],
                        result['duplicates_found'],
                        'default',
                        json.dumps(result.get('errors', [])),
                        json.dumps(result.get('warnings', []))
                    ))
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Fehler beim Protokollieren: {e}")
        finally:
            conn.close()
    
    def _parse_german_date(self, date_str: str) -> Optional[str]:
        """Konvertiert deutsches Datumsformat zu ISO"""
        if not date_str or pd.isna(date_str):
            return None
        
        german_days = {
            'Montag': 'Monday', 'Dienstag': 'Tuesday', 'Mittwoch': 'Wednesday',
            'Donnerstag': 'Thursday', 'Freitag': 'Friday', 'Samstag': 'Saturday',
            'Sonntag': 'Sunday'
        }
        
        german_months = {
            'Januar': 'January', 'Februar': 'February', 'März': 'March',
            'April': 'April', 'Mai': 'May', 'Juni': 'June',
            'Juli': 'July', 'August': 'August', 'September': 'September',
            'Oktober': 'October', 'November': 'November', 'Dezember': 'December'
        }
        
        try:
            english_date = str(date_str)
            for german, english in german_days.items():
                english_date = english_date.replace(german, english)
            for german, english in german_months.items():
                english_date = english_date.replace(german, english)
            
            parsed_date = datetime.strptime(english_date, "%A, %d. %B %Y")
            return parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning(f"Datum konnte nicht geparst werden: {date_str} - {e}")
            return None
    
    def get_import_history(self) -> pd.DataFrame:
        """Holt die Import-Historie"""
        conn = self.db_manager.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT session_id, timestamp, total_rows, processed_rows,
                   inserted_rows, updated_rows, skipped_rows, error_rows,
                   duplicates_found, import_strategy
            FROM import_sessions 
            ORDER BY timestamp DESC
            LIMIT 50
            """
            df = pd.read_sql(query, conn)
            return df
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Import-Historie: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

class TimeMotoAnalytics:
    """Analytics-Klasse für TimeMoto Daten"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = self.prepare_data(df)
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereitet die Daten für Analysen vor"""
        df = df.copy()
        
        # Datum parsen
        df['parsed_date'] = pd.to_datetime(df['entry_date'])
        df['weekday'] = df['parsed_date'].dt.day_name()
        df['week_number'] = df['parsed_date'].dt.isocalendar().week
        df['month'] = df['parsed_date'].dt.month
        df['year'] = df['parsed_date'].dt.year
        
        # Arbeitscodes aus Remarks extrahieren
        df['work_code'] = df['remarks'].apply(self.extract_work_code)
        
        # Balance zu numerisch konvertieren
        df['balance_minutes'] = df['balance'].apply(self.time_to_minutes)
        df['duration_minutes'] = df['duration_excluding_breaks'].apply(self.time_to_minutes)
        
        # Arbeitszeitttyp klassifizieren
        df['time_type'] = df.apply(self.classify_time_type, axis=1)
        
        return df
    
    def extract_work_code(self, remarks: str) -> Optional[str]:
        """Extrahiert Arbeitscodes aus Remarks"""
        if not remarks or pd.isna(remarks):
            return None
        
        match = re.search(r'Arbeitscodes:\s*([^.\r\n]+)', str(remarks))
        if match:
            return match.group(1).strip()
        return None
    
    def time_to_minutes(self, time_str: str) -> int:
        """Konvertiert Zeitstring zu Minuten"""
        if not time_str or pd.isna(time_str) or time_str in ['', '-']:
            return 0
        
        try:
            clean_time = str(time_str).replace('+', '').replace('-', '')
            
            if ':' in clean_time:
                parts = clean_time.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                total_minutes = hours * 60 + minutes
                
                if str(time_str).startswith('-'):
                    total_minutes = -total_minutes
                
                return total_minutes
        except:
            pass
        
        return 0
    
    def classify_time_type(self, row) -> str:
        """Klassifiziert den Arbeitszeitttyp"""
        start_time = str(row['start_time'])
        end_time = str(row['end_time'])
        absence = row['absence_name']
        
        if absence and not pd.isna(absence):
            return 'Abwesenheit'
        elif start_time == '-' or end_time == '-':
            return 'Keine Erfassung'
        elif start_time == '<':
            return 'Früher Beginn'
        elif end_time == '>':
            return 'Spätes Ende'
        elif start_time and end_time and start_time != '' and end_time != '':
            return 'Normal'
        else:
            return 'Unbekannt'
    
    def get_employee_summary(self) -> pd.DataFrame:
        """Erstellt Mitarbeiter-Zusammenfassung"""
        summary = self.df.groupby('username').agg({
            'parsed_date': 'count',
            'balance_minutes': 'sum',
            'duration_minutes': 'sum',
            'absence_name': lambda x: x.notna().sum(),
            'work_code': lambda x: x.notna().sum()
        }).reset_index()
        
        summary.columns = ['Mitarbeiter', 'Arbeitstage', 'Saldo_Minuten', 'Arbeitszeit_Minuten', 'Abwesenheiten', 'Projekte']
        
        summary['Saldo_Stunden'] = round(summary['Saldo_Minuten'] / 60, 2)
        summary['Arbeitszeit_Stunden'] = round(summary['Arbeitszeit_Minuten'] / 60, 2)
        
        return summary
    
    def get_daily_analysis(self) -> pd.DataFrame:
        """Tägliche Analyse"""
        daily = self.df.groupby('parsed_date').agg({
            'username': 'nunique',
            'balance_minutes': 'sum',
            'duration_minutes': 'sum',
            'absence_name': lambda x: x.notna().sum(),
            'time_type': lambda x: (x == 'Normal').sum()
        }).reset_index()
        
        daily.columns = ['Datum', 'Mitarbeiter_Anzahl', 'Gesamt_Saldo_Min', 'Gesamt_Arbeitszeit_Min', 'Abwesenheiten', 'Normale_Erfassungen']
        daily['Gesamt_Saldo_Std'] = round(daily['Gesamt_Saldo_Min'] / 60, 2)
        daily['Gesamt_Arbeitszeit_Std'] = round(daily['Gesamt_Arbeitszeit_Min'] / 60, 2)
        
        return daily
    
    def calculate_overtime(self, standard_hours: float = 8.0) -> pd.DataFrame:
        """Berechnet Überstunden pro Mitarbeiter"""
        overtime_df = self.df.copy()
        overtime_df['standard_minutes'] = standard_hours * 60
        overtime_df['overtime_minutes'] = overtime_df['duration_minutes'] - overtime_df['standard_minutes']
        overtime_df['overtime_minutes'] = overtime_df['overtime_minutes'].clip(lower=0)
        
        summary = overtime_df.groupby(['username', pd.Grouper(key='parsed_date', freq='W')])['overtime_minutes'].sum().reset_index()
        summary['overtime_hours'] = round(summary['overtime_minutes'] / 60, 2)
        
        return summary
    
    def calculate_productivity_metrics(self) -> pd.DataFrame:
        """Berechnet Produktivitätsmetriken"""
        # Arbeitszeit-Effizienz
        work_df = self.df[self.df['time_type'] == 'Normal'].copy()
        
        if work_df.empty:
            return pd.DataFrame()
        
        metrics = []
        for user in work_df['username'].unique():
            user_data = work_df[work_df['username'] == user]
            
            if len(user_data) == 0:
                continue
            
            # Durchschnittliche Tagesleistung
            avg_daily_hours = user_data['duration_minutes'].mean() / 60
            
            # Konsistenz (niedrigere Standardabweichung = höhere Konsistenz)
            if len(user_data) > 1 and user_data['duration_minutes'].std() > 0:
                consistency = 100 - min(user_data['duration_minutes'].std() / user_data['duration_minutes'].mean() * 100, 100)
            else:
                consistency = 100
            
            # Pünktlichkeit (Anteil der Tage mit normalem Start)
            punctuality = (user_data['time_type'] == 'Normal').sum() / len(user_data) * 100
            
            # Produktivitätsscore
            productivity_score = (avg_daily_hours / 8 * 40 + consistency * 0.3 + punctuality * 0.3)
            
            metrics.append({
                'username': user,
                'avg_daily_hours': round(avg_daily_hours, 2),
                'consistency_score': round(consistency, 1),
                'punctuality_score': round(punctuality, 1),
                'productivity_score': round(productivity_score, 1)
            })
        
        return pd.DataFrame(metrics)
    
    def analyze_absence_patterns(self) -> Dict:
        """Analysiert Abwesenheitsmuster"""
        absence_df = self.df[self.df['absence_name'].notna()].copy()
        
        if absence_df.empty:
            return {'patterns': pd.DataFrame(), 'insights': []}
        
        # Abwesenheit nach Wochentag
        absence_df['weekday'] = absence_df['parsed_date'].dt.day_name()
        weekday_pattern = absence_df.groupby('weekday').size()
        
        # Abwesenheit nach Monat
        absence_df['month'] = absence_df['parsed_date'].dt.month_name()
        monthly_pattern = absence_df.groupby('month').size()
        
        # Häufigste Abwesenheitsgründe
        reason_pattern = absence_df['absence_name'].value_counts()
        
        # Abwesenheitsdauer-Analyse
        absence_duration = absence_df.groupby(['username', 'absence_name']).size()
        
        # Insights generieren
        insights = []
        
        # Wochentag-Insight
        if len(weekday_pattern) > 0:
            max_weekday = weekday_pattern.idxmax()
            if weekday_pattern[max_weekday] > weekday_pattern.mean() * 1.5:
                insights.append(f"📊 Auffällig viele Abwesenheiten am {max_weekday}")
        
        # Monats-Insight
        if len(monthly_pattern) > 0:
            max_month = monthly_pattern.idxmax()
            if monthly_pattern[max_month] > monthly_pattern.mean() * 1.5:
                insights.append(f"📅 Erhöhte Abwesenheiten im {max_month}")
        
        return {
            'weekday_pattern': weekday_pattern,
            'monthly_pattern': monthly_pattern,
            'reason_pattern': reason_pattern,
            'duration_analysis': absence_duration,
            'insights': insights
        }
    
    def calculate_team_workload_distribution(self) -> pd.DataFrame:
        """Berechnet die Arbeitsverteilung im Team"""
        workload = self.df.groupby('username').agg({
            'duration_minutes': ['sum', 'mean', 'std'],
            'work_code': lambda x: x.notna().sum()
        }).reset_index()
        
        workload.columns = ['username', 'total_minutes', 'avg_minutes', 'std_minutes', 'project_count']
        workload['total_hours'] = round(workload['total_minutes'] / 60, 2)
        workload['workload_share'] = round(workload['total_minutes'] / workload['total_minutes'].sum() * 100, 2)
        
        # Gini-Koeffizient für Gleichverteilung
        sorted_workload = np.sort(workload['total_minutes'])
        n = len(sorted_workload)
        if n > 0:
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * sorted_workload)) / (n * np.sum(sorted_workload)) - (n + 1) / n
        else:
            gini = 0
        
        workload['team_gini_coefficient'] = round(gini, 3)
        
        return workload
    
    def predict_future_workload(self, days_ahead: int = 30) -> pd.DataFrame:
        """Einfache Vorhersage der zukünftigen Arbeitsbelastung"""
        # Historische Daten nach Wochentag gruppieren
        historical = self.df.groupby([self.df['parsed_date'].dt.dayofweek, 'username'])['duration_minutes'].mean().reset_index()
        
        # Zukünftige Daten generieren
        future_dates = pd.date_range(start=datetime.now(), periods=days_ahead, freq='D')
        predictions = []
        
        for date in future_dates:
            dow = date.dayofweek
            for user in self.df['username'].unique():
                hist_data = historical[(historical['parsed_date'] == dow) & (historical['username'] == user)]
                if not hist_data.empty:
                    predicted_minutes = hist_data['duration_minutes'].iloc[0]
                    predictions.append({
                        'date': date,
                        'username': user,
                        'predicted_hours': round(predicted_minutes / 60, 2),
                        'weekday': date.strftime('%A')
                    })
        
        return pd.DataFrame(predictions)

class AdvancedVisualizationManager:
    """Manager für erweiterte Visualisierungen"""
    
    @staticmethod
    def create_heatmap(data: pd.DataFrame, x_col: str, y_col: str, z_col: str, title: str) -> go.Figure:
        """Erstellt eine Heatmap"""
        try:
            pivot_data = data.pivot(index=y_col, columns=x_col, values=z_col)
            
            fig = go.Figure(data=go.Heatmap(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Viridis',
                text=pivot_data.values,
                texttemplate='%{text:.0f}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title=x_col,
                yaxis_title=y_col,
                height=600
            )
            
            return fig
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Heatmap: {e}")
            return None
    
    @staticmethod
    def create_radar_chart(data: pd.DataFrame, categories: List[str], title: str) -> go.Figure:
        """Erstellt ein Radar-Chart"""
        try:
            fig = go.Figure()
            
            for _, row in data.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row[cat] for cat in categories],
                    theta=categories,
                    fill='toself',
                    name=row['username']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                title=title
            )
            
            return fig
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Radar-Charts: {e}")
            return None
    
    @staticmethod
    def create_box_plot(data: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
        """Erstellt Box-Plots"""
        try:
            fig = px.box(data, x=x_col, y=y_col, title=title)
            fig.update_layout(showlegend=False)
            return fig
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Box-Plots: {e}")
            return None
    
    @staticmethod
    def create_treemap(data: pd.DataFrame, path: List[str], values: str, title: str) -> go.Figure:
        """Erstellt eine Treemap"""
        try:
            fig = px.treemap(
                data,
                path=path,
                values=values,
                title=title
            )
            fig.update_traces(textinfo="label+percent parent+value")
            return fig
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Treemap: {e}")
            return None

class ReportGenerator:
    """Generiert professionelle Berichte"""
    
    def __init__(self, analytics: TimeMotoAnalytics):
        self.analytics = analytics
    
    def generate_executive_summary(self) -> Dict:
        """Generiert Executive Summary"""
        df = self.analytics.df
        
        # Kernmetriken berechnen
        total_employees = df['username'].nunique()
        total_work_hours = df['duration_minutes'].sum() / 60
        avg_daily_hours = df.groupby(['username', 'parsed_date'])['duration_minutes'].sum().mean() / 60 if not df.empty else 0
        absence_rate = (df['absence_name'].notna().sum() / len(df)) * 100 if len(df) > 0 else 0
        
        # Trends berechnen
        weekly_hours = df.groupby(pd.Grouper(key='parsed_date', freq='W'))['duration_minutes'].sum()
        if len(weekly_hours) > 1:
            trend = "steigend" if weekly_hours.iloc[-1] > weekly_hours.mean() else "fallend"
        else:
            trend = "unbekannt"
        
        # Top-Performer identifizieren
        productivity_metrics = self.analytics.calculate_productivity_metrics()
        if not productivity_metrics.empty:
            top_performer = productivity_metrics.nlargest(1, 'productivity_score').iloc[0].to_dict()
        else:
            top_performer = {}
        
        # Insights generieren
        insights = []
        
        if absence_rate > 10:
            insights.append(f"⚠️ Hohe Abwesenheitsrate von {absence_rate:.1f}%")
        
        if avg_daily_hours < 7.5:
            insights.append(f"📉 Unterdurchschnittliche Arbeitszeit: {avg_daily_hours:.1f}h/Tag")
        
        overtime_df = self.analytics.calculate_overtime()
        if not overtime_df.empty:
            total_overtime = overtime_df['overtime_hours'].sum()
            if total_overtime > total_employees * 10:
                insights.append(f"⏰ Erhebliche Überstunden: {total_overtime:.0f}h gesamt")
        
        return {
            'metrics': {
                'total_employees': total_employees,
                'total_work_hours': round(total_work_hours, 2),
                'avg_daily_hours': round(avg_daily_hours, 2),
                'absence_rate': round(absence_rate, 2),
                'trend': trend
            },
            'top_performer': top_performer,
            'insights': insights,
            'recommendation': self._generate_recommendations(absence_rate, avg_daily_hours)
        }
    
    def _generate_recommendations(self, absence_rate: float, avg_daily_hours: float) -> List[str]:
        """Generiert Handlungsempfehlungen"""
        recommendations = []
        
        if absence_rate > 10:
            recommendations.append("🎯 Abwesenheitsmanagement: Implementieren Sie ein Frühwarnsystem für Abwesenheiten")
        
        if avg_daily_hours < 7.5:
            recommendations.append("📊 Arbeitszeitoptimierung: Überprüfen Sie die Arbeitszeit-Richtlinien")
        elif avg_daily_hours > 9:
            recommendations.append("⚖️ Work-Life-Balance: Achten Sie auf Überlastung der Mitarbeiter")
        
        return recommendations
    
    def generate_detailed_report(self) -> str:
        """Generiert detaillierten Markdown-Bericht"""
        summary = self.generate_executive_summary()
        
        report = f"""
# TimeMoto Analytics Report
**Erstellt am:** {datetime.now().strftime('%d.%m.%Y %H:%M')}

## Executive Summary

### Kernmetriken
- **Mitarbeiter:** {summary['metrics']['total_employees']}
- **Gesamtarbeitszeit:** {summary['metrics']['total_work_hours']}h
- **Ø Tägliche Arbeitszeit:** {summary['metrics']['avg_daily_hours']}h
- **Abwesenheitsrate:** {summary['metrics']['absence_rate']}%
- **Trend:** {summary['metrics']['trend']}

### Top Performer
"""
        
        if summary['top_performer']:
            report += f"**{summary['top_performer']['username']}** - Produktivitätsscore: {summary['top_performer']['productivity_score']}\n"
        
        report += "\n### Wichtige Erkenntnisse\n"
        for insight in summary['insights']:
            report += f"- {insight}\n"
        
        report += "\n### Handlungsempfehlungen\n"
        for rec in summary['recommendation']:
            report += f"- {rec}\n"
        
        return report
    
    def generate_employee_report(self, username: str) -> str:
        """Generiert Mitarbeiter-spezifischen Bericht"""
        emp_data = self.analytics.df[self.analytics.df['username'] == username]
        
        if emp_data.empty:
            return f"Keine Daten für Mitarbeiter {username} gefunden."
        
        # Metriken berechnen
        total_days = emp_data['entry_date'].nunique()
        work_days = emp_data[emp_data['absence_name'].isna()]['entry_date'].nunique()
        absence_days = emp_data[emp_data['absence_name'].notna()]['entry_date'].nunique()
        avg_hours = emp_data['duration_minutes'].mean() / 60
        total_hours = emp_data['duration_minutes'].sum() / 60
        balance_hours = emp_data['balance_minutes'].sum() / 60
        
        report = f"""
# Mitarbeiter-Report: {username}
**Erstellt am:** {datetime.now().strftime('%d.%m.%Y %H:%M')}

## Übersicht
- **Erfasste Tage:** {total_days}
- **Arbeitstage:** {work_days}
- **Abwesenheitstage:** {absence_days}
- **Durchschnittliche Arbeitszeit:** {avg_hours:.1f}h
- **Gesamtarbeitszeit:** {total_hours:.1f}h
- **Saldo:** {balance_hours:.1f}h

## Abwesenheiten
"""
        
        # Abwesenheitsdetails
        absences = emp_data[emp_data['absence_name'].notna()]['absence_name'].value_counts()
        if not absences.empty:
            for reason, count in absences.items():
                report += f"- **{reason}:** {count} Tage\n"
        else:
            report += "Keine Abwesenheiten verzeichnet.\n"
        
        return report

def safe_plotly_chart(fig, title="Chart", fallback_data=None):
    """Sichere Plotly-Chart Anzeige"""
    try:
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)
            return True
        else:
            st.warning(f"⚠️ {title}: Keine Daten verfügbar")
            return False
    except Exception as e:
        st.error(f"❌ Fehler bei {title}: {str(e)}")
        if fallback_data is not None:
            st.dataframe(fallback_data, use_container_width=True)
        return False

class TimeMotoApp:
    """Hauptanwendungsklasse mit erweiterten Reporting-Funktionen"""
    
    def __init__(self):
        self.db_manager = RobustDatabaseManager()
        self.viz_manager = AdvancedVisualizationManager()
    
    def run(self):
        """Startet die erweiterte Streamlit Anwendung"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>⏰ TimeMoto Analytics Pro</h1>
            <p>Erweiterte Zeiterfassung mit KI-gestützten Insights und professionellen Auswertungen</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar Navigation mit erweiterten Optionen
        st.sidebar.title("🔧 Navigation")
        page = st.sidebar.selectbox(
            "Seite auswählen:",
            [
                "🏠 Executive Dashboard",
                "📊 Team Performance",
                "📈 Trend-Analysen", 
                "⏱️ Überstunden-Management",
                "🏥 Abwesenheits-Analyse",
                "👤 Mitarbeiter-Details",
                "📋 Produktivitäts-Matrix",
                "🎯 KPI-Tracking",
                "📄 Berichte",
                "📤 Robuster Import", 
                "📋 Daten anzeigen", 
                "⚙️ Einstellungen"
            ]
        )
        
        # Seiten-Routing
        if page == "🏠 Executive Dashboard":
            self.show_executive_dashboard()
        elif page == "📊 Team Performance":
            self.show_team_performance()
        elif page == "📈 Trend-Analysen":
            self.show_trend_analysis()
        elif page == "⏱️ Überstunden-Management":
            self.show_overtime_management()
        elif page == "🏥 Abwesenheits-Analyse":
            self.show_absence_analysis()
        elif page == "👤 Mitarbeiter-Details":
            self.show_employee_details()
        elif page == "📋 Produktivitäts-Matrix":
            self.show_productivity_matrix()
        elif page == "🎯 KPI-Tracking":
            self.show_kpi_tracking()
        elif page == "📄 Berichte":
            self.show_reports()
        elif page == "📤 Robuster Import":
            self.show_robust_import()
        elif page == "📋 Daten anzeigen":
            self.show_data_view()
        elif page == "⚙️ Einstellungen":
            self.show_settings()
    
    def show_executive_dashboard(self):
        """Zeigt Executive Dashboard"""
        st.header("🏠 Executive Dashboard")
        
        # Daten laden
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭 Keine Daten verfügbar.")
            return
        
        analytics = TimeMotoAnalytics(df)
        productivity_metrics = analytics.calculate_productivity_metrics()
        
        if productivity_metrics.empty:
            st.warning("Nicht genügend Daten für Produktivitätsanalyse")
            return
        
        # Produktivitäts-Scatter
        fig = px.scatter(
            productivity_metrics,
            x='consistency_score',
            y='punctuality_score',
            size='avg_daily_hours',
            color='productivity_score',
            hover_name='username',
            title="Produktivitäts-Matrix",
            labels={
                'consistency_score': 'Konsistenz-Score',
                'punctuality_score': 'Pünktlichkeits-Score'
            },
            color_continuous_scale='Viridis'
        )
        
        # Quadranten hinzufügen
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.add_vline(x=50, line_dash="dash", line_color="gray")
        
        # Quadranten-Labels
        fig.add_annotation(x=75, y=75, text="High Performer", showarrow=False)
        fig.add_annotation(x=25, y=75, text="Pünktlich aber inkonsistent", showarrow=False)
        fig.add_annotation(x=75, y=25, text="Konsistent aber unpünktlich", showarrow=False)
        fig.add_annotation(x=25, y=25, text="Verbesserungsbedarf", showarrow=False)
        
        safe_plotly_chart(fig, "Produktivitäts-Matrix")
        
        # Top und Bottom Performer
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 Top 5 Performer")
            top_performers = productivity_metrics.nlargest(5, 'productivity_score')[
                ['username', 'productivity_score', 'avg_daily_hours']
            ]
            st.dataframe(top_performers, use_container_width=True)
        
        with col2:
            st.subheader("📉 Verbesserungspotential")
            bottom_performers = productivity_metrics.nsmallest(5, 'productivity_score')[
                ['username', 'productivity_score', 'avg_daily_hours']
            ]
            st.dataframe(bottom_performers, use_container_width=True)
        
        # Detaillierte Metriken
        st.subheader("📊 Detaillierte Produktivitätsmetriken")
        
        # Sortierbare Tabelle
        st.dataframe(
            productivity_metrics.style.background_gradient(
                subset=['productivity_score', 'consistency_score', 'punctuality_score'],
                cmap='RdYlGn'
            ),
            use_container_width=True,
            height=400
        )
    
    def show_kpi_tracking(self):
        """KPI-Tracking Dashboard"""
        st.header("🎯 KPI-Tracking")
        
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭 Keine Daten verfügbar.")
            return
        
        analytics = TimeMotoAnalytics(df)
        
        # KPI-Definitionen
        st.subheader("📏 KPI-Definitionen")
        
        kpi_definitions = {
            "Anwesenheitsrate": "Prozentsatz der geplanten Arbeitstage ohne Abwesenheit",
            "Durchschnittliche Arbeitszeit": "Mittlere tägliche Arbeitszeit aller Mitarbeiter",
            "Überstundenquote": "Prozentsatz der Arbeit über 8 Stunden/Tag",
            "Pünktlichkeitsrate": "Prozentsatz der Tage mit normalem Arbeitsbeginn",
            "Team-Effizienz": "Verhältnis von produktiver Zeit zu Anwesenheitszeit"
        }
        
        with st.expander("📖 KPI-Erklärungen"):
            for kpi, definition in kpi_definitions.items():
                st.write(f"**{kpi}:** {definition}")
        
        # KPI-Berechnung
        total_days = df['entry_date'].nunique()
        absence_days = df[df['absence_name'].notna()]['entry_date'].nunique()
        anwesenheitsrate = ((total_days - absence_days) / total_days * 100) if total_days > 0 else 0
        
        avg_work_time = analytics.df['duration_minutes'].mean() / 60 if not analytics.df.empty else 0
        
        overtime_days = df[analytics.df['duration_minutes'] > 480]['entry_date'].nunique()
        überstundenquote = (overtime_days / total_days * 100) if total_days > 0 else 0
        
        normal_starts = analytics.df[analytics.df['time_type'] == 'Normal']['entry_date'].nunique()
        pünktlichkeitsrate = (normal_starts / total_days * 100) if total_days > 0 else 0
        
        # KPI-Dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=anwesenheitsrate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Anwesenheitsrate"},
                delta={'reference': 95},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            safe_plotly_chart(fig, "Anwesenheitsrate KPI")
        
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=avg_work_time,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Ø Arbeitszeit (h)"},
                delta={'reference': 8},
                gauge={
                    'axis': {'range': [0, 12]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 6], 'color': "lightgray"},
                        {'range': [6, 9], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 8
                    }
                }
            ))
            fig.update_layout(height=300)
            safe_plotly_chart(fig, "Arbeitszeit KPI")
        
        with col3:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pünktlichkeitsrate,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Pünktlichkeitsrate"},
                delta={'reference': 90},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkorange"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig.update_layout(height=300)
            safe_plotly_chart(fig, "Pünktlichkeit KPI")
        
        # KPI-Verlauf
        st.subheader("📈 KPI-Entwicklung")
        
        # Wöchentliche KPI-Entwicklung berechnen
        weekly_kpis = []
        for week_start in pd.date_range(df['entry_date'].min(), df['entry_date'].max(), freq='W'):
            week_end = week_start + timedelta(days=6)
            week_data = df[(df['entry_date'] >= week_start) & (df['entry_date'] <= week_end)]
            
            if not week_data.empty:
                week_analytics = TimeMotoAnalytics(week_data)
                week_total_days = week_data['entry_date'].nunique()
                week_absence_days = week_data[week_data['absence_name'].notna()]['entry_date'].nunique()
                week_anwesenheit = ((week_total_days - week_absence_days) / week_total_days * 100) if week_total_days > 0 else 0
                
                week_avg_time = week_analytics.df['duration_minutes'].mean() / 60 if not week_analytics.df.empty else 0
                
                weekly_kpis.append({
                    'week': week_start,
                    'anwesenheitsrate': week_anwesenheit,
                    'avg_arbeitszeit': week_avg_time
                })
        
        if weekly_kpis:
            kpi_df = pd.DataFrame(weekly_kpis)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Anwesenheitsrate", "Durchschnittliche Arbeitszeit"),
                vertical_spacing=0.15
            )
            
            fig.add_trace(
                go.Scatter(
                    x=kpi_df['week'],
                    y=kpi_df['anwesenheitsrate'],
                    mode='lines+markers',
                    name='Anwesenheitsrate',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=kpi_df['week'],
                    y=kpi_df['avg_arbeitszeit'],
                    mode='lines+markers',
                    name='Arbeitszeit',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
            
            fig.update_yaxes(title_text="Prozent", row=1, col=1)
            fig.update_yaxes(title_text="Stunden", row=2, col=1)
            fig.update_xaxes(title_text="Woche", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=False)
            safe_plotly_chart(fig, "KPI Verlauf")
    
    def show_reports(self):
        """Berichts-Center"""
        st.header("📄 Berichts-Center")
        
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭 Keine Daten verfügbar.")
            return
        
        analytics = TimeMotoAnalytics(df)
        report_gen = ReportGenerator(analytics)
        
        # Report-Typ auswählen
        report_type = st.selectbox(
            "Bericht auswählen:",
            ["Executive Summary", "Detaillierter Analysebericht", "Mitarbeiter-Report", "Team-Performance-Report"]
        )
        
        # Zeitraum wählen
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Von:", df['entry_date'].min())
        with col2:
            end_date = st.date_input("Bis:", df['entry_date'].max())
        
        # Report generieren
        if st.button("📊 Bericht generieren", type="primary"):
            
            # Daten filtern
            filtered_df = df[(df['entry_date'] >= start_date) & (df['entry_date'] <= end_date)]
            
            if filtered_df.empty:
                st.warning("Keine Daten im gewählten Zeitraum")
                return
            
            filtered_analytics = TimeMotoAnalytics(filtered_df)
            filtered_report_gen = ReportGenerator(filtered_analytics)
            
            if report_type == "Executive Summary":
                summary = filtered_report_gen.generate_executive_summary()
                
                # Summary anzeigen
                st.markdown("### 📊 Executive Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mitarbeiter", summary['metrics']['total_employees'])
                    st.metric("Gesamtstunden", f"{summary['metrics']['total_work_hours']:.0f}h")
                
                with col2:
                    st.metric("Ø Tagesstunden", f"{summary['metrics']['avg_daily_hours']:.1f}h")
                    st.metric("Abwesenheitsrate", f"{summary['metrics']['absence_rate']:.1f}%")
                
                # Insights
                st.markdown("### 💡 Wichtige Erkenntnisse")
                for insight in summary['insights']:
                    st.info(insight)
                
                # Empfehlungen
                st.markdown("### 🎯 Handlungsempfehlungen")
                for rec in summary['recommendation']:
                    st.success(rec)
                
                # Download-Option
                report_content = filtered_report_gen.generate_detailed_report()
                st.download_button(
                    label="📥 Report als Markdown herunterladen",
                    data=report_content,
                    file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            
            elif report_type == "Detaillierter Analysebericht":
                # Umfassender Bericht mit allen Analysen
                st.markdown("### 📊 Detaillierter Analysebericht")
                
                # Verschiedene Analysen durchführen
                tabs = st.tabs(["Übersicht", "Produktivität", "Abwesenheiten", "Überstunden", "Trends"])
                
                with tabs[0]:
                    summary = filtered_report_gen.generate_executive_summary()
                    st.write(filtered_report_gen.generate_detailed_report())
                
                with tabs[1]:
                    productivity = filtered_analytics.calculate_productivity_metrics()
                    if not productivity.empty:
                        st.dataframe(productivity, use_container_width=True)
                    else:
                        st.info("Keine Produktivitätsdaten verfügbar")
                
                with tabs[2]:
                    absence = filtered_analytics.analyze_absence_patterns()
                    if absence['insights']:
                        for insight in absence['insights']:
                            st.info(insight)
                    
                    if not absence['reason_pattern'].empty:
                        st.bar_chart(absence['reason_pattern'])
                
                with tabs[3]:
                    overtime = filtered_analytics.calculate_overtime()
                    if not overtime.empty:
                        st.metric("Gesamt-Überstunden", f"{overtime['overtime_hours'].sum():.0f}h")
                        st.line_chart(overtime.groupby('parsed_date')['overtime_hours'].sum())
                
                with tabs[4]:
                    predictions = filtered_analytics.predict_future_workload(14)
                    if not predictions.empty:
                        st.line_chart(predictions.groupby('date')['predicted_hours'].sum())
            
            elif report_type == "Mitarbeiter-Report":
                # Mitarbeiter-spezifischer Report
                selected_employee = st.selectbox(
                    "Mitarbeiter wählen:",
                    sorted(filtered_df['username'].unique())
                )
                
                if selected_employee:
                    report_content = filtered_report_gen.generate_employee_report(selected_employee)
                    
                    # Report anzeigen
                    st.markdown(report_content)
                    
                    # Download-Option
                    st.download_button(
                        label="📥 Report herunterladen",
                        data=report_content,
                        file_name=f"mitarbeiter_report_{selected_employee}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )
            
            elif report_type == "Team-Performance-Report":
                # Team-Performance Report
                st.markdown("### 📊 Team-Performance Report")
                
                # Team-Metriken
                workload = filtered_analytics.calculate_team_workload_distribution()
                
                if not workload.empty:
                    st.subheader("Arbeitsverteilung")
                    st.dataframe(workload[['username', 'total_hours', 'workload_share']], use_container_width=True)
                    
                    # Performance-Metriken
                    productivity = filtered_analytics.calculate_productivity_metrics()
                    if not productivity.empty:
                        st.subheader("Produktivitäts-Scores")
                        st.dataframe(productivity, use_container_width=True)
                
                # Export-Option
                export_data = {
                    'workload': workload.to_dict() if not workload.empty else {},
                    'productivity': productivity.to_dict() if not productivity.empty else {}
                }
                
                st.download_button(
                    label="📥 Daten als JSON exportieren",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"team_performance_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
    
    def show_robust_import(self):
        """Zeigt die robuste Import-Seite"""
        st.header("📤 Robuster Datenimport mit Duplikatskontrolle")
        
        # Datenbankverbindung testen
        success, message = self.db_manager.test_connection()
        if not success:
            st.error(message)
            return
        
        # Tabellen sicherstellen
        if not self.db_manager.ensure_tables():
            st.error("❌ Datenbankschema konnte nicht erstellt werden")
            return
        
        # Import-Manager initialisieren
        import_manager = ImportManager(self.db_manager)
        
        # Import-Strategie wählen
        st.subheader("🎯 Import-Strategie")
        
        strategy = st.radio(
            "Wie sollen Duplikate behandelt werden?",
            [
                ("skip_duplicates", "🚫 Duplikate überspringen (Standard)"),
                ("update_duplicates", "🔄 Duplikate aktualisieren"),
                ("error_on_duplicates", "❌ Bei Duplikaten Fehler anzeigen")
            ],
            format_func=lambda x: x[1],
            index=0
        )
        
        strategy_value = strategy[0]
        
        # Strategie-Erklärung
        strategy_descriptions = {
            "skip_duplicates": "Bereits vorhandene Einträge werden übersprungen. Sicherste Option.",
            "update_duplicates": "Bereits vorhandene Einträge werden mit neuen Daten überschrieben.",
            "error_on_duplicates": "Import wird abgebrochen wenn Duplikate gefunden werden."
        }
        
        st.info(f"💡 **{strategy_descriptions[strategy_value]}**")
        
        # Datei-Upload
        st.subheader("📁 Datei hochladen")
        
        uploaded_file = st.file_uploader(
            "TimeMoto Export-Datei auswählen",
            type=['xlsx', 'xls', 'csv'],
            help="Unterstützte Formate: Excel (.xlsx, .xls) und CSV (.csv)"
        )
        
        if uploaded_file is not None:
            
            # Datei-Informationen anzeigen
            file_info = {
                'name': uploaded_file.name,
                'size': len(uploaded_file.getvalue()),
                'type': uploaded_file.type
            }
            
            st.markdown(f"""
            <div class="import-status">
                <h4>📋 Datei-Informationen</h4>
                <ul>
                    <li><strong>Name:</strong> {file_info['name']}</li>
                    <li><strong>Größe:</strong> {file_info['size']:,} Bytes</li>
                    <li><strong>Typ:</strong> {file_info['type']}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            try:
                # Datei laden
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Datei erfolgreich geladen: {len(df)} Zeilen gefunden")
                
                # Datenvorschau
                with st.expander("👀 Datenvorschau", expanded=False):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Import starten
                if st.button("🚀 **ROBUSTEN IMPORT STARTEN**", type="primary", use_container_width=True):
                    
                    # Import durchführen
                    with st.spinner("🔄 Führe robusten Import durch..."):
                        result = import_manager.validate_and_import(df, strategy_value)
                    
                    # Ergebnisse anzeigen
                    if result['success']:
                        st.markdown(f"""
                        <div class="success-message">
                            <h4>🎉 Import erfolgreich abgeschlossen!</h4>
                            <ul>
                                <li><strong>Session ID:</strong> {result['session_id']}</li>
                                <li><strong>Verarbeitete Zeilen:</strong> {result['processed_rows']} von {result['total_rows']}</li>
                                <li><strong>Neue Einträge:</strong> {result['inserted_rows']}</li>
                                <li><strong>Aktualisierte Einträge:</strong> {result['updated_rows']}</li>
                                <li><strong>Übersprungene Einträge:</strong> {result['skipped_rows']}</li>
                                <li><strong>Fehlerhafte Zeilen:</strong> {result['error_rows']}</li>
                                <li><strong>Gefundene Duplikate:</strong> {result['duplicates_found']}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.balloons()
                        
                    else:
                        st.markdown("""
                        <div class="error-message">
                            <h4>❌ Import fehlgeschlagen</h4>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Fehler anzeigen
                    if result.get('errors'):
                        with st.expander(f"❌ {len(result['errors'])} Fehler aufgetreten"):
                            for error in result['errors']:
                                st.error(error)
                                
            except Exception as e:
                st.error(f"❌ Fehler beim Laden der Datei: {str(e)}")
        
        # Import-Historie anzeigen
        st.subheader("📊 Import-Historie")
        
        if st.button("🔄 Historie laden"):
            history_df = import_manager.get_import_history()
            
            if not history_df.empty:
                st.dataframe(
                    history_df,
                    use_container_width=True,
                    column_config={
                        "timestamp": st.column_config.DatetimeColumn("Zeitstempel"),
                        "session_id": st.column_config.TextColumn("Session ID"),
                        "total_rows": st.column_config.NumberColumn("Gesamt"),
                        "inserted_rows": st.column_config.NumberColumn("Eingefügt"),
                        "updated_rows": st.column_config.NumberColumn("Aktualisiert"),
                        "skipped_rows": st.column_config.NumberColumn("Übersprungen"),
                        "error_rows": st.column_config.NumberColumn("Fehler")
                    }
                )
            else:
                st.info("Noch keine Import-Historie vorhanden.")
    
    def show_data_view(self):
        """Datenansicht"""
        st.header("📋 Zeiterfassungsdaten")
        
        # Filter-Optionen
        col1, col2, col3 = st.columns(3)
        
        with col1:
            limit = st.selectbox("📊 Anzahl Einträge:", [50, 100, 200, 500, 1000, 5000], index=1)
        
        with col2:
            username_filter = st.selectbox(
                "👤 Mitarbeiter:",
                ["Alle"] + sorted(self.db_manager.get_time_entries(limit=1000)['username'].unique().tolist())
            )
        
        with col3:
            if st.button("🔄 Daten aktualisieren"):
                st.rerun()
        
        # Datumsfilter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Von:", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("Bis:", datetime.now())
        
        # Daten laden
        username_param = None if username_filter == "Alle" else username_filter
        df = self.db_manager.get_time_entries(
            limit=limit,
            username=username_param,
            start_date=start_date,
            end_date=end_date
        )
        
        if not df.empty:
            st.info(f"📈 {len(df)} Einträge gefunden")
            
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "entry_date": st.column_config.DateColumn("Datum"),
                    "username": st.column_config.TextColumn("Mitarbeiter"),
                    "start_time": st.column_config.TextColumn("Start"),
                    "end_time": st.column_config.TextColumn("Ende"),
                    "total_duration": st.column_config.TextColumn("Dauer"),
                    "balance": st.column_config.TextColumn("Saldo"),
                    "absence_name": st.column_config.TextColumn("Abwesenheit")
                }
            )
            
            # Export-Option
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Als CSV exportieren"):
                    try:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="💾 CSV herunterladen",
                            data=csv,
                            file_name=f"zeiterfassung_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Fehler beim CSV-Export: {str(e)}")
            
            with col2:
                if st.button("📥 Als Excel exportieren"):
                    try:
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Zeiterfassung')
                        
                        st.download_button(
                            label="💾 Excel herunterladen",
                            data=output.getvalue(),
                            file_name=f"zeiterfassung_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Fehler beim Excel-Export: {str(e)}")
        else:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
    
    def show_settings(self):
        """Einstellungen"""
        st.header("⚙️ Einstellungen")
        
        # Datenbankstatus
        st.subheader("🗄️ Datenbankverbindung")
        
        success, message = self.db_manager.test_connection()
        
        if success:
            st.success(message)
            
            # Statistiken anzeigen
            stats = self.db_manager.get_statistics()
            if 'error' not in stats:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📊 Einträge in DB", stats.get('total_entries', 0))
                    st.metric("👥 Mitarbeiter", stats.get('total_users', 0))
                
                with col2:
                    st.metric("🏗️ Tabellen", "✅" if stats.get('table_exists') else "❌")
                    st.metric("🏥 Abwesenheiten", stats.get('total_absences', 0))
        else:
            st.error(message)
        
        # Diagnose-Tools
        st.subheader("🔧 Diagnose-Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏗️ Tabellen neu erstellen"):
                with st.spinner("Erstelle Tabellen..."):
                    success = self.db_manager.ensure_tables()
                
                if success:
                    st.success("✅ Tabellen erfolgreich erstellt/aktualisiert")
                else:
                    st.error("❌ Fehler beim Erstellen der Tabellen")
        
        with col2:
            if st.button("📊 Statistiken aktualisieren"):
                st.rerun()
        
        # Erweiterte Einstellungen
        st.subheader("🎨 Anzeige-Einstellungen")
        
        col1, col2 = st.columns(2)
        with col1:
            standard_hours = st.number_input(
                "Standard-Arbeitsstunden pro Tag",
                min_value=1.0,
                max_value=12.0,
                value=8.0,
                step=0.5,
                help="Wird für Überstunden-Berechnung verwendet"
            )
            st.session_state['standard_hours'] = standard_hours
        
        with col2:
            cost_per_hour = st.number_input(
                "Stundensatz für Kostenberechnung (€)",
                min_value=0.0,
                max_value=200.0,
                value=35.0,
                step=5.0,
                help="Wird für Überstunden-Kostenberechnung verwendet"
            )
            st.session_state['cost_per_hour'] = cost_per_hour
        
        # Konfigurationshilfe
        st.subheader("🔧 Konfiguration")
        
        st.markdown("""
        <div class="analytics-section">
            <h4>📝 Neon.tech Konfiguration</h4>
            <p><strong>Für secrets.toml:</strong></p>
            <pre>
[secrets]
DATABASE_URL = "postgresql://user:pass@host-pooler.region.aws.neon.tech/db?sslmode=require"
            </pre>
            
            <p><strong>Für Streamlit Cloud:</strong> Setzen Sie DATABASE_URL in den App-Secrets.</p>
            
            <h5>🔍 Wichtige Hinweise:</h5>
            <ul>
                <li>Verwenden Sie den Pooler-Endpoint (-pooler in der URL)</li>
                <li>Entfernen Sie channel_binding=require</li>
                <li>Nutzen Sie nur sslmode=require</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Export/Import von Einstellungen
        st.subheader("💾 Daten-Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Alle Daten löschen", type="secondary"):
                st.warning("⚠️ Diese Aktion kann nicht rückgängig gemacht werden!")
                if st.button("❌ Wirklich alle Daten löschen?"):
                    conn = self.db_manager.get_connection()
                    if conn:
                        try:
                            with conn:
                                with conn.cursor() as cur:
                                    cur.execute("TRUNCATE TABLE time_entries, import_sessions RESTART IDENTITY CASCADE")
                                conn.commit()
                                st.success("✅ Alle Daten wurden gelöscht")
                                time.sleep(2)
                                st.rerun()
                        except Exception as e:
                            st.error(f"Fehler beim Löschen: {e}")
                        finally:
                            conn.close()
        
        with col2:
            if st.button("📊 Datenbank-Backup erstellen"):
                conn = self.db_manager.get_connection()
                if conn:
                    try:
                        # Alle Daten exportieren
                        df = pd.read_sql("SELECT * FROM time_entries", conn)
                        
                        # Als Excel speichern
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='Backup')
                        
                        st.download_button(
                            label="💾 Backup herunterladen",
                            data=output.getvalue(),
                            file_name=f"timemoto_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    except Exception as e:
                        st.error(f"Fehler beim Backup: {e}")
                    finally:
                        conn.close()

# Anwendung starten
if __name__ == "__main__":
    app = TimeMotoApp()
    
    stats = app.db_manager.get_statistics()
    if stats.get('total_entries', 0) == 0:
        st.warning("Keine Daten verfügbar. Bitte importieren Sie zuerst Daten.")
    else:
        app.run()
        
        analytics = TimeMotoAnalytics(df)
        report_gen = ReportGenerator(analytics)
        summary = report_gen.generate_executive_summary()
        
        # KPI-Karten
        st.subheader("📊 Kern-KPIs")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="stat-value">{summary['metrics']['total_employees']}</div>
                <div class="stat-label">Aktive Mitarbeiter</div>
                <div class="metric-trend">👥 Team-Größe</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="stat-value">{summary['metrics']['total_work_hours']:.0f}h</div>
                <div class="stat-label">Gesamtarbeitszeit</div>
                <div class="metric-trend trend-up">↑ {summary['metrics']['trend']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="stat-value">{summary['metrics']['avg_daily_hours']:.1f}h</div>
                <div class="stat-label">Ø Tägliche Arbeitszeit</div>
                <div class="metric-trend">📈 Durchschnitt</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            absence_class = "trend-down" if summary['metrics']['absence_rate'] > 10 else "trend-up"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="stat-value">{summary['metrics']['absence_rate']:.1f}%</div>
                <div class="stat-label">Abwesenheitsrate</div>
                <div class="metric-trend {absence_class}">🏥 Gesundheit</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Insights
        if summary['insights']:
            st.subheader("💡 Wichtige Erkenntnisse")
            for insight in summary['insights']:
                st.markdown(f"""
                <div class="insight-card">
                    {insight}
                </div>
                """, unsafe_allow_html=True)
        
        # Visualisierungen
        st.subheader("📈 Übersichts-Visualisierungen")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Arbeitszeit-Trend
            daily_data = analytics.get_daily_analysis()
            if not daily_data.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=daily_data['Datum'],
                    y=daily_data['Gesamt_Arbeitszeit_Std'],
                    mode='lines+markers',
                    name='Arbeitszeit',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)'
                ))
                
                # Trendlinie hinzufügen
                if len(daily_data) > 1:
                    z = np.polyfit(range(len(daily_data)), daily_data['Gesamt_Arbeitszeit_Std'], 1)
                    p = np.poly1d(z)
                    fig.add_trace(go.Scatter(
                        x=daily_data['Datum'],
                        y=p(range(len(daily_data))),
                        mode='lines',
                        name='Trend',
                        line=dict(dash='dash', color='red')
                    ))
                
                fig.update_layout(
                    title="Arbeitszeit-Trend",
                    xaxis_title="Datum",
                    yaxis_title="Stunden",
                    height=400,
                    showlegend=True
                )
                safe_plotly_chart(fig, "Arbeitszeit-Trend")
        
        with col2:
            # Mitarbeiter-Verteilung
            emp_summary = analytics.get_employee_summary()
            if not emp_summary.empty:
                fig = px.pie(
                    emp_summary.nlargest(10, 'Arbeitszeit_Stunden'),
                    values='Arbeitszeit_Stunden',
                    names='Mitarbeiter',
                    title="Top 10 Mitarbeiter nach Arbeitszeit"
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(height=400)
                safe_plotly_chart(fig, "Mitarbeiter-Verteilung")
    
    def show_team_performance(self):
        """Team Performance Dashboard"""
        st.header("📊 Team Performance Dashboard")
        
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭 Keine Daten verfügbar.")
            return
        
        analytics = TimeMotoAnalytics(df)
        
        # Team-Metriken
        workload_dist = analytics.calculate_team_workload_distribution()
        
        st.subheader("⚖️ Arbeitsverteilung im Team")
        
        # Gini-Koeffizient anzeigen
        if not workload_dist.empty:
            gini = workload_dist['team_gini_coefficient'].iloc[0]
            interpretation = "sehr gut" if gini < 0.3 else "gut" if gini < 0.5 else "verbesserungswürdig"
            
            st.info(f"📊 **Gini-Koeffizient:** {gini:.3f} - Arbeitsverteilung ist {interpretation}")
        
        # Treemap für Arbeitsverteilung
        if not workload_dist.empty:
            fig = self.viz_manager.create_treemap(
                workload_dist,
                path=['username'],
                values='total_hours',
                title="Arbeitszeit-Verteilung im Team"
            )
            safe_plotly_chart(fig, "Arbeitsverteilung Treemap")
        
        # Box-Plot für Arbeitszeit-Varianz
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self.viz_manager.create_box_plot(
                analytics.df,
                x_col='username',
                y_col='duration_minutes',
                title="Arbeitszeit-Varianz pro Mitarbeiter"
            )
            if fig:
                fig.update_yaxis(title="Minuten")
                fig.update_xaxis(tickangle=45)
            safe_plotly_chart(fig, "Arbeitszeit-Varianz")
        
        with col2:
            # Performance-Radar
            productivity_metrics = analytics.calculate_productivity_metrics()
            if not productivity_metrics.empty:
                top_performers = productivity_metrics.nlargest(5, 'productivity_score')
                
                fig = self.viz_manager.create_radar_chart(
                    top_performers,
                    ['consistency_score', 'punctuality_score', 'productivity_score'],
                    "Top 5 Performer - Vergleich"
                )
                safe_plotly_chart(fig, "Performance Radar")
    
    def show_trend_analysis(self):
        """Trend-Analysen"""
        st.header("📈 Trend-Analysen")
        
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭 Keine Daten verfügbar.")
            return
        
        analytics = TimeMotoAnalytics(df)
        
        # Zeitraum-Auswahl
        col1, col2 = st.columns(2)
        with col1:
            analysis_period = st.selectbox(
                "Analyse-Zeitraum:",
                ["Letzte 30 Tage", "Letzte 90 Tage", "Letztes Jahr", "Gesamt"]
            )
        
        # Heatmap für Anwesenheitsmuster
        st.subheader("🗓️ Anwesenheitsmuster")
        
        # Daten für Heatmap vorbereiten
        heatmap_data = analytics.df.copy()
        heatmap_data['hour'] = pd.to_datetime(heatmap_data['start_time'], format='%H:%M', errors='coerce').dt.hour
        heatmap_data['weekday'] = heatmap_data['parsed_date'].dt.day_name()
        
        attendance_pivot = heatmap_data.groupby(['weekday', 'hour']).size().reset_index(name='count')
        
        if not attendance_pivot.empty:
            fig = self.viz_manager.create_heatmap(
                attendance_pivot,
                x_col='hour',
                y_col='weekday',
                z_col='count',
                title="Anwesenheitsmuster nach Wochentag und Stunde"
            )
            safe_plotly_chart(fig, "Anwesenheits-Heatmap")
        
        # Vorhersage
        st.subheader("🔮 Arbeitsbelastungs-Vorhersage")
        
        predictions = analytics.predict_future_workload(days_ahead=14)
        if not predictions.empty:
            fig = px.line(
                predictions.groupby('date')['predicted_hours'].sum().reset_index(),
                x='date',
                y='predicted_hours',
                title="Vorhergesagte Gesamtarbeitszeit (nächste 14 Tage)"
            )
            fig.add_hline(
                y=analytics.df['duration_minutes'].sum() / 60 / analytics.df['parsed_date'].nunique(),
                line_dash="dash",
                annotation_text="Historischer Durchschnitt"
            )
            safe_plotly_chart(fig, "Arbeitsbelastungs-Vorhersage")
    
    def show_overtime_management(self):
        """Überstunden-Management"""
        st.header("⏱️ Überstunden-Management")
        
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭 Keine Daten verfügbar.")
            return
        
        analytics = TimeMotoAnalytics(df)
        
        # Überstunden berechnen
        overtime_df = analytics.calculate_overtime()
        
        if not overtime_df.empty:
            # Gesamt-Überstunden
            total_overtime = overtime_df['overtime_hours'].sum()
            avg_overtime_per_person = overtime_df.groupby('username')['overtime_hours'].sum().mean()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("⏰ Gesamt-Überstunden", f"{total_overtime:.0f}h")
            
            with col2:
                st.metric("📊 Ø pro Mitarbeiter", f"{avg_overtime_per_person:.1f}h")
            
            with col3:
                cost_per_hour = st.number_input("💶 Stundensatz", value=35.0, min_value=0.0, step=1.0)
                st.metric("💰 Geschätzte Kosten", f"{total_overtime * cost_per_hour:.0f}€")
            
            # Überstunden-Trend
            weekly_overtime = overtime_df.groupby('parsed_date')['overtime_hours'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=weekly_overtime['parsed_date'],
                y=weekly_overtime['overtime_hours'],
                name='Überstunden',
                marker_color='orange'
            ))
            
            fig.update_layout(
                title="Wöchentliche Überstunden-Entwicklung",
                xaxis_title="Woche",
                yaxis_title="Überstunden",
                showlegend=False
            )
            
            safe_plotly_chart(fig, "Überstunden-Trend")
            
            # Top Überstunden-Verursacher
            st.subheader("🎯 Mitarbeiter mit den meisten Überstunden")
            
            overtime_by_user = overtime_df.groupby('username')['overtime_hours'].sum().reset_index()
            overtime_by_user = overtime_by_user.sort_values('overtime_hours', ascending=False).head(10)
            
            fig = px.bar(
                overtime_by_user,
                x='overtime_hours',
                y='username',
                orientation='h',
                title="Top 10 Mitarbeiter nach Überstunden",
                color='overtime_hours',
                color_continuous_scale='Reds'
            )
            
            safe_plotly_chart(fig, "Top Überstunden")
            
            # Warnung bei kritischen Überstunden
            critical_overtime = overtime_by_user[overtime_by_user['overtime_hours'] > 50]
            if not critical_overtime.empty:
                st.markdown("""
                <div class="warning-card">
                    <h4>⚠️ Kritische Überstunden-Belastung</h4>
                    <p>Folgende Mitarbeiter haben mehr als 50 Überstunden:</p>
                </div>
                """, unsafe_allow_html=True)
                
                for _, emp in critical_overtime.iterrows():
                    st.warning(f"• {emp['username']}: {emp['overtime_hours']:.0f} Stunden")
    
    def show_absence_analysis(self):
        """Abwesenheits-Analyse"""
        st.header("🏥 Abwesenheits-Analyse")
        
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭 Keine Daten verfügbar.")
            return
        
        analytics = TimeMotoAnalytics(df)
        absence_analysis = analytics.analyze_absence_patterns()
        
        # Insights anzeigen
        if absence_analysis['insights']:
            st.subheader("💡 Erkenntnisse")
            for insight in absence_analysis['insights']:
                st.info(insight)
        
        # Visualisierungen
        col1, col2 = st.columns(2)
        
        with col1:
            # Abwesenheit nach Wochentag
            if not absence_analysis['weekday_pattern'].empty:
                fig = px.bar(
                    x=absence_analysis['weekday_pattern'].index,
                    y=absence_analysis['weekday_pattern'].values,
                    title="Abwesenheiten nach Wochentag",
                    labels={'x': 'Wochentag', 'y': 'Anzahl'}
                )
                safe_plotly_chart(fig, "Wochentags-Abwesenheiten")
        
        with col2:
            # Abwesenheitsgründe
            if not absence_analysis['reason_pattern'].empty:
                fig = px.pie(
                    values=absence_analysis['reason_pattern'].values,
                    names=absence_analysis['reason_pattern'].index,
                    title="Abwesenheitsgründe"
                )
                safe_plotly_chart(fig, "Abwesenheitsgründe")
        
        # Monatlicher Verlauf
        if not absence_analysis['monthly_pattern'].empty:
            fig = px.line(
                x=absence_analysis['monthly_pattern'].index,
                y=absence_analysis['monthly_pattern'].values,
                title="Abwesenheiten nach Monat",
                markers=True
            )
            fig.update_layout(xaxis_title="Monat", yaxis_title="Anzahl Abwesenheiten")
            safe_plotly_chart(fig, "Monatliche Abwesenheiten")
        
        # Bradford-Faktor berechnen
        st.subheader("📊 Bradford-Faktor Analyse")
        st.info("Der Bradford-Faktor identifiziert Mitarbeiter mit häufigen kurzen Abwesenheiten")
        
        # Bradford-Faktor: S² × D (S = Anzahl Abwesenheitsspannen, D = Gesamttage)
        bradford_data = []
        for user in df['username'].unique():
            user_absences = df[(df['username'] == user) & (df['absence_name'].notna())]
            if not user_absences.empty:
                # Abwesenheitsspannen identifizieren
                user_absences = user_absences.sort_values('entry_date')
                spans = 1
                total_days = len(user_absences)
                
                for i in range(1, len(user_absences)):
                    if (user_absences.iloc[i]['entry_date'] - user_absences.iloc[i-1]['entry_date']).days > 1:
                        spans += 1
                
                bradford_score = spans * spans * total_days
                bradford_data.append({
                    'Mitarbeiter': user,
                    'Abwesenheitsspannen': spans,
                    'Abwesenheitstage': total_days,
                    'Bradford-Score': bradford_score
                })
        
        if bradford_data:
            bradford_df = pd.DataFrame(bradford_data).sort_values('Bradford-Score', ascending=False)
            
            fig = px.scatter(
                bradford_df,
                x='Abwesenheitsspannen',
                y='Abwesenheitstage',
                size='Bradford-Score',
                hover_name='Mitarbeiter',
                title="Bradford-Faktor Analyse",
                color='Bradford-Score',
                color_continuous_scale='Reds'
            )
            safe_plotly_chart(fig, "Bradford-Faktor")
            
            # Kritische Fälle
            critical_bradford = bradford_df[bradford_df['Bradford-Score'] > 100]
            if not critical_bradford.empty:
                st.warning("⚠️ Mitarbeiter mit kritischem Bradford-Score (>100):")
                st.dataframe(critical_bradford, use_container_width=True)
    
    def show_employee_details(self):
        """Detaillierte Mitarbeiter-Ansicht"""
        st.header("👤 Mitarbeiter-Details")
        
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭 Keine Daten verfügbar.")
            return
        
        analytics = TimeMotoAnalytics(df)
        
        # Mitarbeiter auswählen
        selected_employee = st.selectbox(
            "Mitarbeiter auswählen:",
            sorted(df['username'].unique())
        )
        
        # Mitarbeiter-Daten filtern
        employee_data = df[df['username'] == selected_employee]
        employee_analytics = TimeMotoAnalytics(employee_data)
        
        # Metriken
        col1, col2, col3, col4 = st.columns(4)
        
        total_days = employee_data['entry_date'].nunique()
        work_days = employee_data[employee_data['absence_name'].isna()]['entry_date'].nunique()
        absence_days = employee_data[employee_data['absence_name'].notna()]['entry_date'].nunique()
        avg_hours = employee_analytics.df['duration_minutes'].mean() / 60 if not employee_analytics.df.empty else 0
        
        with col1:
            st.metric("📅 Erfasste Tage", total_days)
        
        with col2:
            st.metric("💼 Arbeitstage", work_days)
        
        with col3:
            st.metric("🏥 Abwesenheitstage", absence_days)
        
        with col4:
            st.metric("⏰ Ø Arbeitszeit", f"{avg_hours:.1f}h")
        
        # Zeitverlauf
        st.subheader("📈 Arbeitszeitverlauf")
        
        daily_hours = employee_analytics.df.groupby('entry_date')['duration_minutes'].sum().reset_index()
        daily_hours['hours'] = daily_hours['duration_minutes'] / 60
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_hours['entry_date'],
            y=daily_hours['hours'],
            mode='lines+markers',
            name='Arbeitszeit',
            line=dict(color='blue', width=2)
        ))
        
        # Durchschnittslinie
        fig.add_hline(
            y=8,
            line_dash="dash",
            line_color="green",
            annotation_text="Standard (8h)"
        )
        
        fig.update_layout(
            title=f"Arbeitszeitverlauf - {selected_employee}",
            xaxis_title="Datum",
            yaxis_title="Stunden",
            height=400
        )
        
        safe_plotly_chart(fig, "Mitarbeiter Zeitverlauf")
        
        # Arbeitszeitmuster
        col1, col2 = st.columns(2)
        
        with col1:
            # Start-/Endzeiten
            time_pattern = employee_analytics.df[employee_analytics.df['start_time'].str.match(r'^\d{2}:\d{2}$', na=False)].copy()
            if not time_pattern.empty:
                time_pattern['start_hour'] = pd.to_datetime(time_pattern['start_time'], format='%H:%M').dt.hour
                
                fig = px.histogram(
                    time_pattern,
                    x='start_hour',
                    title="Arbeitsbeginn-Verteilung",
                    nbins=24,
                    labels={'start_hour': 'Stunde', 'count': 'Häufigkeit'}
                )
                safe_plotly_chart(fig, "Arbeitsbeginn")
        
        with col2:
            # Wochentags-Muster
            weekday_counts = employee_analytics.df['weekday'].value_counts()
            
            fig = px.bar(
                x=weekday_counts.index,
                y=weekday_counts.values,
                title="Anwesenheit nach Wochentag",
                labels={'x': 'Wochentag', 'y': 'Anzahl'}
            )
            safe_plotly_chart(fig, "Wochentags-Muster")
    
    def show_productivity_matrix(self):
        """Produktivitäts-Matrix"""
        st.header("📋 Produktivitäts-Matrix")
        
        df = self.db_manager.get_time_entries(limit=5000)
        if df.empty:
            st.info("📭
