import streamlit as st
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
from io import BytesIO
import re
from typing import Dict, List, Optional, Tuple
import time
import logging
import numpy as np
import hashlib
import json

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit Konfiguration
st.set_page_config(
    page_title="TimeMoto Analytics",
    page_icon="⏰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS für modernes Design
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
    
    .analytics-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
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
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    
    def get_time_entries(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Holt Zeiterfassungsdaten"""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT id, username, entry_date, start_time, end_time, 
                   breaks_duration, total_duration, duration_excluding_breaks,
                   work_schedule, balance, absence_name, remarks, 
                   created_at, updated_at
            FROM time_entries 
            ORDER BY entry_date DESC, username
            LIMIT %s OFFSET %s
            """
            df = pd.read_sql(query, conn, params=[limit, offset])
            return df
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Daten: {e}")
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
    
    def validate_and_import(self, df: pd.DataFrame, import_strategy: str = "skip_duplicates") -> Dict:
        """Hauptfunktion für validierten Import"""
        
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
                for _, row in df.iterrows():
                    cur.execute(
                        "SELECT id, created_at FROM time_entries WHERE username = %s AND entry_date = %s",
                        (row['Username'], row['normalized_date'])
                    )
                    
                    existing = cur.fetchone()
                    if existing:
                        existing_duplicates.append({
                            'username': row['Username'],
                            'date': row['normalized_date'],
                            'existing_id': existing[0],
                            'existing_created_at': existing[1],
                            'row_index': row.name
                        })
        
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
        """Führt den eigentlichen Import durch"""
        
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
            balance, absence_name, remarks, created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
        )
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
            updated_at = CURRENT_TIMESTAMP
        WHERE username = %s AND entry_date = %s
        """
        
        check_sql = "SELECT id FROM time_entries WHERE username = %s AND entry_date = %s"
        
        try:
            with conn:
                with conn.cursor() as cur:
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    total_rows = len(df)
                    
                    for index, row in df.iterrows():
                        try:
                            # Progress Update
                            progress = (index + 1) / total_rows
                            progress_bar.progress(progress)
                            progress_text.text(f"Verarbeite {index + 1}/{total_rows}: {row['Username']}")
                            
                            # Prüfe ob Eintrag bereits existiert
                            cur.execute(check_sql, (row['Username'], row['normalized_date']))
                            existing = cur.fetchone()
                            
                            # Datenwerte vorbereiten
                            values = (
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
                                row.get('Remarks') if pd.notna(row.get('Remarks')) else None
                            )
                            
                            if existing:
                                # Eintrag existiert bereits
                                if strategy == "skip_duplicates":
                                    result['skipped_rows'] += 1
                                    continue
                                elif strategy == "update_duplicates":
                                    # Update durchführen
                                    update_values = values[1:] + (row['Username'], row['normalized_date'])
                                    cur.execute(update_sql, update_values)
                                    result['updated_rows'] += 1
                                else:
                                    result['skipped_rows'] += 1
                                    continue
                            else:
                                # Neuen Eintrag erstellen
                                cur.execute(insert_sql, values)
                                result['inserted_rows'] += 1
                            
                        except Exception as row_error:
                            error_msg = f"Zeile {index + 1}: {str(row_error)}"
                            result['errors'].append(error_msg)
                            result['error_rows'] += 1
                            continue
                    
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
    """Hauptanwendungsklasse"""
    
    def __init__(self):
        self.db_manager = RobustDatabaseManager()
    
    def run(self):
        """Startet die Streamlit Anwendung"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>⏰ TimeMoto Analytics & Management</h1>
            <p>Robuste Zeiterfassung mit Duplikatskontrolle und erweiterten Auswertungen</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar Navigation
        st.sidebar.title("🔧 Navigation")
        page = st.sidebar.selectbox(
            "Seite auswählen:",
            [
                "🏠 Dashboard", 
                "📤 Robuster Import", 
                "👥 Mitarbeiter-Analyse", 
                "📅 Datums-Analyse",
                "📋 Daten anzeigen", 
                "⚙️ Einstellungen"
            ]
        )
        
        # Seiten-Routing
        if page == "🏠 Dashboard":
            self.show_dashboard()
        elif page == "📤 Robuster Import":
            self.show_robust_import()
        elif page == "👥 Mitarbeiter-Analyse":
            self.show_employee_analysis()
        elif page == "📅 Datums-Analyse":
            self.show_date_analysis()
        elif page == "📋 Daten anzeigen":
            self.show_data_view()
        elif page == "⚙️ Einstellungen":
            self.show_settings()
    
    def show_dashboard(self):
        """Zeigt das Dashboard"""
        st.header("🏠 Dashboard")
        
        # Datenbankverbindung testen
        success, message = self.db_manager.test_connection()
        if not success:
            st.error(message)
            return
        
        # Tabellen sicherstellen
        if not self.db_manager.ensure_tables():
            st.error("❌ Datenbankschema konnte nicht erstellt werden")
            return
        
        # Statistiken abrufen
        stats = self.db_manager.get_statistics()
        
        if 'error' not in stats and stats.get('table_exists'):
            # Metriken anzeigen
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("📝 Gesamte Einträge", stats.get('total_entries', 0))
            
            with col2:
                st.metric("👥 Mitarbeiter", stats.get('total_users', 0))
            
            with col3:
                st.metric("🏥 Abwesenheiten", stats.get('total_absences', 0))
            
            with col4:
                if stats.get('first_date') and stats.get('last_date'):
                    range_text = f"{stats['first_date']} bis {stats['last_date']}"
                else:
                    range_text = "Keine Daten"
                st.metric("📅 Zeitraum", range_text)
            
            with col5:
                last_import = stats.get('last_import')
                if last_import:
                    import_text = last_import.strftime("%d.%m.%Y %H:%M")
                else:
                    import_text = "Noch keine Daten"
                st.metric("🔄 Letzter Import", import_text)
            
            # Visualisierungen wenn Daten vorhanden
            if stats.get('total_entries', 0) > 0:
                self.show_dashboard_charts()
        else:
            st.warning("⚠️ Keine Daten verfügbar. Importieren Sie zuerst TimeMoto-Daten.")
    
    def show_dashboard_charts(self):
        """Zeigt Dashboard-Visualisierungen"""
        try:
            df = self.db_manager.get_time_entries(limit=1000)
            
            if df.empty:
                st.info("📊 Keine Daten für Visualisierungen verfügbar")
                return
            
            analytics = TimeMotoAnalytics(df)
            
            # Visualisierungen
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    daily_data = analytics.get_daily_analysis()
                    if not daily_data.empty:
                        fig = px.bar(
                            daily_data, 
                            x='Datum', 
                            y='Gesamt_Arbeitszeit_Std',
                            title="📊 Tägliche Gesamtarbeitszeit"
                        )
                        fig.update_layout(height=400)
                        safe_plotly_chart(fig, "Tägliche Arbeitszeit")
                except Exception as e:
                    st.error(f"Fehler bei Dashboard-Chart 1: {str(e)}")
            
            with col2:
                try:
                    daily_data = analytics.get_daily_analysis()
                    if not daily_data.empty:
                        fig = px.line(
                            daily_data, 
                            x='Datum', 
                            y='Gesamt_Saldo_Std',
                            title="📈 Täglicher Gesamt-Saldo"
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(height=400)
                        safe_plotly_chart(fig, "Täglicher Saldo")
                except Exception as e:
                    st.error(f"Fehler bei Dashboard-Chart 2: {str(e)}")
                    
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Dashboard-Daten: {str(e)}")
    
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
    
    def show_employee_analysis(self):
        """Mitarbeiter-Analyse"""
        st.header("👥 Mitarbeiter-Analyse")
        
        df = self.db_manager.get_time_entries(limit=1000)
        
        if df.empty:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
            return
        
        analytics = TimeMotoAnalytics(df)
        summary = analytics.get_employee_summary()
        
        st.subheader("📊 Mitarbeiter-Übersicht")
        st.dataframe(
            summary[['Mitarbeiter', 'Arbeitstage', 'Arbeitszeit_Stunden', 'Saldo_Stunden', 'Abwesenheiten', 'Projekte']],
            use_container_width=True
        )
        
        # Visualisierungen
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    summary, 
                    x='Mitarbeiter', 
                    y='Saldo_Stunden',
                    title="⚖️ Saldo-Vergleich alle Mitarbeiter",
                    color='Saldo_Stunden',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_layout(xaxis_tickangle=45)
                safe_plotly_chart(fig, "Saldo-Vergleich")
            
            with col2:
                fig = px.scatter(
                    summary,
                    x='Arbeitszeit_Stunden',
                    y='Saldo_Stunden',
                    size='Arbeitstage',
                    hover_name='Mitarbeiter',
                    title="💼 Arbeitszeit vs. Saldo"
                )
                safe_plotly_chart(fig, "Arbeitszeit vs Saldo")
                
        except Exception as e:
            st.error(f"Fehler bei Mitarbeiter-Visualisierungen: {str(e)}")
    
    def show_date_analysis(self):
        """Datums-Analyse"""
        st.header("📅 Datums-Analyse")
        
        df = self.db_manager.get_time_entries(limit=1000)
        
        if df.empty:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
            return
        
        analytics = TimeMotoAnalytics(df)
        daily_data = analytics.get_daily_analysis()
        
        if daily_data.empty:
            st.warning("Keine Daten für Datums-Analyse verfügbar.")
            return
        
        st.info(f"📊 Analyse für Zeitraum: {daily_data['Datum'].min()} bis {daily_data['Datum'].max()}")
        
        # Tagesvergleich
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    daily_data,
                    x='Datum',
                    y='Mitarbeiter_Anzahl',
                    title="👥 Anwesende Mitarbeiter pro Tag"
                )
                safe_plotly_chart(fig, "Mitarbeiter pro Tag")
            
            with col2:
                fig = px.bar(
                    daily_data,
                    x='Datum',
                    y='Abwesenheiten',
                    title="🏥 Abwesenheiten pro Tag",
                    color='Abwesenheiten',
                    color_continuous_scale='Reds'
                )
                safe_plotly_chart(fig, "Abwesenheiten pro Tag")
                
        except Exception as e:
            st.error(f"Fehler bei Datums-Visualisierungen: {str(e)}")
        
        # Detailtabelle
        st.subheader("📋 Tägliche Übersicht")
        st.dataframe(
            daily_data[['Datum', 'Mitarbeiter_Anzahl', 'Gesamt_Arbeitszeit_Std', 'Gesamt_Saldo_Std', 'Abwesenheiten']],
            use_container_width=True
        )
    
    def show_data_view(self):
        """Datenansicht"""
        st.header("📋 Zeiterfassungsdaten")
        
        # Filter-Optionen
        col1, col2 = st.columns(2)
        
        with col1:
            limit = st.selectbox("📊 Anzahl Einträge:", [50, 100, 200, 500, 1000], index=1)
        
        with col2:
            if st.button("🔄 Daten aktualisieren"):
                st.rerun()
        
        # Daten laden
        df = self.db_manager.get_time_entries(limit)
        
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

# Anwendung starten
if __name__ == "__main__":
    app = TimeMotoApp()
    app.run()
