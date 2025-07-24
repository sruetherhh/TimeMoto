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
from typing import Dict, List, Optional
import time
import logging
import numpy as np

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
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #667eea;
        margin-bottom: 1rem;
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
    """Robuste Datenbankverbindung mit dauerhafter Speicherung"""
    
    def __init__(self):
        self.connection_string = self._get_connection_string()
        self.max_retries = 3
        self.retry_delay = 2
    
    def _get_connection_string(self) -> str:
        """Erstellt die Verbindungszeichenfolge für neon.tech"""
        # Versuche zuerst DATABASE_URL (empfohlen für Neon.tech)
        database_url = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
        
        if database_url:
            # Bereinige problematische Parameter
            clean_url = database_url.replace("&channel_binding=require", "")
            clean_url = clean_url.replace("channel_binding=require&", "")
            clean_url = clean_url.replace("?channel_binding=require", "?sslmode=require")
            
            # Stelle sicher, dass SSL aktiviert ist
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
            st.error("⚠️ Datenbankverbindung nicht konfiguriert! Bitte DATABASE_URL oder einzelne DB-Parameter in secrets.toml setzen.")
            st.stop()
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}?sslmode=require"
    
    def get_connection(self):
        """Erstellt eine robuste Datenbankverbindung mit Retry-Logik"""
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Datenbankverbindung Versuch {attempt + 1}/{self.max_retries}")
                
                conn = psycopg2.connect(
                    self.connection_string,
                    connect_timeout=30,
                    keepalives=1,
                    keepalives_idle=600,
                    keepalives_interval=30,
                    keepalives_count=3
                )
                
                # Teste die Verbindung
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    cur.fetchone()
                
                logger.info("Datenbankverbindung erfolgreich")
                return conn
                
            except psycopg2.OperationalError as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"Verbindungsversuch {attempt + 1} fehlgeschlagen, warte {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Maximale Anzahl Verbindungsversuche erreicht")
                    st.error(f"❌ Datenbankverbindung fehlgeschlagen nach {self.max_retries} Versuchen: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unerwarteter Datenbankfehler: {e}")
                st.error(f"❌ Unerwarteter Datenbankfehler: {e}")
                return None
        
        return None
    
    def create_tables_with_validation(self) -> bool:
        """Erstellt Tabellen und validiert sie"""
        create_sql = """
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
            
            -- Unique constraint für Duplikate vermeiden
            CONSTRAINT unique_user_date UNIQUE(username, entry_date)
        );
        
        -- Indices für Performance
        CREATE INDEX IF NOT EXISTS idx_time_entries_user_date 
        ON time_entries(username, entry_date);
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_date 
        ON time_entries(entry_date);
        
        CREATE INDEX IF NOT EXISTS idx_time_entries_username 
        ON time_entries(username);
        
        -- Trigger für updated_at
        CREATE OR REPLACE FUNCTION update_modified_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        DROP TRIGGER IF EXISTS update_time_entries_updated_at ON time_entries;
        CREATE TRIGGER update_time_entries_updated_at 
            BEFORE UPDATE ON time_entries 
            FOR EACH ROW EXECUTE FUNCTION update_modified_column();
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
                            cur.execute(statement)
                    
                    conn.commit()
                    logger.info("Tabellen erfolgreich erstellt")
                    
                    # Validiere Tabellenerstellung
                    cur.execute("""
                        SELECT column_name, data_type, is_nullable 
                        FROM information_schema.columns 
                        WHERE table_name = 'time_entries'
                        ORDER BY ordinal_position
                    """)
                    
                    columns = cur.fetchall()
                    if len(columns) >= 10:
                        logger.info(f"Tabelle validiert: {len(columns)} Spalten gefunden")
                        return True
                    else:
                        logger.error("Tabellenerstellung fehlgeschlagen - zu wenige Spalten")
                        return False
                        
        except Exception as e:
            logger.error(f"Fehler beim Erstellen der Tabellen: {e}")
            st.error(f"❌ Tabellenerstellung fehlgeschlagen: {str(e)}")
            return False
        finally:
            conn.close()
        
        return False
    
    def insert_time_entries_robust(self, df: pd.DataFrame) -> tuple[int, int, List[str]]:
        """Robuste Dateneinfügung mit detailliertem Logging"""
        inserted_count = 0
        updated_count = 0
        errors = []
        
        if df.empty:
            return 0, 0, ["Keine Daten zum Einfügen"]
        
        # Bereinige DataFrame
        df_clean = self._clean_dataframe(df)
        
        insert_sql = """
        INSERT INTO time_entries (
            username, entry_date, start_time, end_time, breaks_duration,
            total_duration, duration_excluding_breaks, work_schedule,
            balance, absence_name, remarks
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        ) ON CONFLICT (username, entry_date) 
        DO UPDATE SET
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            breaks_duration = EXCLUDED.breaks_duration,
            total_duration = EXCLUDED.total_duration,
            duration_excluding_breaks = EXCLUDED.duration_excluding_breaks,
            work_schedule = EXCLUDED.work_schedule,
            balance = EXCLUDED.balance,
            absence_name = EXCLUDED.absence_name,
            remarks = EXCLUDED.remarks,
            updated_at = CURRENT_TIMESTAMP
        RETURNING id, 
        (CASE WHEN created_at = updated_at THEN 'inserted' ELSE 'updated' END) as action
        """
        
        conn = self.get_connection()
        if not conn:
            return 0, 0, ["Datenbankverbindung fehlgeschlagen"]
        
        try:
            with conn:
                with conn.cursor() as cur:
                    progress_bar = st.progress(0)
                    progress_text = st.empty()
                    total_rows = len(df_clean)
                    
                    for index, row in df_clean.iterrows():
                        try:
                            # Überspringe Total-Zeilen
                            if str(row.get('Username', '')).strip().lower() == 'total':
                                continue
                            
                            # Konvertiere Datum
                            entry_date = self._parse_german_date(row.get('Date'))
                            if not entry_date:
                                errors.append(f"Zeile {index + 1}: Ungültiges Datum '{row.get('Date')}'")
                                continue
                            
                            # Bereite Daten vor
                            values = (
                                str(row.get('Username', '')).strip(),
                                entry_date,
                                self._clean_string(row.get('StartTime')),
                                self._clean_string(row.get('EndTime')),
                                self._clean_string(row.get('Breaks')),
                                self._clean_string(row.get('Duration')),
                                self._clean_string(row.get('DurationExcludingBreaks')),
                                self._clean_string(row.get('WorkSchedule')),
                                self._clean_string(row.get('Balance')),
                                self._clean_string(row.get('AbsenceName')) if pd.notna(row.get('AbsenceName')) else None,
                                self._clean_string(row.get('Remarks')) if pd.notna(row.get('Remarks')) else None
                            )
                            
                            # Führe INSERT aus
                            cur.execute(insert_sql, values)
                            result = cur.fetchone()
                            
                            if result:
                                if result[1] == 'inserted':
                                    inserted_count += 1
                                else:
                                    updated_count += 1
                            
                            # Update Progress
                            progress = (index + 1) / total_rows
                            progress_bar.progress(progress)
                            progress_text.text(f"Verarbeite Zeile {index + 1} von {total_rows}: {row.get('Username', 'Unbekannt')}")
                            
                            logger.debug(f"Zeile {index + 1} erfolgreich verarbeitet: {row.get('Username')} - {entry_date}")
                            
                        except Exception as row_error:
                            error_msg = f"Zeile {index + 1} ({row.get('Username', 'Unbekannt')}): {str(row_error)}"
                            errors.append(error_msg)
                            logger.warning(error_msg)
                            continue
                    
                    # Explizit committen
                    conn.commit()
                    progress_bar.empty()
                    progress_text.empty()
                    logger.info(f"Daten erfolgreich committet: {inserted_count} eingefügt, {updated_count} aktualisiert")
                    
                    # Validiere Einfügung
                    cur.execute("SELECT COUNT(*) FROM time_entries")
                    total_count = cur.fetchone()[0]
                    logger.info(f"Gesamtzahl Einträge in Datenbank: {total_count}")
                    
                    st.success(f"✅ **Daten erfolgreich in Datenbank gespeichert!** Gesamt: {total_count} Einträge")
                    
        except Exception as e:
            error_msg = f"Datenbankfehler beim Einfügen: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
            st.error(f"❌ {error_msg}")
        finally:
            conn.close()
        
        return inserted_count, updated_count, errors
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereinigt DataFrame vor der Einfügung"""
        df_clean = df.copy()
        
        # Entferne leere Zeilen
        df_clean = df_clean.dropna(how='all')
        
        # Fülle NaN-Werte
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].fillna('')
        
        return df_clean
    
    def _clean_string(self, value) -> str:
        """Bereinigt String-Werte"""
        if pd.isna(value):
            return ''
        return str(value).strip()
    
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
    
    def test_database_connection(self) -> tuple[bool, str, Dict]:
        """Umfassender Datenbankverbindungstest"""
        try:
            start_time = time.time()
            
            conn = self.get_connection()
            if not conn:
                return False, "❌ Verbindung fehlgeschlagen", {}
            
            try:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Grundlegende Verbindung
                    cur.execute("SELECT version()")
                    version = cur.fetchone()['version']
                    
                    # Tabellen prüfen
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = 'time_entries'
                    """)
                    tables = cur.fetchall()
                    
                    # Datenanzahl prüfen
                    if tables:
                        cur.execute("SELECT COUNT(*) as count FROM time_entries")
                        count_result = cur.fetchone()
                        entry_count = count_result['count'] if count_result else 0
                    else:
                        entry_count = 0
                    
                    # Letzter Eintrag
                    last_entry = None
                    if entry_count > 0:
                        cur.execute("""
                            SELECT username, entry_date, created_at 
                            FROM time_entries 
                            ORDER BY created_at DESC 
                            LIMIT 1
                        """)
                        last_entry = cur.fetchone()
                    
                    connection_time = round(time.time() - start_time, 2)
                    
                    info = {
                        'version': version,
                        'connection_time': connection_time,
                        'tables_exist': len(tables) > 0,
                        'entry_count': entry_count,
                        'last_entry': dict(last_entry) if last_entry else None
                    }
                    
                    return True, f"✅ Verbindung erfolgreich ({connection_time}s)", info
            finally:
                conn.close()
                    
        except Exception as e:
            return False, f"❌ Verbindung fehlgeschlagen: {str(e)}", {}
    
    def get_statistics(self) -> Dict:
        """Holt erweiterte Statistiken"""
        conn = self.get_connection()
        if not conn:
            return {'error': 'Keine Datenbankverbindung'}
        
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                stats = {}
                
                # Grundstatistiken
                cur.execute("SELECT COUNT(*) as total_entries FROM time_entries")
                result = cur.fetchone()
                stats['total_entries'] = result['total_entries'] if result else 0
                
                cur.execute("SELECT COUNT(DISTINCT username) as total_users FROM time_entries")
                result = cur.fetchone()
                stats['total_users'] = result['total_users'] if result else 0
                
                cur.execute("SELECT COUNT(*) as absences FROM time_entries WHERE absence_name IS NOT NULL AND absence_name != ''")
                result = cur.fetchone()
                stats['total_absences'] = result['absences'] if result else 0
                
                # Zeitstatistiken
                cur.execute("SELECT MIN(entry_date) as first_date, MAX(entry_date) as last_date FROM time_entries")
                result = cur.fetchone()
                if result and result['first_date']:
                    stats['date_range'] = {
                        'first_date': result['first_date'],
                        'last_date': result['last_date']
                    }
                
                # Letzter Import
                cur.execute("SELECT MAX(created_at) as last_import FROM time_entries")
                result = cur.fetchone()
                stats['last_import'] = result['last_import'] if result else None
                
                return stats
                
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Statistiken: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
    
    def get_time_entries(self, limit: int = 100, offset: int = 0) -> pd.DataFrame:
        """Holt Zeiterfassungsdaten mit Pagination"""
        conn = self.get_connection()
        if not conn:
            return pd.DataFrame()
        
        try:
            query = """
            SELECT 
                id, username, entry_date, start_time, end_time, 
                breaks_duration, total_duration, duration_excluding_breaks,
                work_schedule, balance, absence_name, remarks, 
                created_at, updated_at
            FROM time_entries 
            ORDER BY entry_date DESC, username
            LIMIT %s OFFSET %s
            """
            df = pd.read_sql(query, conn, params=[limit, offset])
            logger.info(f"Abgerufen: {len(df)} Einträge (Limit: {limit}, Offset: {offset})")
            return df
        except Exception as e:
            logger.error(f"Fehler beim Abrufen der Daten: {e}")
            st.error(f"❌ Fehler beim Laden der Daten: {str(e)}")
            return pd.DataFrame()
        finally:
            conn.close()

class TimeMotoAnalytics:
    """Erweiterte Analytics-Klasse für TimeMoto Daten"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = self.prepare_data(df)
        self.insights = {}
    
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bereitet die Daten für Analysen vor"""
        # Kopie erstellen
        df = df.copy()
        
        # Datum parsen
        df['parsed_date'] = pd.to_datetime(df['entry_date'] if 'entry_date' in df.columns else df['Date'])
        df['weekday'] = df['parsed_date'].dt.day_name()
        df['week_number'] = df['parsed_date'].dt.isocalendar().week
        
        # Arbeitscodes aus Remarks extrahieren
        df['work_code'] = df['remarks' if 'remarks' in df.columns else 'Remarks'].apply(self.extract_work_code)
        
        # Balance zu numerisch konvertieren
        balance_col = 'balance' if 'balance' in df.columns else 'Balance'
        duration_col = 'duration_excluding_breaks' if 'duration_excluding_breaks' in df.columns else 'DurationExcludingBreaks'
        
        df['balance_minutes'] = df[balance_col].apply(self.time_to_minutes)
        df['duration_minutes'] = df[duration_col].apply(self.time_to_minutes)
        
        # Arbeitszeitttyp klassifizieren
        df['time_type'] = df.apply(self.classify_time_type, axis=1)
        
        # Total-Zeilen entfernen
        username_col = 'username' if 'username' in df.columns else 'Username'
        df = df[df[username_col] != 'Total'].copy()
        
        return df
    
    def extract_work_code(self, remarks: str) -> Optional[str]:
        """Extrahiert Arbeitscodes aus Remarks"""
        if not remarks or pd.isna(remarks):
            return None
        
        # Suche nach "Arbeitscodes: XXXX"
        match = re.search(r'Arbeitscodes:\s*([^.\r\n]+)', str(remarks))
        if match:
            return match.group(1).strip()
        return None
    
    def time_to_minutes(self, time_str: str) -> int:
        """Konvertiert Zeitstring zu Minuten"""
        if not time_str or pd.isna(time_str) or time_str in ['', '-']:
            return 0
        
        try:
            # Entferne Vorzeichen
            clean_time = str(time_str).replace('+', '').replace('-', '')
            
            # Parse HH:MM Format
            if ':' in clean_time:
                parts = clean_time.split(':')
                hours = int(parts[0])
                minutes = int(parts[1])
                total_minutes = hours * 60 + minutes
                
                # Berücksichtige negatives Vorzeichen
                if str(time_str).startswith('-'):
                    total_minutes = -total_minutes
                
                return total_minutes
        except:
            pass
        
        return 0
    
    def classify_time_type(self, row) -> str:
        """Klassifiziert den Arbeitszeitttyp"""
        start_col = 'start_time' if 'start_time' in row.index else 'StartTime'
        end_col = 'end_time' if 'end_time' in row.index else 'EndTime'
        absence_col = 'absence_name' if 'absence_name' in row.index else 'AbsenceName'
        
        start_time = str(row[start_col])
        end_time = str(row[end_col])
        absence = row[absence_col]
        
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
        username_col = 'username' if 'username' in self.df.columns else 'Username'
        summary = self.df.groupby(username_col).agg({
            'parsed_date': 'count',
            'balance_minutes': 'sum',
            'duration_minutes': 'sum',
            'absence_name' if 'absence_name' in self.df.columns else 'AbsenceName': lambda x: x.notna().sum(),
            'work_code': lambda x: x.notna().sum()
        }).reset_index()
        
        summary.columns = ['Mitarbeiter', 'Arbeitstage', 'Saldo_Minuten', 'Arbeitszeit_Minuten', 'Abwesenheiten', 'Projekte']
        
        # Konvertiere Minuten zurück zu Stunden
        summary['Saldo_Stunden'] = round(summary['Saldo_Minuten'] / 60, 2)
        summary['Arbeitszeit_Stunden'] = round(summary['Arbeitszeit_Minuten'] / 60, 2)
        
        return summary
    
    def get_daily_analysis(self) -> pd.DataFrame:
        """Tägliche Analyse"""
        username_col = 'username' if 'username' in self.df.columns else 'Username'
        daily = self.df.groupby('parsed_date').agg({
            username_col: 'nunique',
            'balance_minutes': 'sum',
            'duration_minutes': 'sum',
            'absence_name' if 'absence_name' in self.df.columns else 'AbsenceName': lambda x: x.notna().sum(),
            'time_type': lambda x: (x == 'Normal').sum()
        }).reset_index()
        
        daily.columns = ['Datum', 'Mitarbeiter_Anzahl', 'Gesamt_Saldo_Min', 'Gesamt_Arbeitszeit_Min', 'Abwesenheiten', 'Normale_Erfassungen']
        daily['Gesamt_Saldo_Std'] = round(daily['Gesamt_Saldo_Min'] / 60, 2)
        daily['Gesamt_Arbeitszeit_Std'] = round(daily['Gesamt_Arbeitszeit_Min'] / 60, 2)
        
        return daily
    
    def get_work_code_analysis(self) -> pd.DataFrame:
        """Analysiert Arbeitscodes/Projekte"""
        work_codes = self.df[self.df['work_code'].notna()]
        
        if work_codes.empty:
            return pd.DataFrame()
        
        username_col = 'username' if 'username' in work_codes.columns else 'Username'
        analysis = work_codes.groupby(['work_code', username_col]).agg({
            'parsed_date': 'count',
            'duration_minutes': 'sum'
        }).reset_index()
        
        analysis.columns = ['Projekt', 'Mitarbeiter', 'Tage', 'Arbeitszeit_Minuten']
        analysis['Arbeitszeit_Stunden'] = round(analysis['Arbeitszeit_Minuten'] / 60, 2)
        
        return analysis
    
    def generate_insights(self) -> Dict:
        """Generiert automatische Insights"""
        insights = {}
        
        # Mitarbeiter mit meisten Überstunden
        summary = self.get_employee_summary()
        if not summary.empty:
            top_overtime = summary.loc[summary['Saldo_Stunden'].idxmax()]
            insights['top_overtime'] = {
                'employee': top_overtime['Mitarbeiter'],
                'hours': top_overtime['Saldo_Stunden']
            }
            
            # Mitarbeiter mit meisten Minusstunden
            bottom_overtime = summary.loc[summary['Saldo_Stunden'].idxmin()]
            insights['most_undertime'] = {
                'employee': bottom_overtime['Mitarbeiter'],
                'hours': bottom_overtime['Saldo_Stunden']
            }
        
        # Häufigste Arbeitscodes
        work_codes = self.df[self.df['work_code'].notna()]['work_code'].value_counts()
        if not work_codes.empty:
            insights['top_project'] = {
                'project': work_codes.index[0],
                'count': work_codes.iloc[0]
            }
        
        # Arbeitszeitmuster
        time_types = self.df['time_type'].value_counts()
        insights['time_patterns'] = time_types.to_dict()
        
        return insights

class TimeMotoApp:
    """Hauptanwendungsklasse mit robusten DB-Features und Analytics"""
    
    def __init__(self):
        self.db_manager = RobustDatabaseManager()
        self.analytics = None
    
    def run(self):
        """Startet die Streamlit Anwendung"""
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>⏰ TimeMoto Analytics & Management</h1>
            <p>Professionelle Zeiterfassung mit robusten Datenbank-Features und erweiterten Auswertungen</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar Navigation
        st.sidebar.title("🔧 Navigation")
        page = st.sidebar.selectbox(
            "Seite auswählen:",
            [
                "🏠 Dashboard", 
                "📤 Daten importieren", 
                "👥 Mitarbeiter-Analyse", 
                "📅 Datums-Analyse",
                "🏗️ Projekt-Analyse", 
                "🏥 Abwesenheit-Analyse",
                "⚡ Insights & KPIs",
                "📋 Daten anzeigen", 
                "⚙️ Einstellungen"
            ]
        )
        
        # Seiten-Routing
        if page == "🏠 Dashboard":
            self.show_dashboard()
        elif page == "📤 Daten importieren":
            self.show_import_page()
        elif page == "👥 Mitarbeiter-Analyse":
            self.show_employee_analysis()
        elif page == "📅 Datums-Analyse":
            self.show_date_analysis()
        elif page == "🏗️ Projekt-Analyse":
            self.show_project_analysis()
        elif page == "🏥 Abwesenheit-Analyse":
            self.show_absence_analysis()
        elif page == "⚡ Insights & KPIs":
            self.show_insights()
        elif page == "📋 Daten anzeigen":
            self.show_data_view()
        elif page == "⚙️ Einstellungen":
            self.show_settings()
    
    def show_dashboard(self):
        """Zeigt das Dashboard"""
        st.header("🏠 Dashboard")
        
        # Datenbankverbindung testen
        if not self.test_database_connection():
            return
        
        # Statistiken abrufen
        stats = self.db_manager.get_statistics()
        
        if stats and 'error' not in stats:
            # Metriken anzeigen
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric(
                    label="📝 Gesamte Einträge",
                    value=stats.get('total_entries', 0)
                )
            
            with col2:
                st.metric(
                    label="👥 Mitarbeiter",
                    value=stats.get('total_users', 0)
                )
            
            with col3:
                st.metric(
                    label="🏥 Abwesenheiten",
                    value=stats.get('total_absences', 0)
                )
            
            with col4:
                if stats.get('date_range'):
                    date_range = stats['date_range']
                    range_text = f"{date_range['first_date']} bis {date_range['last_date']}"
                else:
                    range_text = "Keine Daten"
                
                st.metric(
                    label="📅 Zeitraum",
                    value=range_text
                )
            
            with col5:
                last_import = stats.get('last_import')
                if last_import:
                    import_text = last_import.strftime("%d.%m.%Y %H:%M")
                else:
                    import_text = "Noch keine Daten"
                
                st.metric(
                    label="🔄 Letzter Import",
                    value=import_text
                )
            
            # Visualisierungen wenn Daten vorhanden
            if stats.get('total_entries', 0) > 0:
                self.show_dashboard_charts()
        else:
            st.warning("⚠️ Keine Datenbank-Statistiken verfügbar.")
    
    def show_dashboard_charts(self):
        """Zeigt Dashboard-Visualisierungen"""
        try:
            # Lade aktuelle Daten für Visualisierungen
            df = self.db_manager.get_time_entries(limit=1000)
            
            if not df.empty:
                # Erstelle Analytics-Objekt
                analytics = TimeMotoAnalytics(df)
                
                # Visualisierungen
                col1, col2 = st.columns(2)
                
                with col1:
                    # Arbeitszeit pro Tag
                    daily_data = analytics.get_daily_analysis()
                    if not daily_data.empty:
                        fig = px.bar(
                            daily_data, 
                            x='Datum', 
                            y='Gesamt_Arbeitszeit_Std',
                            title="📊 Tägliche Gesamtarbeitszeit",
                            labels={'Gesamt_Arbeitszeit_Std': 'Stunden'}
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Saldo pro Tag
                    if not daily_data.empty:
                        fig = px.line(
                            daily_data, 
                            x='Datum', 
                            y='Gesamt_Saldo_Std',
                            title="📈 Täglicher Gesamt-Saldo",
                            labels={'Gesamt_Saldo_Std': 'Saldo (Stunden)'}
                        )
                        fig.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Arbeitszeitmuster
                st.subheader("🔍 Arbeitszeitmuster")
                time_patterns = analytics.df['time_type'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        values=time_patterns.values,
                        names=time_patterns.index,
                        title="Verteilung der Arbeitszeitttypen"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top Mitarbeiter nach Arbeitszeit
                    employee_summary = analytics.get_employee_summary()
                    top_employees = employee_summary.nlargest(10, 'Arbeitszeit_Stunden')
                    
                    fig = px.bar(
                        top_employees,
                        x='Arbeitszeit_Stunden',
                        y='Mitarbeiter',
                        orientation='h',
                        title="🏆 Top 10 Mitarbeiter (Arbeitszeit)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Fehler beim Laden der Dashboard-Daten: {str(e)}")
    
    def show_import_page(self):
        """Zeigt die Import-Seite"""
        st.header("📤 Daten importieren")
        
        # Datenbankverbindung testen
        if not self.test_database_connection():
            return
        
        # Tabellen erstellen
        if not self.db_manager.create_tables_with_validation():
            st.error("❌ Fehler beim Erstellen der Datenbanktabellen!")
            return
        
        st.markdown("""
        <div class="analytics-section">
            <h4>📋 Unterstützte Dateiformate</h4>
            <ul>
                <li><strong>Excel (.xlsx, .xls):</strong> TimeMoto Export Dateien</li>
                <li><strong>CSV (.csv):</strong> Komma-getrennte Werte</li>
            </ul>
            <p><strong>Erwartete Spalten:</strong> Username, Date, StartTime, EndTime, Breaks, Duration, DurationExcludingBreaks, WorkSchedule, Balance, AbsenceName, Remarks</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Datei-Upload
        uploaded_file = st.file_uploader(
            "📁 TimeMoto Export-Datei auswählen",
            type=['xlsx', 'xls', 'csv'],
            help="Unterstützte Formate: Excel (.xlsx, .xls) und CSV (.csv)"
        )
        
        if uploaded_file is not None:
            try:
                # Datei lesen
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ Datei erfolgreich gelesen: {len(df)} Zeilen gefunden")
                
                # Datenvorschau
                with st.expander("👀 Datenvorschau", expanded=True):
                    st.dataframe(df.head(10), use_container_width=True)
                
                # Validierung
                required_columns = ['Username', 'Date', 'StartTime', 'EndTime', 'Duration']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Fehlende Spalten: {', '.join(missing_columns)}")
                    st.info("💡 Stellen Sie sicher, dass die Datei die richtige Struktur hat.")
                    return
                
                # Import-Button
                if st.button("🚀 Daten robust importieren", type="primary", use_container_width=True):
                    with st.spinner("Importiere Daten mit robustem Verfahren..."):
                        inserted, updated, errors = self.db_manager.insert_time_entries_robust(df)
                    
                    if inserted > 0 or updated > 0:
                        st.markdown(f"""
                        <div class="success-message">
                            <h4>✅ Robuster Import erfolgreich!</h4>
                            <ul>
                                <li><strong>Neue Einträge:</strong> {inserted}</li>
                                <li><strong>Aktualisierte Einträge:</strong> {updated}</li>
                                <li><strong>Gesamt verarbeitet:</strong> {inserted + updated}</li>
                                <li><strong>Fehler:</strong> {len(errors)}</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Zeige Fehler falls vorhanden
                        if errors:
                            with st.expander(f"⚠️ {len(errors)} Fehler aufgetreten"):
                                for error in errors[:10]:  # Zeige max 10 Fehler
                                    st.warning(error)
                                if len(errors) > 10:
                                    st.info(f"... und {len(errors) - 10} weitere Fehler")
                        
                        # Cache leeren für Dashboard-Update
                        if hasattr(st, 'cache_data'):
                            st.cache_data.clear()
                        
                        # Erfolg-Nachricht mit Weiterleitung
                        st.balloons()
                        st.info("🎉 Sie können jetzt zu den Auswertungen wechseln!")
                    else:
                        st.warning("⚠️ Keine Daten wurden importiert. Überprüfen Sie die Fehlermeldungen.")
            
            except Exception as e:
                st.error(f"❌ Fehler beim Verarbeiten der Datei: {str(e)}")
                logger.error(f"Import-Fehler: {e}")
    
    def show_employee_analysis(self):
        """Mitarbeiter-spezifische Analyse"""
        st.header("👥 Mitarbeiter-Analyse")
        
        # Lade Daten
        df = self.db_manager.get_time_entries(limit=1000)
        
        if df.empty:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
            return
        
        # Erstelle Analytics-Objekt
        analytics = TimeMotoAnalytics(df)
        
        # Filter
        employees = ['Alle'] + list(analytics.df['username' if 'username' in analytics.df.columns else 'Username'].unique())
        selected_employee = st.selectbox("👤 Mitarbeiter auswählen:", employees)
        
        if selected_employee == 'Alle':
            # Alle Mitarbeiter Übersicht
            summary = analytics.get_employee_summary()
            
            st.subheader("📊 Mitarbeiter-Übersicht")
            st.dataframe(
                summary[['Mitarbeiter', 'Arbeitstage', 'Arbeitszeit_Stunden', 'Saldo_Stunden', 'Abwesenheiten', 'Projekte']],
                use_container_width=True
            )
            
            # Visualisierungen
            col1, col2 = st.columns(2)
            
            with col1:
                # Saldo-Vergleich
                fig = px.bar(
                    summary, 
                    x='Mitarbeiter', 
                    y='Saldo_Stunden',
                    title="⚖️ Saldo-Vergleich alle Mitarbeiter",
                    color='Saldo_Stunden',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Arbeitszeit vs Saldo
                fig = px.scatter(
                    summary,
                    x='Arbeitszeit_Stunden',
                    y='Saldo_Stunden',
                    size='Arbeitstage',
                    hover_name='Mitarbeiter',
                    title="💼 Arbeitszeit vs. Saldo",
                    labels={
                        'Arbeitszeit_Stunden': 'Arbeitszeit (Stunden)',
                        'Saldo_Stunden': 'Saldo (Stunden)'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            # Einzelner Mitarbeiter
            username_col = 'username' if 'username' in analytics.df.columns else 'Username'
            employee_data = analytics.df[analytics.df[username_col] == selected_employee]
            
            # Metriken für den Mitarbeiter
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                work_days = len(employee_data)
                st.metric("📅 Arbeitstage", work_days)
            
            with col2:
                total_hours = round(employee_data['duration_minutes'].sum() / 60, 1)
                st.metric("⏰ Arbeitszeit", f"{total_hours}h")
            
            with col3:
                balance = round(employee_data['balance_minutes'].sum() / 60, 1)
                st.metric("⚖️ Saldo", f"{balance:+.1f}h")
            
            with col4:
                absence_col = 'absence_name' if 'absence_name' in employee_data.columns else 'AbsenceName'
                absences = employee_data[absence_col].notna().sum()
                st.metric("🏥 Abwesenheiten", absences)
            
            # Detailanalyse
            col1, col2 = st.columns(2)
            
            with col1:
                # Täglicher Saldo
                daily_balance = employee_data.groupby('parsed_date')['balance_minutes'].sum() / 60
                
                fig = px.bar(
                    x=daily_balance.index,
                    y=daily_balance.values,
                    title=f"📊 Täglicher Saldo - {selected_employee}",
                    labels={'x': 'Datum', 'y': 'Saldo (Stunden)'}
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Arbeitszeitmuster
                time_patterns = employee_data['time_type'].value_counts()
                
                fig = px.pie(
                    values=time_patterns.values,
                    names=time_patterns.index,
                    title=f"🔍 Arbeitszeitmuster - {selected_employee}"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailtabelle
            st.subheader("📋 Detaildaten")
            display_cols = ['parsed_date', 'start_time', 'end_time', 'duration_excluding_breaks', 'balance', 'work_code', 'absence_name', 'time_type']
            available_cols = [col for col in display_cols if col in employee_data.columns]
            
            if 'start_time' not in employee_data.columns:
                # Fallback für ursprüngliche Spaltennamen
                display_data = employee_data[['parsed_date', 'StartTime', 'EndTime', 'DurationExcludingBreaks', 'Balance', 'work_code', 'AbsenceName', 'time_type']].copy()
                display_data.columns = ['Datum', 'Start', 'Ende', 'Arbeitszeit', 'Saldo', 'Projekt', 'Abwesenheit', 'Typ']
            else:
                display_data = employee_data[available_cols].copy()
            
            st.dataframe(display_data, use_container_width=True)
    
    def show_date_analysis(self):
        """Datums-basierte Analyse"""
        st.header("📅 Datums-Analyse")
        
        # Lade Daten
        df = self.db_manager.get_time_entries(limit=1000)
        
        if df.empty:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
            return
        
        # Erstelle Analytics-Objekt
        analytics = TimeMotoAnalytics(df)
        daily_data = analytics.get_daily_analysis()
        
        if daily_data.empty:
            st.warning("Keine Daten für Datums-Analyse verfügbar.")
            return
        
        # Datumsbereich
        st.info(f"📊 Analyse für Zeitraum: {daily_data['Datum'].min()} bis {daily_data['Datum'].max()}")
        
        # Tagesvergleich
        col1, col2 = st.columns(2)
        
        with col1:
            # Mitarbeiteranzahl pro Tag
            fig = px.bar(
                daily_data,
                x='Datum',
                y='Mitarbeiter_Anzahl',
                title="👥 Anwesende Mitarbeiter pro Tag"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Abwesenheiten pro Tag
            fig = px.bar(
                daily_data,
                x='Datum',
                y='Abwesenheiten',
                title="🏥 Abwesenheiten pro Tag",
                color='Abwesenheiten',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Wochentagsanalyse
        analytics.df['weekday_num'] = analytics.df['parsed_date'].dt.dayofweek
        weekday_data = analytics.df.groupby(['weekday', 'weekday_num']).agg({
            'username' if 'username' in analytics.df.columns else 'Username': 'nunique',
            'balance_minutes': 'sum',
            'duration_minutes': 'sum'
        }).reset_index().sort_values('weekday_num')
        
        weekday_data['balance_hours'] = weekday_data['balance_minutes'] / 60
        weekday_data['duration_hours'] = weekday_data['duration_minutes'] / 60
        
        st.subheader("📊 Wochentagsanalyse")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                weekday_data,
                x='weekday',
                y='duration_hours',
                title="⏰ Durchschnittliche Arbeitszeit pro Wochentag"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                weekday_data,
                x='weekday',
                y='balance_hours',
                title="⚖️ Durchschnittlicher Saldo pro Wochentag",
                color='balance_hours',
                color_continuous_scale='RdYlGn'
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailtabelle
        st.subheader("📋 Tägliche Übersicht")
        st.dataframe(
            daily_data[['Datum', 'Mitarbeiter_Anzahl', 'Gesamt_Arbeitszeit_Std', 'Gesamt_Saldo_Std', 'Abwesenheiten', 'Normale_Erfassungen']],
            use_container_width=True
        )
    
    def show_project_analysis(self):
        """Projekt-basierte Analyse"""
        st.header("🏗️ Projekt-Analyse")
        
        # Lade Daten
        df = self.db_manager.get_time_entries(limit=1000)
        
        if df.empty:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
            return
        
        # Erstelle Analytics-Objekt
        analytics = TimeMotoAnalytics(df)
        work_analysis = analytics.get_work_code_analysis()
        
        if work_analysis.empty:
            st.warning("⚠️ Keine Projekt-/Arbeitscodedaten in den Remarks gefunden.")
            st.info("💡 Stellen Sie sicher, dass Ihre Daten Arbeitscodes in den Bemerkungen enthalten.")
            return
        
        # Projekt-Übersicht
        project_summary = work_analysis.groupby('Projekt').agg({
            'Mitarbeiter': 'nunique',
            'Tage': 'sum',
            'Arbeitszeit_Stunden': 'sum'
        }).reset_index()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_projects = len(project_summary)
            st.metric("🏗️ Projekte", total_projects)
        
        with col2:
            total_project_hours = round(project_summary['Arbeitszeit_Stunden'].sum(), 1)
            st.metric("⏰ Projektzeit", f"{total_project_hours}h")
        
        with col3:
            avg_project_size = round(project_summary['Arbeitszeit_Stunden'].mean(), 1)
            st.metric("📊 Ø Projektgröße", f"{avg_project_size}h")
        
        # Visualisierungen
        col1, col2 = st.columns(2)
        
        with col1:
            # Projektverteilung nach Zeit
            fig = px.pie(
                project_summary,
                values='Arbeitszeit_Stunden',
                names='Projekt',
                title="⏰ Zeitverteilung nach Projekten"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Mitarbeiter pro Projekt
            fig = px.bar(
                project_summary,
                x='Projekt',
                y='Mitarbeiter',
                title="👥 Mitarbeiter pro Projekt"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailanalyse
        st.subheader("📊 Projekt-Details")
        
        selected_project = st.selectbox(
            "🏗️ Projekt auswählen:",
            ['Alle'] + list(project_summary['Projekt'].unique())
        )
        
        if selected_project == 'Alle':
            st.dataframe(
                project_summary[['Projekt', 'Mitarbeiter', 'Tage', 'Arbeitszeit_Stunden']],
                use_container_width=True
            )
        else:
            project_details = work_analysis[work_analysis['Projekt'] == selected_project]
            
            # Mitarbeiterverteilung im Projekt
            fig = px.bar(
                project_details,
                x='Mitarbeiter',
                y='Arbeitszeit_Stunden',
                title=f"👥 Mitarbeiterzeiten - {selected_project}"
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                project_details[['Mitarbeiter', 'Tage', 'Arbeitszeit_Stunden']],
                use_container_width=True
            )
    
    def show_absence_analysis(self):
        """Abwesenheits-Analyse"""
        st.header("🏥 Abwesenheits-Analyse")
        
        # Lade Daten
        df = self.db_manager.get_time_entries(limit=1000)
        
        if df.empty:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
            return
        
        # Erstelle Analytics-Objekt
        analytics = TimeMotoAnalytics(df)
        
        # Abwesenheitsdaten filtern
        absence_col = 'absence_name' if 'absence_name' in analytics.df.columns else 'AbsenceName'
        username_col = 'username' if 'username' in analytics.df.columns else 'Username'
        
        absence_data = analytics.df[analytics.df[absence_col].notna()]
        
        if absence_data.empty:
            st.info("ℹ️ Keine Abwesenheiten in den Daten gefunden.")
            return
        
        # Abwesenheits-Übersicht
        absence_summary = absence_data.groupby(absence_col).agg({
            username_col: 'nunique',
            'parsed_date': 'count'
        }).reset_index()
        
        absence_summary.columns = ['Abwesenheitstyp', 'Mitarbeiter', 'Tage']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_absence_days = absence_summary['Tage'].sum()
            st.metric("📅 Abwesenheitstage", total_absence_days)
        
        with col2:
            affected_employees = absence_data[username_col].nunique()
            st.metric("👥 Betroffene Mitarbeiter", affected_employees)
        
        with col3:
            absence_types = len(absence_summary)
            st.metric("🏥 Abwesenheitstypen", absence_types)
        
        # Visualisierungen
        col1, col2 = st.columns(2)
        
        with col1:
            # Abwesenheitstypen
            fig = px.pie(
                absence_summary,
                values='Tage',
                names='Abwesenheitstyp',
                title="📊 Verteilung der Abwesenheitstypen"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Mitarbeiter mit meisten Abwesenheiten
            employee_absences = absence_data.groupby(username_col)['parsed_date'].count().reset_index()
            employee_absences.columns = ['Mitarbeiter', 'Tage']
            top_absences = employee_absences.nlargest(10, 'Tage')
            
            if not top_absences.empty:
                fig = px.bar(
                    top_absences,
                    x='Tage',
                    y='Mitarbeiter',
                    orientation='h',
                    title="👥 Mitarbeiter mit meisten Abwesenheitstagen"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailanalyse
        st.subheader("📋 Abwesenheits-Details")
        st.dataframe(
            absence_data[[username_col, 'parsed_date', absence_col, 'remarks' if 'remarks' in absence_data.columns else 'Remarks']].rename(columns={
                username_col: 'Mitarbeiter',
                'parsed_date': 'Datum',
                absence_col: 'Abwesenheitstyp',
                'remarks' if 'remarks' in absence_data.columns else 'Remarks': 'Bemerkungen'
            }),
            use_container_width=True
        )
    
    def show_insights(self):
        """KPIs und automatische Insights"""
        st.header("⚡ Insights & KPIs")
        
        # Lade Daten
        df = self.db_manager.get_time_entries(limit=1000)
        
        if df.empty:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
            return
        
        # Erstelle Analytics-Objekt
        analytics = TimeMotoAnalytics(df)
        insights = analytics.generate_insights()
        
        # Automatische Insights
        st.subheader("🎯 Automatische Insights")
        
        if 'top_overtime' in insights:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏆 Mitarbeiter mit meisten Überstunden</h4>
                <p><strong>{insights['top_overtime']['employee']}</strong> hat <strong>{insights['top_overtime']['hours']:+.1f} Stunden</strong> Überstunden</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'most_undertime' in insights:
            st.markdown(f"""
            <div class="warning-box">
                <h4>⚠️ Mitarbeiter mit meisten Minusstunden</h4>
                <p><strong>{insights['most_undertime']['employee']}</strong> hat <strong>{insights['most_undertime']['hours']:+.1f} Stunden</strong> Minusstunden</p>
            </div>
            """, unsafe_allow_html=True)
        
        if 'top_project' in insights:
            st.markdown(f"""
            <div class="insight-box">
                <h4>🏗️ Häufigstes Projekt</h4>
                <p><strong>{insights['top_project']['project']}</strong> wurde <strong>{insights['top_project']['count']} mal</strong> erfasst</p>
            </div>
            """, unsafe_allow_html=True)
        
        # KPI Dashboard
        st.subheader("📊 Key Performance Indicators")
        
        # Berechne KPIs
        username_col = 'username' if 'username' in analytics.df.columns else 'Username'
        total_employees = analytics.df[username_col].nunique()
        total_workdays = analytics.df.groupby(username_col)['parsed_date'].nunique().mean()
        avg_daily_hours = analytics.df.groupby([username_col, 'parsed_date'])['duration_minutes'].sum().mean() / 60
        overtime_ratio = (analytics.df['balance_minutes'] > 0).mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Ø Arbeitstage/MA", f"{total_workdays:.1f}")
        
        with col2:
            st.metric("⏰ Ø Tägliche Arbeitszeit", f"{avg_daily_hours:.1f}h")
        
        with col3:
            st.metric("📈 Überstunden-Quote", f"{overtime_ratio:.1f}%")
        
        with col4:
            absence_col = 'absence_name' if 'absence_name' in analytics.df.columns else 'AbsenceName'
            absence_rate = (analytics.df[absence_col].notna().sum() / len(analytics.df)) * 100
            st.metric("🏥 Abwesenheits-Quote", f"{absence_rate:.1f}%")
        
        # Export aller Insights
        if st.button("📥 Insights-Report exportieren"):
            report_data = {
                'Zeitraum': f"{analytics.df['parsed_date'].min()} bis {analytics.df['parsed_date'].max()}",
                'Mitarbeiter_Gesamt': total_employees,
                'Durchschnittliche_Arbeitstage': round(total_workdays, 1),
                'Durchschnittliche_Tagesarbeitszeit': round(avg_daily_hours, 1),
                'Überstunden_Quote_Prozent': round(overtime_ratio, 1),
                'Abwesenheits_Quote_Prozent': round(absence_rate, 1),
                **insights
            }
            
            import json
            report_json = json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
            
            st.download_button(
                label="💾 JSON-Report herunterladen",
                data=report_json,
                file_name=f"timemoto_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def show_data_view(self):
        """Zeigt die Datenansicht"""
        st.header("📋 Zeiterfassungsdaten")
        
        if not self.test_database_connection():
            return
        
        # Filter-Optionen
        col1, col2 = st.columns(2)
        
        with col1:
            limit = st.selectbox(
                "📊 Anzahl Einträge:",
                [50, 100, 200, 500, 1000],
                index=1
            )
        
        with col2:
            if st.button("🔄 Daten aktualisieren"):
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
        
        # Daten laden
        df = self.db_manager.get_time_entries(limit)
        
        if not df.empty:
            st.info(f"📈 {len(df)} Einträge gefunden")
            
            # Interaktive Tabelle
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
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 CSV herunterladen",
                    data=csv,
                    file_name=f"zeiterfassung_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("📭 Keine Daten gefunden. Importieren Sie zuerst TimeMoto Daten.")
    
    def show_settings(self):
        """Zeigt die Einstellungen"""
        st.header("⚙️ Einstellungen")
        
        # Datenbankstatus
        st.subheader("🗄️ Datenbankverbindung")
        
        success, message, info = self.db_manager.test_database_connection()
        
        if success:
            st.success(message)
            
            # Detaillierte Informationen
            if info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("⏱️ Verbindungszeit", f"{info.get('connection_time', 0)}s")
                    st.metric("📊 Einträge in DB", info.get('entry_count', 0))
                
                with col2:
                    st.metric("🏗️ Tabellen", "✅" if info.get('tables_exist') else "❌")
                    if info.get('last_entry'):
                        last = info['last_entry']
                        st.write(f"**Letzter Eintrag:** {last.get('username')} am {last.get('entry_date')}")
                
                # Vollständige Info
                with st.expander("🔍 Detaillierte Verbindungsinfo"):
                    st.json(info)
        else:
            st.error(message)
        
        # Diagnose-Tools
        st.subheader("🔧 Diagnose-Tools")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🏗️ Tabellen neu erstellen"):
                with st.spinner("Erstelle Tabellen..."):
                    success = self.db_manager.create_tables_with_validation()
                
                if success:
                    st.success("✅ Tabellen erfolgreich erstellt/aktualisiert")
                else:
                    st.error("❌ Fehler beim Erstellen der Tabellen")
        
        with col2:
            if st.button("📊 Statistiken aktualisieren"):
                stats = self.db_manager.get_statistics()
                
                if 'error' not in stats:
                    st.success("✅ Statistiken erfolgreich abgerufen")
                    st.json(stats)
                else:
                    st.error(f"❌ Fehler: {stats['error']}")
        
        # Konfigurationshilfe
        st.subheader("🔧 Konfiguration")
        
        st.markdown("""
        <div class="analytics-section">
            <h4>📝 Neon.tech Konfiguration</h4>
            <p><strong>Option 1 (Empfohlen):</strong> Verwenden Sie DATABASE_URL in <code>secrets.toml</code>:</p>
            <pre>
[secrets]
DATABASE_URL = "postgresql://user:pass@host-pooler.region.aws.neon.tech/db?sslmode=require"
            </pre>
            
            <p><strong>Option 2:</strong> Einzelne Parameter in <code>secrets.toml</code>:</p>
            <pre>
[secrets]
DB_HOST = "your-neon-host-pooler.neon.tech"
DB_PORT = "5432"
DB_NAME = "your_database_name"
DB_USER = "your_username"
DB_PASSWORD = "your_password"
            </pre>
            
            <p><strong>Für Streamlit Cloud:</strong> Setzen Sie die gleichen Werte in den App-Secrets.</p>
            
            <h5>🔍 Troubleshooting:</h5>
            <ul>
                <li>Entfernen Sie <code>channel_binding=require</code> aus der URL</li>
                <li>Verwenden Sie nur <code>sslmode=require</code></li>
                <li>Stellen Sie sicher, dass der Pooler-Endpoint verwendet wird</li>
                <li>Überprüfen Sie die Neon.tech Datenbank-Aktivität</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def test_database_connection(self) -> bool:
        """Testet die Datenbankverbindung"""
        try:
            success, message, info = self.db_manager.test_database_connection()
            if not success:
                st.error(message)
                st.info("💡 Überprüfen Sie Ihre Datenbankverbindung in den Einstellungen.")
            return success
        except Exception as e:
            st.error(f"❌ Fehler beim Testen der Datenbankverbindung: {str(e)}")
            return False

# Anwendung starten
if __name__ == "__main__":
    app = TimeMotoApp()
    app.run()
